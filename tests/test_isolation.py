"""Isolation (#8): a trial gets a worktree, not the machine.

The unit tests pin the argv contract (what is bound, what is hidden, which env
vars pass, whether sockets are allowed) without needing namespaces. The
integration tests run real bwrap when it is installed and skip otherwise.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from dacode.discipline.critic import PstackCritic
from dacode.executor.pi import PiExecutor
from dacode.sandbox import bwrap as bwrap_mod
from dacode.sandbox.bwrap import (
    IsolationPolicy,
    IsolationUnavailable,
    allowed_env,
    interpreter_root,
    wrap_command,
)

HAS_BWRAP = shutil.which("bwrap") is not None
needs_bwrap = pytest.mark.skipif(not HAS_BWRAP, reason="bubblewrap not installed")


def _argv(**kwargs) -> list[str]:
    policy = kwargs.pop("policy", IsolationPolicy(enabled=True))
    return wrap_command(kwargs.pop("command", ["bash", "-lc", "true"]),
                        workspace=kwargs.pop("workspace", Path.cwd()), policy=policy, **kwargs)


# --- argv contract ---------------------------------------------------------

def test_disabled_policy_returns_the_command_unchanged():
    cmd = ["python", "-m", "pytest"]
    assert wrap_command(cmd, workspace=Path.cwd(), policy=IsolationPolicy(enabled=False)) == cmd


def test_missing_bwrap_raises_instead_of_running_unsandboxed(monkeypatch):
    monkeypatch.setattr(bwrap_mod, "available", lambda: False)
    with pytest.raises(IsolationUnavailable):
        wrap_command(["true"], workspace=Path.cwd(), policy=IsolationPolicy(enabled=True))


def test_network_is_unshared_by_default():
    assert "--unshare-net" in _argv()


def test_network_can_be_allowed():
    assert "--unshare-net" not in _argv(policy=IsolationPolicy(enabled=True, network=True))


def test_the_workspace_is_the_only_writable_place():
    argv = _argv(workspace=Path.cwd())
    index = argv.index("--bind")
    assert argv[index + 1] == argv[index + 2] == str(Path.cwd().resolve())


def test_read_only_workspace_binds_read_only(tmp_path):
    # The baseline measures the live main checkout: its bind must not be the
    # writable one a trial workspace gets.
    ws = tmp_path / "repo"
    ws.mkdir()
    argv = _argv(workspace=ws, read_only_workspace=True)
    assert ["--ro-bind", str(ws.resolve()), str(ws.resolve())] in [
        argv[i:i + 3] for i in range(len(argv) - 2)
    ]
    assert ["--bind", str(ws.resolve()), str(ws.resolve())] not in [
        argv[i:i + 3] for i in range(len(argv) - 2)
    ]


def test_exposes_home_flags_the_workspace_that_would_re_expose_home(tmp_path, monkeypatch):
    from dacode.sandbox.bwrap import exposes_home

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path(home)))

    (home / ".git").mkdir()                    # $HOME made into a checkout
    assert exposes_home(home) is True
    outside = tmp_path / "repos" / "plain"
    outside.mkdir(parents=True)
    assert exposes_home(outside) is False
    assert exposes_home(tmp_path) is True      # an ancestor of $HOME contains it


def test_home_is_replaced_by_an_empty_tmpfs():
    argv = _argv(home=Path.home())
    index = argv.index("--tmpfs")
    assert argv[index + 1] == str(Path.home().resolve())


def test_read_only_paths_are_bound_when_they_exist(tmp_path):
    real = tmp_path / "venv"
    real.mkdir()
    missing = tmp_path / "absent"
    argv = _argv(policy=IsolationPolicy(enabled=True, read_only_paths=(real, missing)))
    assert str(real) in argv
    assert str(missing) not in argv


def test_environment_is_cleared_and_allow_listed(monkeypatch):
    argv = _argv()
    assert "--clearenv" in argv
    env = {"PATH": "/usr/bin", "OPENROUTER_API_KEY": "secret", "AWS_SECRET_ACCESS_KEY": "nope"}
    names = [argv[i + 1] for i, token in enumerate(argv) if token == "--setenv"]
    assert "PATH" in names
    assert "OPENROUTER_API_KEY" not in names       # network is off
    assert "AWS_SECRET_ACCESS_KEY" not in names    # never allow-listed


def test_provider_keys_pass_only_with_network():
    env = {"OPENROUTER_API_KEY": "secret"}
    offline = allowed_env(IsolationPolicy(enabled=True, network=False), env)
    online = allowed_env(IsolationPolicy(enabled=True, network=True), env)
    assert "OPENROUTER_API_KEY" not in offline
    assert online["OPENROUTER_API_KEY"] == "secret"


def test_the_working_directory_is_the_workspace():
    argv = _argv(workspace=Path.cwd())
    index = argv.index("--chdir")
    assert argv[index + 1] == str(Path.cwd().resolve())


def test_interpreter_root_finds_an_absolute_python():
    root = interpreter_root(("/opt/venv/bin/python", "-m", "pytest"))
    assert root == Path("/opt/venv/bin")
    assert interpreter_root(("python", "-m", "pytest")) is None


# --- components refuse to pretend ------------------------------------------

def test_provider_env_aliases_the_zai_key_for_pi(monkeypatch):
    """DaCode's registry names the z.ai key GLM_API_KEY; Pi's provider layer
    reads ZAI_API_KEY for the same provider id (verified: Pi 1.4 exits with
    "No API key found for zai" without the alias)."""
    from dacode.executor.pi import provider_env

    monkeypatch.delenv("GLM_API_KEY", raising=False)
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    assert provider_env("zai", {"GLM_API_KEY": "sk-one"}) == {"GLM_API_KEY": "sk-one", "ZAI_API_KEY": "sk-one"}
    # an existing ZAI_API_KEY always wins
    assert provider_env("zai", {"GLM_API_KEY": "a", "ZAI_API_KEY": "b"})["ZAI_API_KEY"] == "b"
    # nothing is invented for other providers or a missing key
    assert provider_env("openrouter", {"GLM_API_KEY": "a"}) == {"GLM_API_KEY": "a"}
    assert provider_env("zai", {}) == {}
    assert provider_env(None, {"GLM_API_KEY": "a"}) == {"GLM_API_KEY": "a"}


def test_pi_executor_passes_the_aliased_env_to_pi(tmp_path, monkeypatch):
    """run() must hand the alias to the real subprocess, not just compute it."""
    import types

    from dacode.executor.pi import provider_env

    seen = {}
    real_run = subprocess.run

    def fake_run(cmd, **kwargs):
        seen.update(kwargs.get("env") or {})
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("dacode.executor.pi.subprocess.run", fake_run)
    monkeypatch.setattr("dacode.executor.pi.shutil.which", lambda cmd: f"/usr/bin/{cmd}")
    monkeypatch.setenv("GLM_API_KEY", "sk-alias-test")
    monkeypatch.delenv("ZAI_API_KEY", raising=False)

    executor = PiExecutor(provider="zai", isolation=None)
    result = executor.run(_variant(), tmp_path)

    assert result.exit_code == 0
    assert seen.get("ZAI_API_KEY") == "sk-alias-test"
    assert provider_env("zai", {"GLM_API_KEY": "sk-alias-test"})["ZAI_API_KEY"] == "sk-alias-test"


def test_sandbox_policy_binds_the_node_runtime(tmp_path, monkeypatch):
    """Node installed under $HOME (nvm/fnm/volta) is hidden by the home tmpfs;
    the policy must bind the runtime the way it binds the agent binary, or
    `env node` falls through to an older system interpreter."""
    fake_node = tmp_path / "node-22" / "bin" / "node"
    fake_node.parent.mkdir(parents=True)
    fake_node.write_text("#!/bin/sh\n")
    fake_pi = tmp_path / "pi-agent" / "bin" / "pi"
    fake_pi.parent.mkdir(parents=True)
    fake_pi.write_text("#!/bin/sh\n")

    monkeypatch.setattr(
        "dacode.executor.pi.shutil.which",
        lambda cmd: str(fake_node) if cmd == "node" else (str(fake_pi) if cmd == "pi" else None),
    )
    monkeypatch.delenv("NODE_PATH", raising=False)

    executor = PiExecutor(isolation=IsolationPolicy(enabled=True))
    policy = executor._sandbox_policy(str(fake_pi))

    assert Path(fake_node).resolve() in policy.read_only_paths
    assert fake_node.resolve().parent in policy.read_only_paths


def test_pi_executor_reports_a_missing_sandbox_instead_of_running(monkeypatch, tmp_path):
    monkeypatch.setattr(bwrap_mod, "available", lambda: False)
    monkeypatch.setattr("dacode.executor.pi.shutil.which", lambda cmd: f"/usr/bin/{cmd}")
    executor = PiExecutor(isolation=IsolationPolicy(enabled=True))
    result = executor.run(_variant(), tmp_path)
    assert result.exit_code == 126
    assert "isolation unavailable" in result.stderr


def test_critic_fails_the_gate_when_the_sandbox_disappears(monkeypatch, tmp_path):
    (tmp_path / "test_ok.py").write_text("def test_a():\n    assert True\n")
    monkeypatch.setattr(bwrap_mod, "available", lambda: False)
    critic = PstackCritic(
        test_command=(sys.executable, "-m", "pytest", "-q"),
        isolation=IsolationPolicy(enabled=True),
    )
    report = critic.review_workspace(tmp_path)
    assert report.exit_code == 125
    assert report.passed is False


def _variant():
    from dacode.harness.models import HarnessVariant

    return HarnessVariant(payload={"task": "x"})


# --- real namespaces -------------------------------------------------------

@needs_bwrap
def test_a_sandboxed_command_cannot_read_the_home_directory(tmp_path):
    secret = Path.home() / ".dacode-isolation-probe"
    secret.write_text("host-only")
    try:
        argv = wrap_command(
            ["bash", "-lc", f"cat {secret}"],
            workspace=tmp_path,
            policy=IsolationPolicy(enabled=True),
        )
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        assert "host-only" not in proc.stdout
    finally:
        secret.unlink(missing_ok=True)


@needs_bwrap
def test_a_sandboxed_command_can_write_inside_the_workspace(tmp_path):
    argv = wrap_command(
        ["bash", "-lc", "echo inside > made.txt"],
        workspace=tmp_path,
        policy=IsolationPolicy(enabled=True),
    )
    subprocess.run(argv, capture_output=True, text=True, timeout=60, check=True)
    assert (tmp_path / "made.txt").read_text().strip() == "inside"


@needs_bwrap
def test_a_sandboxed_command_cannot_open_a_socket(tmp_path):
    argv = wrap_command(
        [sys.executable, "-c",
         "import socket; socket.create_connection(('1.1.1.1', 53), timeout=3); print('CONNECTED')"],
        workspace=tmp_path,
        policy=IsolationPolicy(enabled=True),
    )
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    assert "CONNECTED" not in proc.stdout


@needs_bwrap
def test_the_gate_can_run_a_real_suite_inside_the_sandbox(tmp_path):
    (tmp_path / "test_inner.py").write_text("def test_ok():\n    assert 1 + 1 == 2\n")
    critic = PstackCritic(
        test_command=(sys.executable, "-m", "pytest", "-q"),
        isolation=IsolationPolicy(enabled=True),
    )
    report = critic.review_workspace(tmp_path)
    assert report.exit_code == 0, report.findings
    assert (report.tests_total, report.tests_passed) == (1, 1)


def test_the_gate_never_gets_the_network_even_when_the_executor_does(rig_repo):
    """Found by an external audit: one shared policy gave the gate sockets + keys."""
    from dacode.config import DaCodeConfig
    from dacode.orchestrator import DaCodeRig

    cfg = DaCodeConfig(repo=rig_repo, isolate="bwrap", isolate_network=True)
    rig = DaCodeRig(cfg)

    assert rig.isolation is not None and rig.isolation.network is True
    assert rig.critic.isolation is not None
    assert rig.critic.isolation.network is False
    # and the environment the gate sees carries no provider keys
    from dacode.sandbox.bwrap import allowed_env

    assert "OPENROUTER_API_KEY" not in allowed_env(rig.critic.isolation, {"OPENROUTER_API_KEY": "x"})


# --- audit regressions -----------------------------------------------------

@needs_bwrap
def test_a_workspace_under_tmp_still_works(tmp_path):
    """Audit H1: --tmpfs /tmp came after the workspace bind and broke /tmp checkouts."""
    ws = Path("/tmp") / f"dacode-ws-{tmp_path.name}"
    ws.mkdir(parents=True, exist_ok=True)
    try:
        argv = wrap_command(
            ["bash", "-c", "echo ok; echo TMPDIR=$TMPDIR"],
            workspace=ws,
            policy=IsolationPolicy(enabled=True),
        )
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, proc.stderr
        assert "TMPDIR=/tmp" in proc.stdout
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def test_binding_home_or_its_ancestor_is_refused(tmp_path):
    """Audit C5a: a symlink two levels under $HOME used to re-mount the whole home."""
    from dacode.sandbox.bwrap import _bind_candidates

    home = Path.home().resolve()
    assert _bind_candidates(home) == []
    assert _bind_candidates(home.parent) == []
    keepable = tmp_path / "venv"
    keepable.mkdir()
    assert _bind_candidates(keepable) == [keepable]


def test_unknown_isolate_values_are_rejected(rig_repo):
    from dacode.config import DaCodeConfig
    from dacode.orchestrator import DaCodeRig

    rig = DaCodeRig.__new__(DaCodeRig)
    rig.config = DaCodeConfig(repo=rig_repo, isolate="false")
    assert DaCodeRig._isolation_policy(rig) is None

    rig.config = DaCodeConfig(repo=rig_repo, isolate="sandcastle")
    with pytest.raises(RuntimeError, match="unknown --isolate value"):
        DaCodeRig._isolation_policy(rig)


def test_credentials_directories_under_home_are_refused(tmp_path):
    """Verification audit: `$HOME` itself was refused, its secret subdirs were not."""
    from dacode.sandbox.bwrap import _bind_candidates

    home = Path.home().resolve()
    for name in (".ssh", ".gnupg", ".aws"):
        candidate = home / name
        if candidate.exists():
            assert _bind_candidates(candidate) == [], f"{name} must never be bound"
    for name in (".hermes/.env", ".hermes/config.yaml"):
        candidate = home / name
        if candidate.exists():
            assert _bind_candidates(candidate) == [], f"{name} must never be bound"
    # interpreters and venvs under $HOME stay bindable, or the sandbox cannot start
    for name in (".hermes/tools", ".local"):
        candidate = home / name
        if candidate.exists():
            assert _bind_candidates(candidate), f"{name} must stay bindable"


def test_a_symlink_under_a_home_top_level_directory_does_not_expose_it(tmp_path, monkeypatch):
    """Audit C4: binding real.parent.parent exposed all of ~/.local."""
    from dacode.sandbox.bwrap import _bind_candidates

    home = tmp_path / "home"
    (home / ".local" / "bin").mkdir(parents=True)
    (home / ".local" / "state").mkdir(parents=True)
    real = home / ".local" / "bin" / "tool-real"
    real.write_text("#!/bin/sh\n")
    link = home / ".local" / "bin" / "tool"
    link.symlink_to(real)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    binds = _bind_candidates(link)

    assert (home / ".local") not in binds
    assert binds


# --- the writable ~/.pi mount must not expose the agent's binary (Oct 2026) --

def _bind_positions(argv: list[str], target: Path) -> list[int]:
    """Indexes in ``argv`` where ``target`` is bound (ro or rw)."""
    text = str(target)
    return [i + 1 for i, flag in enumerate(argv) if flag in ("--bind", "--ro-bind")
            and i + 1 < len(argv) and argv[i + 1] == text]


def test_a_protected_path_is_bound_after_the_writable_parent(tmp_path):
    """Later mounts shadow earlier ones, so the protective ro-bind must come last."""
    home = tmp_path / "home"
    bindir = home / ".pi" / "agent" / "bin"
    bindir.mkdir(parents=True)
    (tmp_path / "ws").mkdir()
    argv = wrap_command(
        ["bash", "-c", "true"],
        workspace=tmp_path / "ws",
        policy=IsolationPolicy(
            enabled=True,
            read_only_paths=(bindir,),
            writable_paths=(home,),
            protected_paths=(bindir,),
        ),
        home=home,
    )
    ro_after = _bind_positions(argv, bindir)
    # bound read-only twice: once with the other read_only_paths, once last
    assert len(ro_after) == 2
    writable = _bind_positions(argv, home)
    assert writable and max(writable) < max(ro_after), (
        "the protected re-bind must be emitted after the writable mount, "
        "or the writable mount shadows it and the child stays writable"
    )


@needs_bwrap
def test_a_trial_cannot_replace_the_agent_binary_under_a_writable_home(tmp_path):
    """The executor policy makes ~/.pi writable for pi's lock file; the binary
    directory must survive that as read-only *and* on the host afterwards."""
    home = tmp_path / "home"
    bindir = home / ".pi" / "agent" / "bin"
    bindir.mkdir(parents=True)
    binary = bindir / "pi"
    binary.write_text("#!/bin/sh\necho real-pi\n")
    binary.chmod(0o755)
    ws = tmp_path / "ws"
    ws.mkdir()
    policy = IsolationPolicy(
        enabled=True,
        read_only_paths=(bindir,),
        writable_paths=(home,),
        protected_paths=(bindir,),
    )
    argv = wrap_command(
        ["bash", "-c",
         f"cat {binary}"
         f" && {{ echo trojan > {binary} || echo WRITE-BLOCKED; }}"
         f" && cat {binary}"
         f" && echo trial-lock > {home / '.pi' / 'agent' / 'session.lock'} && echo LOCK-OK"],
        workspace=ws,
        policy=policy,
        home=home,
    )
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    assert "trojan" not in proc.stdout
    # the binary was readable before and after the blocked write...
    assert proc.stdout.count("real-pi") == 2, proc.stdout
    # ...the lock file the writable mount exists for still works...
    assert "LOCK-OK" in proc.stdout
    assert (home / ".pi" / "agent" / "session.lock").read_text().strip() == "trial-lock"
    # ...and the host copy is exactly what it was before the run
    assert binary.read_text() == "#!/bin/sh\necho real-pi\n"


def test_the_executor_policy_protects_the_pi_binary_directory(tmp_path, monkeypatch):
    """PiExecutor.run must re-bind the resolved binary's directory read-only on
    top of the writable ~/.pi mount (persistence audit, Oct 2026)."""
    import dacode.executor.pi as pi_mod
    from dacode.executor.pi import PiExecutor

    home = tmp_path / "home"
    fake_pi = home / ".pi" / "agent" / "bin" / "pi"
    fake_pi.parent.mkdir(parents=True)
    fake_pi.write_text("#!/bin/sh\n")
    fake_pi.chmod(0o755)
    monkeypatch.setattr(pi_mod.shutil, "which", lambda name: str(fake_pi))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    captured: dict = {}

    def fake_wrap(command, *, workspace, policy, **kwargs):
        captured["policy"] = policy
        return ["true"]

    monkeypatch.setattr(pi_mod, "wrap_command", fake_wrap)
    executor = PiExecutor(isolation=IsolationPolicy(enabled=True))
    (tmp_path / "ws").mkdir()
    result = executor.run(_variant(), tmp_path / "ws")
    assert result.exit_code == 0
    policy = captured["policy"]
    assert fake_pi.parent in policy.protected_paths
    assert home / ".pi" in policy.writable_paths
