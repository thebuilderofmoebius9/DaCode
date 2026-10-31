"""bwrap isolation for trials — the worktree protects the repo, this protects the host.

A trial runs model-written code. The git worktree gives it a private branch, not
a private machine: without isolation it can read `~/.ssh`, write outside the
checkout, and talk to anything. Bubblewrap gives us namespaces without a daemon
and without root.

Design choices:

- **Read-only root, private home.** The whole filesystem is mounted read-only and
  the user's home is replaced by an empty tmpfs, then the trial workspace is bound
  read-write on top. Paths the executor legitimately needs (a venv, the agent's
  own config) are re-bound explicitly, read-only.
- **Network off by default.** A test suite has no business opening sockets. The Pi
  executor *does* need the network to reach a model, so the caller opts in with
  ``network=True``; the critic never does.
- **Cleared environment, explicit allow-list.** Provider keys are passed only when
  asked for, so a stray `AWS_SECRET_ACCESS_KEY` in the shell cannot leak into a
  trial by accident.

Fail closed: if bwrap is missing, ``wrap`` raises ``IsolationUnavailable`` and the
CLI says so, rather than silently running unsandboxed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

__all__ = [
    "IsolationPolicy",
    "IsolationUnavailable",
    "available",
    "bwrap_version",
    "wrap_command",
]

#: Variables a trial normally needs to function at all.
BASE_ENV = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TMPDIR", "TERM")

#: Passed only when the caller opts in with ``network=True`` (an agent needs its key).
NETWORK_ENV = (
    "OPENROUTER_API_KEY",
    "GLM_API_KEY",
    "ZAI_API_KEY",  # Pi's own name for the z.ai key — see executor/pi.py
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "DEEPSEEK_API_KEY",
    "MISTRAL_API_KEY",
    "XAI_API_KEY",
)


class IsolationUnavailable(RuntimeError):
    """bwrap is not installed (or cannot create namespaces) on this host."""


@dataclass(frozen=True)
class IsolationPolicy:
    """How much of the host a trial may see."""

    enabled: bool = True
    network: bool = False
    #: Extra paths bound read-only (a venv, the agent's config directory…).
    read_only_paths: tuple[Path, ...] = field(default_factory=tuple)
    #: Extra environment variable names to pass through.
    env_allow: tuple[str, ...] = field(default_factory=tuple)
    #: Files replaced by /dev/null inside the sandbox (a repository's own .env, an
    #: agent's auth file): readable parent, invisible file.
    hidden_files: tuple[Path, ...] = field(default_factory=tuple)
    #: Extra directories bound read-write (the gate's JUnit report directory).
    writable_paths: tuple[Path, ...] = field(default_factory=tuple)
    #: Paths re-bound read-only *after* the writable binds. Later mounts shadow
    #: earlier ones, so a read-only entry in ``read_only_paths`` that lives under
    #: a ``writable_paths`` parent is silently shadowed — an executor policy that
    #: makes ``~/.pi`` writable for the agent's lock file would otherwise leave
    #: the agent's own binary directory writable, and a trial could replace the
    #: executable the host runs next (follow-up audit, Oct 2026). Anything listed
    #: here wins over the writable ancestor above it.
    protected_paths: tuple[Path, ...] = field(default_factory=tuple)

    def with_paths(self, *paths: str | Path | None) -> "IsolationPolicy":
        """Return a copy with more read-only bindings (None entries are ignored)."""
        extra = tuple(Path(p).expanduser() for p in paths if p)
        merged: list[Path] = list(self.read_only_paths)
        for path in extra:
            if path not in merged:
                merged.append(path)
        return IsolationPolicy(
            enabled=self.enabled,
            network=self.network,
            read_only_paths=tuple(merged),
            env_allow=self.env_allow,
            hidden_files=self.hidden_files,
            writable_paths=self.writable_paths,
            protected_paths=self.protected_paths,
        )


_AVAILABLE: bool | None = None
_EMPTY_MASK: str | None = None


def _empty_mask_file() -> str:
    """Return a reusable 0-byte file on the host for masking hidden_files."""
    import atexit

    global _EMPTY_MASK
    if _EMPTY_MASK is not None and os.path.isfile(_EMPTY_MASK):
        return _EMPTY_MASK
    fd, path = tempfile.mkstemp(prefix="dacode-empty-mask-")
    os.close(fd)
    _EMPTY_MASK = path
    atexit.register(lambda p=path: Path(p).unlink(missing_ok=True))
    return path


def available(*, probe: bool = True) -> bool:
    """True when bwrap is present *and* can actually create namespaces.

    Being on PATH is not enough: Ubuntu 24.04 restricts unprivileged user
    namespaces (AppArmor, ``max_user_namespaces=0``) and containers often block
    them, in which case every wrapped command would fail with bwrap's own exit
    code 1 instead of failing closed (external audit, Oct 2026). The probe runs
    once and is cached.
    """
    global _AVAILABLE
    if _AVAILABLE is not None:
        return _AVAILABLE
    binary = shutil.which("bwrap")
    if binary is None:
        _AVAILABLE = False
        return False
    if not probe:
        _AVAILABLE = True
        return True
    try:
        proc = subprocess.run(
            [binary, "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev",
             "--unshare-pid", "--die-with-parent", "/bin/true"],
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        _AVAILABLE = False
        return False
    _AVAILABLE = proc.returncode == 0
    return _AVAILABLE


def bwrap_version() -> str | None:
    if not available():
        return None
    try:
        import subprocess

        proc = subprocess.run(["bwrap", "--version"], capture_output=True, text=True, timeout=10)
    except OSError:
        return None
    return (proc.stdout or proc.stderr).strip() or None


def allowed_env(
    policy: IsolationPolicy,
    env: dict[str, str] | None = None,
) -> dict[str, str]:
    """The environment a trial may see: base + allow-list, and network keys only if permitted."""
    source = dict(os.environ if env is None else env)
    names = list(BASE_ENV) + list(policy.env_allow)
    if policy.network:
        names += list(NETWORK_ENV)
    return {name: source[name] for name in dict.fromkeys(names) if name in source}


def exposes_home(path: str | Path, home: str | Path | None = None) -> bool:
    """Would mounting ``path`` re-expose the home directory?

    True when the path *is* ``$HOME``, an ancestor of it, or a directory/file
    under it that holds credentials (see :func:`_is_home_secret`). Bind targets
    that pass this check are safe to expose read-write; the rest must stay out.
    """
    home_path = Path(home).expanduser().resolve() if home is not None else Path.home().resolve()
    try:
        resolved = Path(path).expanduser().resolve()
    except OSError:
        return True  # unreadable/unresolvable path: fail closed
    if resolved == home_path or home_path.is_relative_to(resolved):
        return True
    return _is_home_secret(resolved, home_path)


def wrap_command(
    command: Sequence[str],
    *,
    workspace: str | Path,
    policy: IsolationPolicy,
    env: dict[str, str] | None = None,
    home: str | Path | None = None,
    read_only_workspace: bool = False,
) -> list[str]:
    """Prefix ``command`` with a bwrap invocation implementing ``policy``."""
    if not policy.enabled:
        return list(command)
    if not available():
        raise IsolationUnavailable(
            "bwrap is not installed — install bubblewrap or run with isolation disabled"
        )

    workspace_path = Path(workspace).expanduser().resolve()
    if not workspace_path.is_dir():
        raise FileNotFoundError(f"workspace does not exist: {workspace_path}")

    home_path = Path(home).expanduser().resolve() if home is not None else Path.home().resolve()

    argv: list[str] = [
        "bwrap",
        "--die-with-parent",
        "--new-session",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-cgroup-try",
        "--clearenv",
        # Read-only host, then hide the home directory, then expose what is needed.
        "--ro-bind", "/", "/",
    ]
    if home_path.is_dir():
        argv += ["--tmpfs", str(home_path)]
    if not policy.network:
        argv += ["--unshare-net"]

    # /proc, /dev and /tmp come before any bind: a workspace under /tmp would
    # otherwise be mounted over by the tmpfs and bwrap would fail to chdir.
    argv += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    # Host sockets live under /run (/run/user/$UID/bus, ssh-agent, docker.sock) and
    # /var/tmp keeps other processes' droppings; neither belongs in a trial.
    argv += ["--tmpfs", "/run", "--tmpfs", "/var/tmp"]

    # A networked trial needs name resolution. On systemd-resolved hosts
    # /etc/resolv.conf is a symlink into /run — which the sandbox just replaced
    # with an empty tmpfs — so the name dangles and DNS dies. Bind the real
    # directory back over the tmpfs (a file bind onto the dangling symlink name
    # fails: bwrap cannot create through it). Regular-file hosts need nothing.
    if policy.network:
        resolv = Path("/etc/resolv.conf")
        try:
            target = resolv.resolve(strict=True)
        except OSError:
            target = resolv
        if target.is_file() and target.parent != Path("/etc"):
            argv += ["--ro-bind", str(target.parent), str(target.parent)]

    for path in policy.read_only_paths:
        for candidate in _bind_candidates(Path(path).expanduser(), home=home_path):
            argv += ["--ro-bind", str(candidate), str(candidate)]

    # Writable binds come before the masks: a hidden file inside a writable
    # directory (the agent's auth.json under a writable ~/.pi, or <repo>/.env
    # inside workspace_path during measure_baseline) must stay hidden.
    for extra in policy.writable_paths:
        target = Path(extra).expanduser().resolve()
        if target.is_dir():
            argv += ["--bind", str(target), str(target)]

    # A read-only workspace bind is for runs that only measure (the untouched
    # repo's baseline): the gate's sandbox must not be able to write the main
    # checkout. Trial workspaces stay read-write — that is where trials commit.
    bind_flag = "--ro-bind" if read_only_workspace else "--bind"
    argv += [bind_flag, str(workspace_path), str(workspace_path)]

    # Protected re-binds come after the writable binds (including workspace_path)
    # so they shadow them: a read-only mount on a child of a writable parent is
    # the only way the child actually stays read-only (later mounts win).
    for extra in policy.protected_paths:
        target = Path(extra).expanduser().resolve()
        if target.exists():
            argv += ["--ro-bind", str(target), str(target)]

    # Files the policy hides are replaced by an empty regular file, so a
    # repository's own .env (or an agent's auth file) reads back as empty even
    # though its parent is mounted. /dev/null cannot be used: a bind of the
    # character device denies reads outright (EACCES), which readers treat as a
    # hard error instead of "no secrets here". Masks are emitted last on purpose
    # — later mounts shadow earlier ones.
    if any(
        Path(h).expanduser().exists() and not Path(h).expanduser().is_dir()
        for h in policy.hidden_files
    ):
        empty = _empty_mask_file()
        for hidden in policy.hidden_files:
            target = Path(hidden).expanduser()
            if target.exists() and not target.is_dir():
                argv += ["--ro-bind", empty, str(target.resolve())]

    argv += ["--chdir", str(workspace_path)]

    environment = allowed_env(policy, env)
    # TMPDIR from the host may point inside the home tmpfs; inside the sandbox the
    # only temp directory that exists is /tmp.
    environment["TMPDIR"] = "/tmp"
    for name, value in environment.items():
        argv += ["--setenv", name, value]
    argv += ["--", *command]
    return argv


#: Credentials under $HOME that are never bound into a sandbox, whatever a policy
#: asks for. Narrow on purpose: interpreters and venvs often live under
#: ``~/.hermes/tools`` or ``~/.local``, so a whole-directory deny for those would
#: make the sandbox unable to start. ``.pi`` is absent because the agent executor
#: has to authenticate — that is a known open item in SECURITY.md.
HOME_SECRET_DIRS: tuple[str, ...] = (
    ".ssh", ".gnupg", ".aws", ".kube", ".docker", ".password-store", ".mozilla",
)
HOME_SECRET_FILES: tuple[str, ...] = (
    ".git-credentials", ".netrc", ".npmrc", ".pypirc", ".pgpass", ".my.cnf",
)
HOME_SECRET_RELATIVE: tuple[str, ...] = (
    ".hermes/.env", ".hermes/config.yaml", ".hermes/google_token.json",
    ".config/gcloud", ".config/gh", ".docker/config.json",
)


def _is_home_secret(path: Path, home: Path) -> bool:
    """Is this a credential path under $HOME (or a directory holding some)?"""
    try:
        relative = path.resolve().relative_to(home)
    except (OSError, ValueError):
        return False
    if not relative.parts:
        return False
    if relative.parts[0] in HOME_SECRET_DIRS or relative.name in HOME_SECRET_FILES:
        return True
    text = str(relative)
    return any(text == entry or text.startswith(entry + "/") for entry in HOME_SECRET_RELATIVE)


def _bind_candidates(path: Path, *, home: Path | None = None) -> list[Path]:
    """A path plus what it really points at.

    A venv's ``bin/python`` is usually a symlink into the base interpreter, which
    may live under the home directory we just emptied — binding only the symlink
    leaves a dangling target and every command dies with
    ``execvp …: No such file or directory``. Bind the target (and its prefix) too.
    """
    home_path = home or Path.home().resolve()
    out: list[Path] = []

    for candidate in (path,):
        if candidate.exists() and not exposes_home(candidate, home=home_path):
            out.append(candidate)
    try:
        real = path.resolve()
    except OSError:
        real = None
    if real is not None and real != path:
        if real.exists() and not exposes_home(real, home=home_path):
            out.append(real)
        prefix = real.parent.parent  # <prefix>/bin/python → <prefix>
        if prefix.exists() and not exposes_home(prefix, home=home_path):
            # Never bind a top-level directory of $HOME (~/.local, ~/.hermes) just
            # because a symlink resolved inside it: that exposes the state store and
            # anything else kept there. Bind the immediate parent instead.
            if prefix.parent == home_path:
                out.append(real.parent)
            else:
                out.append(prefix)
    return list(dict.fromkeys(out))


def describe(policy: IsolationPolicy) -> str:
    """One line for `dacode layers` and run summaries."""
    if not policy.enabled:
        return "off (trial code has host access)"
    bits = [f"bwrap {bwrap_version() or '?'}", "net:on" if policy.network else "net:off"]
    if policy.read_only_paths:
        bits.append("ro:" + ",".join(p.name for p in policy.read_only_paths))
    return " · ".join(bits)


def interpreter_binds(command: Iterable[str]) -> tuple[Path, ...]:
    """Everything the sandbox needs to start the test interpreter.

    A venv under the home directory disappears with the home tmpfs, and its
    ``bin/python`` is a symlink into a base install that may also live under home.
    Bind, in order: the interpreter itself, its venv root (detected by
    ``pyvenv.cfg``), the base prefix named in that file, and the resolved target.
    """
    python = _python_token(command)
    if python is None or not python.is_absolute():
        return ()
    candidates: list[Path] = [python, python.parent]
    venv_root = python.parent.parent
    cfg = venv_root / "pyvenv.cfg"
    if cfg.is_file():
        candidates.append(venv_root)
        try:
            for line in cfg.read_text(errors="ignore").splitlines():
                if line.lower().startswith("home"):
                    base_bin = Path(line.split("=", 1)[1].strip())
                    candidates.append(base_bin)
                    candidates.append(base_bin.parent)
        except OSError:
            pass
    return tuple(dict.fromkeys(p for p in candidates if p.exists()))


def _python_token(command: Iterable[str]) -> Path | None:
    for token in command:
        text = str(token)
        if text.endswith("python") or text.endswith("python3") or "/python" in text:
            return Path(text)
    return None


def interpreter_root(command: Iterable[str]) -> Path | None:
    """The directory holding the interpreter in a test command, if it is an absolute path."""
    for token in command:
        if token.endswith("python") or "/python" in token or token.endswith("python3"):
            path = Path(token)
            if path.is_absolute():
                return path.parent
    return None
