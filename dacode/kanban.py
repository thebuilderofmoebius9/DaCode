"""Kanban worker — DaCode as one worker on a Hermes Kanban board.

The worker takes one card, claims it atomically (a lost race costs nothing),
reads the gate command from the card body's fenced `````gate` block, runs the
card through the rig, and reports back: ``complete`` with evidence when the
gate passes, ``block`` with a readable reason when it does not.

This is an *optional* integration: the board is any ``hermes`` binary speaking
``hermes kanban``. Defaults are overridable:

* ``DACODE_KANBAN_BIN``   — the CLI binary (default ``hermes``)
* ``DACODE_KANBAN_BOARD`` — the board name (``--board`` is omitted when unset)

Every string that crosses to the board goes through :func:`dacode.redact.redact`
— a card comment is as persistent as a log line.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from dacode import runlog
from dacode.redact import redact, redact_event
from dacode.sandbox import GitSandbox
from dacode.sandbox.bwrap import exposes_home

__all__ = [
    "KanbanError",
    "parse_gate",
    "CardOutcome",
    "HermesKanban",
    "work",
    "changed_files",
    "make_rig_runner",
]


class KanbanError(RuntimeError):
    """The board refused a verb (non-zero exit) — the caller decides what that means."""


DEFAULT_BINARY = "hermes"
DEFAULT_TIMEOUT = 60  # seconds per board call

#: Runner signature: ``runner(argv) -> object with returncode/stdout/stderr``.
Runner = Callable[[list[str]], Any]

#: Output markers that mean *contention* — someone else holds the claim. Any
#: other failure (typo'd id, unknown board, broken CLI) must not be reported
#: as "another worker owns the card"; it raises instead.
_CLAIM_HELD_MARKERS = ("status=running", "lock=")


def parse_gate(body: str | None) -> str | None:
    """The stripped content of the FIRST fenced block info-tagged exactly ``gate``.

    Multi-line commands survive intact (only the whole block is stripped). A
    plain ```` ```bash ```` block is not a gate; neither is an empty one. Fences
    follow the CommonMark rule that closes them: a closing fence is a line of
    nothing but backticks, at least as many as the opening fence — so a
    ```` ```gate ```` block with no closing fence is not a gate (the rest of the
    card would otherwise be handed to ``bash -c``), and a `````markdown
    example block containing an inner ```` ```gate ```` line stays one block.
    """
    if not body:
        return None
    lines = body.splitlines()
    i, n = 0, len(lines)
    while i < n:
        stripped = lines[i].strip()
        if not stripped.startswith("```"):
            i += 1
            continue
        k = len(stripped) - len(stripped.lstrip("`"))  # opening fence width (>= 3)
        info = stripped[k:].strip()
        buf: list[str] = []
        i += 1
        closed = False
        while i < n:
            cur = lines[i].strip()
            if cur and set(cur) <= {"`"} and len(cur) >= k:
                closed = True
                break
            buf.append(lines[i])
            i += 1
        if info == "gate" and closed:
            content = "\n".join(buf).strip()
            return content or None
        if closed:
            i += 1  # some other fenced block: skip past its closing fence
    return None


@dataclass
class CardOutcome:
    """What one pass of the rig did to a card — the evidence for the board."""

    passed: bool
    tests_passed: int = 0
    tests_total: int = 0
    worktree: str | None = None
    changed_files: list[str] = field(default_factory=list)
    output_tail: str = ""
    reason: str = ""


class HermesKanban:
    """Drive a Hermes Kanban board over a subprocess (injectable for tests)."""

    def __init__(
        self,
        board: str | None = None,
        *,
        binary: str | None = None,
        runner: Runner | None = None,
    ) -> None:
        self.board = board or os.environ.get("DACODE_KANBAN_BOARD") or None
        self.binary = binary or os.environ.get("DACODE_KANBAN_BIN") or DEFAULT_BINARY
        self._runner = runner or self._run

    @staticmethod
    def _run(argv: list[str]) -> Any:
        try:
            return subprocess.run(argv, capture_output=True, text=True, timeout=DEFAULT_TIMEOUT)
        except FileNotFoundError as exc:
            # A missing binary is a setup error, not a claim conflict: surface it
            # as a failed board call with the message a human can act on.
            missing = exc.filename or argv[0]
            return subprocess.CompletedProcess(argv, 127, stdout="", stderr=f"{missing} not found on PATH")
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(
                argv, 124, stdout="", stderr=f"{argv[0]} did not answer within {DEFAULT_TIMEOUT}s"
            )

    def _argv(self, verb_args: list[str]) -> list[str]:
        argv = [self.binary, "kanban"]
        if self.board:
            argv += ["--board", self.board]
        return argv + verb_args

    def _fail(self, action: str, proc: Any) -> KanbanError:
        detail = (proc.stderr or proc.stdout or "").strip()
        return KanbanError(detail or f"{action} failed with exit {proc.returncode}")

    def show(self, task_id: str) -> dict:
        proc = self._runner(self._argv(["show", "--json", "--", task_id]))
        if proc.returncode != 0:
            raise self._fail(f"show {task_id}", proc)
        try:
            return json.loads(proc.stdout or "{}")["task"]
        except (json.JSONDecodeError, KeyError) as exc:
            raise KanbanError(
                f"show {task_id}: unreadable board response ({exc.__class__.__name__})"
            ) from exc

    def claim(self, task_id: str, ttl: int) -> bool:
        proc = self._runner(self._argv(["claim", "--ttl", str(ttl), "--", task_id]))
        if proc.returncode == 0:
            return True
        output = f"{proc.stderr or ''}\n{proc.stdout or ''}"
        if any(marker in output for marker in _CLAIM_HELD_MARKERS):
            return False  # genuine contention: someone else holds the claim
        raise self._fail(f"claim {task_id}", proc)

    def complete(self, task_id: str, result: str, metadata: dict) -> None:
        proc = self._runner(
            self._argv(
                [
                    # `--opt=value` keeps a dash-leading result/metadata from being
                    # parsed as flags; `--` does the same for the positionals.
                    "complete",
                    f"--result={result}",
                    f"--metadata={json.dumps(metadata, ensure_ascii=False)}",
                    "--", task_id,
                ]
            )
        )
        if proc.returncode != 0:
            raise self._fail(f"complete {task_id}", proc)

    def block(self, task_id: str, reason: str, kind: str | None = None) -> None:
        verb_args = ["block"]
        if kind:
            verb_args += [f"--kind={kind}"]
        verb_args += ["--", task_id, reason]
        proc = self._runner(self._argv(verb_args))
        if proc.returncode != 0:
            raise self._fail(f"block {task_id}", proc)


def _blocked(board: HermesKanban, task_id: str, reason: str, *, kind: str | None = None) -> bool:
    """Block the card; False (with a stderr note) when the board refused."""
    try:
        board.block(task_id, reason, kind=kind)
        return True
    except KanbanError as exc:
        print(f"[kanban] {task_id}: block refused: {exc}", file=sys.stderr)
        return False


def work(task_id: str, board: HermesKanban, run_card: Callable[..., CardOutcome], *, ttl: int = 3600) -> int:
    """Claim one card, run it through the rig, report the outcome.

    Exit codes: 0 complete, 1 gate failed / rig error, 2 no gate, 3 claim
    refused (someone else holds it), 4 the board lost our claim on
    completion/block, 5 the board call itself failed (no claim ever held).
    """
    try:
        claimed = board.claim(task_id, ttl)
    except KanbanError as exc:
        # A typo'd id, an unknown board, a broken hermes — none of that is
        # contention, and none of it should wear the "another worker" story.
        print(f"[kanban] {task_id}: claim failed: {exc}", file=sys.stderr)
        return 5
    if not claimed:
        print(f"[kanban] {task_id}: claim refused — another worker owns the card", file=sys.stderr)
        return 3

    try:
        task = board.show(task_id)
    except KanbanError as exc:
        print(f"[kanban] {task_id}: show failed: {exc}", file=sys.stderr)
        return 5
    gate = parse_gate(task.get("body"))
    if not gate:
        reason = redact("card body carries no ```gate block — nothing to gate on; add one and requeue")
        return 4 if not _blocked(board, task_id, reason, kind="needs_input") else 2

    text = f"{task.get('title') or ''}\n\n{task.get('body') or ''}"
    try:
        outcome = run_card(text, gate, task.get("workspace_path"))
    except Exception as exc:  # noqa: BLE001 — a dead rig is a blocked card, not a crash
        if not _blocked(board, task_id, redact(f"rig error: {exc}")):
            return 4
        return 1

    if outcome.passed:
        result = redact(f"gate passed: {outcome.tests_passed}/{outcome.tests_total} tests")
        metadata = redact_event(
            {
                "gate": gate,
                "tests_passed": outcome.tests_passed,
                "tests_total": outcome.tests_total,
                "changed_files": outcome.changed_files,
                "worktree": outcome.worktree,
                "gate_output": outcome.output_tail,
                "runlog": str(runlog.current_path()),
            }
        )
        try:
            board.complete(task_id, result, metadata)
        except KanbanError as exc:
            print(f"[kanban] {task_id}: complete refused: {exc}", file=sys.stderr)
            return 4
        return 0

    reason = redact(f"gate failed: {outcome.tests_passed}/{outcome.tests_total} — {outcome.reason}")
    if not _blocked(board, task_id, reason):
        return 4
    return 1


def _hardened_git() -> list[str]:
    """The same flags the sandbox uses for host-side git, plus visible non-ASCII names."""
    return ["git", *GitSandbox._HARDENED_GIT, "-c", "core.quotePath=false"]


def changed_files(workspace: str | Path | None, base: str) -> list[str]:
    """Files a trial workspace changed against ``base``; [] on any doubt."""
    if not workspace:
        return []
    ws = Path(workspace)
    if not ws.is_dir():
        return []
    try:
        proc = subprocess.run(
            [*_hardened_git(), "-C", str(ws), "diff", "--name-only", base, "HEAD"],
            capture_output=True,
            text=True,
            timeout=DEFAULT_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    return sorted({line.strip() for line in proc.stdout.splitlines() if line.strip()})


def make_rig_runner(
    cfg: Any, *, rig_factory: Any = None, explicit_repo: str | Path | None = None
) -> Callable[[str, str, Any], CardOutcome]:
    """Turn a :class:`dacode.config.DaCodeConfig` into a ``run_card`` for :func:`work`.

    The card's gate becomes the rig's test command and the card's git workspace
    becomes the repo — the caller's config is copied, never mutated.
    ``explicit_repo`` is the ``--repo`` the operator actually passed (before
    config resolution): it wins over the card's ``workspace_path``, and a
    mismatch is refused rather than silently working some other checkout. A
    card workspace under ``$HOME`` is refused outright — binding it read-write
    would re-expose the home directory the sandbox just emptied.
    """
    if rig_factory is None:
        from dacode.orchestrator import DaCodeRig  # lazy: pulls in the llm stack

        rig_factory = DaCodeRig

    from dacode.env import DEFAULT_ENV_PATHS, load_env  # light, and only needed here

    def run_card(task: str, gate: str, workspace: Any) -> CardOutcome:
        repo = cfg.repo
        if workspace:
            card_ws = Path(workspace).expanduser()
            if card_ws.is_dir() and (card_ws / ".git").exists():
                card_ws = card_ws.resolve()
                if explicit_repo:
                    wanted = Path(explicit_repo).expanduser().resolve()
                    if card_ws != wanted:
                        raise ValueError(
                            f"card workspace_path {card_ws} conflicts with --repo {wanted} — "
                            "drop --repo or fix the card's workspace"
                        )
                    repo = wanted
                else:
                    if exposes_home(card_ws):
                        raise ValueError(
                            "card workspace_path sits under $HOME (or holds credentials) — "
                            "binding it read-write would re-expose the home directory inside "
                            "the sandbox; work from a checkout outside $HOME"
                        )
                    repo = card_ws
                    # main() loaded .env from cwd/--repo; this card's checkout has
                    # its own. load_env never clobbers, so only missing keys fill in.
                    load_env(*[p.replace("{repo}", str(repo)) for p in DEFAULT_ENV_PATHS])
        new_cfg = dataclasses.replace(
            cfg, test_command=("bash", "-c", gate), isolate="bwrap", repo=repo
        )

        try:
            proc = subprocess.run(
                [*_hardened_git(), "-C", str(repo), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=DEFAULT_TIMEOUT,
            )
            base = proc.stdout.strip() if proc.returncode == 0 and proc.stdout.strip() else "HEAD"
        except (OSError, subprocess.SubprocessError):
            base = "HEAD"

        result = rig_factory(new_cfg).run_task(task)
        probe = result.best.probe
        if probe is not None:
            worktree = probe.workspace or None
            tail = (probe.gate_output or probe.stdout or "")[-2000:]
            # diffs are evidence for the *complete* path only; a failed run never
            # diffs — probe.workspace may not even be a valid checkout then.
            diffs = changed_files(probe.workspace, base) if result.passed else []
        else:
            worktree, tail, diffs = None, "", []
        if result.passed:
            reason = ""
        elif result.needs_human_review:
            reason = "needs human review"
        elif probe is None:
            reason = f"no variant was probed in {result.rounds} round(s)"
        else:
            why = ", ".join(dict.fromkeys(probe.errors)) or "tests did not pass"
            reason = f"gate failed after {result.rounds} round(s): {why}"
            if probe.stderr.startswith("trial error:"):
                reason += f" ({probe.stderr[:300]})"
        return CardOutcome(
            passed=bool(result.passed),
            tests_passed=probe.tests_passed if probe else 0,
            tests_total=probe.tests_total if probe else 0,
            worktree=worktree,
            changed_files=diffs,
            output_tail=tail,
            reason=reason,
        )

    return run_card
