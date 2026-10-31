"""The Harness² optimizer: Propose → Probe → Compose.

Two levels of recursion, exactly as the blueprint describes:

* **Task-level** (``run``): for a single task, propose variants, probe each in
  an isolated worktree, compose the best, and iterate within a budget.
* **Harnessing-level** (``memory.ingest`` via the orchestrator): the winning
  variant's lesson is written back to the vault so future runs start smarter.

All LLM interaction is funneled through the injected ``llm`` client, so the loop
is deterministic and testable with a stub.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import json
import re
import sys
from typing import TYPE_CHECKING

from dacode import runlog

from dacode.harness.models import HarnessResult, HarnessVariant, ProbeResult

if TYPE_CHECKING:
    from dacode.discipline.critic import PstackCritic
    from dacode.executor.base import Executor
    from dacode.judgment import JevJudge
    from dacode.llm.base import LLMClient
    from dacode.memory import VaultMemory
    from dacode.sandbox import GitSandbox

__all__ = ["HarnessOptimizer"]

DEFAULT_SYSTEM_PROMPT = (
    "You are a meticulous coding agent. Implement the task with the smallest "
    "correct change. Write or update tests that assert literal behaviour. "
    "Run the tests and do not stop until they pass."
)

_PROPOSE_SYSTEM = """You are the Harness Optimizer (layer 2 of DaCode).
A "harness" is everything around the model: the system prompt, the tools
enabled, and the memory snippets injected. Your job is to propose DIFFERENT
harness variants for the task below, so the rig can contrast them.

Return ONLY a JSON array. Each element:
{"label": "<short>", "system_prompt": "<full prompt>", "tools": ["..."]}
Propose {n} genuinely different variants (vary the prompt strategy, not wording).
No prose, no markdown fences — JSON only."""

_COMPOSE_SYSTEM = """You are the Harness Optimizer (layer 2 of DaCode).
You are given several probed harness variants and their measured results.
Pick the single best variant for the task. Prefer passing tests, then fewer
critique errors, then higher score. Return ONLY JSON:
{"best_id": "<variant id>", "rationale": "<one sentence>"}"""


class HarnessOptimizer:
    def __init__(
        self,
        llm: "LLMClient",
        sandbox: "GitSandbox",
        executor: "Executor",
        critic: "PstackCritic",
        memory: "VaultMemory | None" = None,
        judge: "JevJudge | None" = None,
        *,
        budget: int = 2,
        variants_per_round: int = 2,
        base_prompt: str = DEFAULT_SYSTEM_PROMPT,
        keep_winning_worktree: bool = False,
        escalation_models: list[str] | None = None,
        baseline: tuple[int, int] | None = None,
        parallel: int = 1,
        fleet: list[str] | None = None,
    ) -> None:
        self.llm = llm
        self.sandbox = sandbox
        self.executor = executor
        self.critic = critic
        self.memory = memory
        self.judge = judge
        self.budget = max(1, budget)
        self.variants_per_round = max(1, variants_per_round)
        self.base_prompt = base_prompt
        self.keep_winning_worktree = keep_winning_worktree
        self.escalation_models = list(escalation_models or [])
        # (total, skipped) measured on the untouched repo; None disables the guard
        self.baseline = baseline
        self.parallel = max(1, parallel)
        # Fleet diversity: one round probes different model families at once, so a
        # spec that only one model's habits can satisfy does not look "solved".
        self.fleet = [m for m in (fleet or []) if m]
        self._used_models: set[str] = set()
        self._sandbox_lock = threading.Lock()

    # -- propose -----------------------------------------------------------

    def propose(self, task: str, base: HarnessVariant, n: int) -> list[HarnessVariant]:
        """Ask the LLM for ``n`` candidate harness variants."""
        user = (
            f"Task:\n{task}\n\n"
            f"Current best harness system_prompt:\n{base.system_prompt}\n\n"
            f"Seeded memory snippets:\n{base.seed_context() or '(none)'}\n\n"
            f"Propose {n} variants."
        )
        raw = self.llm.complete(_PROPOSE_SYSTEM, user)
        specs = _parse_json_array(raw)
        variants: list[HarnessVariant] = []
        for i, spec in enumerate(specs[:n]):
            if not isinstance(spec, dict):
                continue
            raw_label = str(spec.get("label", f"proposal-{i + 1}"))
            clean_label = _sanitize_label(raw_label) or f"proposal-{i + 1}"
            variants.append(
                HarnessVariant(
                    label=clean_label,
                    system_prompt=str(spec.get("system_prompt") or base.system_prompt),
                    tools=[str(t) for t in spec.get("tools", [])] or list(base.tools),
                    memory_snippets=list(base.memory_snippets),
                    parent_id=base.id,
                    payload=dict(base.payload),
                )
            )
        if not variants:
            # Robustness: never let a bad LLM response stall the loop.
            variants.append(
                HarnessVariant(
                    label="fallback",
                    system_prompt=base.system_prompt,
                    tools=list(base.tools),
                    memory_snippets=list(base.memory_snippets),
                    parent_id=base.id,
                    payload=dict(base.payload),
                )
            )
        return variants

    # -- probe -------------------------------------------------------------

    def _trial_nonce(self) -> str:
        """A per-process stamp so two runs of one task do not collide."""
        nonce = getattr(self, "_run_nonce", None)
        if not nonce:
            nonce = f"{int(time.time()) % 0x100000:05x}"
            self._run_nonce = nonce
        return nonce

    def probe(self, variant: HarnessVariant, task: str, trial_id: str) -> ProbeResult:
        """Run one variant end-to-end in a fresh isolated worktree."""
        variant.payload.setdefault("task", task)
        # git worktree add is not safe to run concurrently on one repo: keep
        # creation serialised, run the executor itself in parallel below.
        with self._sandbox_lock:
            workspace = self.sandbox.create(trial_id)
            # Hash the suite before the trial can touch it; the git-status route
            # cannot see edits a trial has committed.
            contract = (
                self.critic.snapshot_contract(workspace)
                if hasattr(self.critic, "snapshot_contract")
                else None
            )
        started = time.perf_counter()
        report = None  # a failed trial must still be recordable
        try:
            exec_result = self.executor.run(variant, workspace)
            # Ask the critic only for the arguments it declares: catching TypeError
            # here would also swallow a real one from inside the review and silently
            # re-run the suite without the baseline guard.
            import inspect

            parameters = inspect.signature(self.critic.review_workspace).parameters
            kwargs: dict[str, object] = {}
            if "baseline" in parameters:
                kwargs["baseline"] = self.baseline
            if "contract" in parameters:
                kwargs["contract"] = contract
            report = self.critic.review_workspace(workspace, **kwargs)
            if report.passed:
                # git worktree add/remove and checkpoint all touch the shared .git;
                # run them one at a time or a lock contention surfaces as a fake
                # "trial error" on a probe that already passed (audit H2).
                with self._sandbox_lock:
                    self.sandbox.checkpoint(
                        trial_id, f"dacode: {variant.label} passed ({trial_id})"
                    )
            result = ProbeResult(
                variant_id=variant.id,
                workspace=str(workspace),
                exit_code=exec_result.exit_code,
                tests_total=report.tests_total,
                tests_passed=report.tests_passed,
                critique_passed=report.passed,
                critique_score=report.score,
                stdout=exec_result.stdout[-4000:],
                stderr=exec_result.stderr[-2000:],
                elapsed_s=round(time.perf_counter() - started, 3),
                tests_skipped=getattr(report, "tests_skipped", 0),
                model=variant.model or getattr(self.executor, "model", None),
                gate_output=_gate_evidence(report),
                errors=[f.check for f in getattr(report, "findings", []) if f.severity == "error"],
            )
        except Exception as exc:  # noqa: BLE001 - a failed trial must not kill the run
            result = ProbeResult(
                variant_id=variant.id,
                workspace=str(workspace),
                exit_code=1,
                critique_passed=False,
                critique_score=0.0,
                stderr=f"trial error: {exc!r}",
                elapsed_s=round(time.perf_counter() - started, 3),
                model=variant.model or getattr(self.executor, "model", None),
            )
        variant.score = _variant_score(result)
        variant.probe = result
        runlog.record({
            "type": "probe",
            "task": variant.payload.get("task", ""),
            "variant_id": variant.id,
            "label": variant.label,
            "exit_code": result.exit_code,
            "tests_passed": result.tests_passed,
            "tests_total": result.tests_total,
            "critique_passed": result.critique_passed,
            "score": variant.score,
            "elapsed_s": result.elapsed_s,
            "tests_skipped": result.tests_skipped,
            "model": result.model,
            "findings": [f"{f.severity}:{f.check}" for f in getattr(report, "findings", [])],
        })
        return result

    def _assign_fleet(self, variants: list[HarnessVariant], round_no: int) -> None:
        """Give each variant in a round a different model from the fleet."""
        pinned = getattr(self, "_pinned_model", None)
        if pinned:
            for variant in variants:
                variant.model = pinned
            return
        if not self.fleet:
            return
        offset = round_no - 1
        for idx, variant in enumerate(variants):
            if variant.model:
                continue
            variant.model = self.fleet[(offset + idx) % len(self.fleet)]

    def next_fleet_model(self) -> str | None:
        """Pick a model whose family has not been tried yet.

        Repeating a family that already failed is how a rig wastes budget, so
        prefer an untried family, then any fleet member, then the legacy ladder.
        """
        used_families = {_family(model) for model in self._used_models}
        for model in self.fleet:
            if _family(model) not in used_families:
                return model
        if self.fleet:
            return self.fleet[0]
        return None

    def probe_round(self, task: str, variants: list[HarnessVariant], round_no: int) -> None:
        """Probe every variant of a round, in parallel when allowed.

        Worktree creation stays serialised (one git index per repo), so the
        parallelism that matters — the executor's own wall time — is what
        overlaps.
        """
        jobs = []
        for idx, variant in enumerate(variants, start=1):
            # A per-run nonce keeps a re-run of the same task from colliding with the
            # worktree the previous run kept (audit H6a: SandboxError, run crashed).
            trial_id = (
                f"{_slug(task)}-{self._trial_nonce()}-r{round_no}-{idx}-{variant.id}"
            )
            variant.payload.setdefault("task", task)
            jobs.append((variant, trial_id))

        if self.parallel <= 1 or len(jobs) == 1:
            for variant, trial_id in jobs:
                self.probe(variant, task, trial_id)
            return

        with ThreadPoolExecutor(max_workers=min(self.parallel, len(jobs))) as pool:
            futures = [pool.submit(self.probe, variant, task, trial_id) for variant, trial_id in jobs]
            for future in futures:
                try:
                    future.result()
                except Exception:  # noqa: BLE001 - probe() already records its own failures
                    pass

    # -- compose -----------------------------------------------------------

    def compose(self, task: str, variants: list[HarnessVariant]) -> HarnessVariant:
        """Pick the best variant — Jev judgment, LLM-assisted, deterministic fallback."""
        if not variants:
            raise ValueError("compose() called with no variants")
        # Gate-passing variants are the only candidates when any exist: a judge, an
        # LLM or the ranking may pick among winners, never over a gate failure.
        passing = [v for v in variants if v.probe is not None and v.probe.critique_passed]
        pool = passing or variants
        ranked = sorted(
            pool,
            key=lambda v: (bool(v.probe and v.probe.critique_passed), v.score or 0.0),
            reverse=True,
        )
        fallback = ranked[0]
        table = "\n".join(
            f"- id={v.id} label={v.label!r} score={v.score} "
            f"tests={v.probe.tests_passed if v.probe else '?'}/"
            f"{v.probe.tests_total if v.probe else '?'} "
            f"passed={v.probe.critique_passed if v.probe else '?'}"
            for v in ranked
        )
        # System One judgment (Jev): a bounded pick among probed variants with
        # typed probabilities. It never issues pass/fail verdicts — the critic
        # gate stays the only quality authority. A failure falls through to the
        # reasoning-model pick, then to the measured ranking.
        chosen, chooser = fallback, "ranking"
        if self.judge is not None and len(ranked) > 1:
            try:
                pick, confidence = self.judge.choice(
                    state=(
                        f"Task:\n{task}\n\nProbed harness variants "
                        f"(deterministic score order, best first):\n{table}"
                    ),
                    instructions=(
                        "Which probed harness variant is the single best "
                        "implementation of the task? Prefer passing tests, then "
                        "fewer critique errors, then higher score. Answer with "
                        "the variant id."
                    ),
                    criteria={
                        v.id: (
                            f"label={v.label!r}; score={v.score}; "
                            f"tests={v.probe.tests_passed if v.probe else '?'}/"
                            f"{v.probe.tests_total if v.probe else '?'}; "
                            f"critique_passed={v.probe.critique_passed if v.probe else '?'}"
                        )
                        for v in ranked
                    },
                )
            except Exception as exc:  # noqa: BLE001 - a judge failure must not kill the run
                print(f"[judge] jev pick failed: {exc}", file=sys.stderr)
                pick = None
            if pick is not None:
                # Resolve inside `ranked`, not `variants`: a judge that names a
                # gate-failing id must not be able to hand the run to it (the pool
                # above exists for exactly this reason).
                candidate = next((v for v in ranked if v.id == pick), None)
                if candidate is not None:
                    chosen, chooser = candidate, "judge"
        if chooser == "ranking":
            user = f"Task:\n{task}\n\nProbed variants (best first):\n{table}\n\nPick the best."
            try:
                raw = self.llm.complete(_COMPOSE_SYSTEM, user)
                data = _parse_json_object(raw)
                candidate = next(
                    (v for v in ranked if v.id == str(data.get("best_id", ""))), None
                )
                if candidate is not None:
                    chosen, chooser = candidate, "llm"
            except Exception:  # noqa: BLE001 - fall back to the measured ranking
                pass
        runlog.record({
            "type": "compose",
            "task": task,
            "table": [
                {
                    "id": v.id,
                    "label": v.label,
                    "score": v.score,
                    "tests_passed": v.probe.tests_passed if v.probe else 0,
                    "tests_total": v.probe.tests_total if v.probe else 0,
                    "passed": bool(v.probe.critique_passed) if v.probe else False,
                }
                for v in ranked
            ],
            "chosen_id": chosen.id,
            "chooser": chooser,
        })
        return chosen

    # -- full loop ---------------------------------------------------------

    def run(self, task: str, base: HarnessVariant | None = None) -> HarnessResult:
        """Execute the Propose–Probe–Compose loop for ``task``.

        With a judge attached, every round that fails the gate asks it for the
        next action — ``retry_same`` (re-probe unchanged), ``new_proposal``
        (fresh variants), ``escalate_model`` (switch the executor up the
        ladder), or ``human_review`` (stop and flag). Without a judge the
        loop simply runs out its budget, exactly as before.
        """
        if base is None:
            base = HarnessVariant(system_prompt=self.base_prompt, label="base")
        self._run_nonce = self._trial_nonce()

        run_started = time.perf_counter()
        all_variants: list[HarnessVariant] = []
        seen: set[str] = set()
        best = base
        best.score = 0.0
        rounds_run = 0
        needs_human_review = False
        ladder = list(self.escalation_models)

        reprobe: list[HarnessVariant] | None = None
        round_no = 0
        while round_no < self.budget:
            round_no += 1
            rounds_run = round_no
            if reprobe is not None:
                variants = reprobe  # triage said: re-probe these unchanged
                reprobe = None
            else:
                variants = self.propose(task, best, self.variants_per_round)
            self._assign_fleet(variants, round_no)
            for variant in variants:
                if variant.model:
                    self._used_models.add(variant.model)
            self.probe_round(task, variants, round_no)
            for variant in variants:
                if variant.id not in seen:
                    seen.add(variant.id)
                    all_variants.append(variant)
            round_best = self.compose(task, variants)
            previous_best = best
            if (round_best.score or 0.0) >= (best.score or 0.0):
                best = round_best
            # cleanup losing trials to keep the repo tidy — including the best of an
            # earlier round that this one just superseded, which used to be left
            # behind on disk (audit M2)
            self._cleanup_losers(variants, keep=best, also=[previous_best])
            if best.score is not None and best.score >= 1.0 and best.probe and best.probe.critique_passed:
                break
            if round_no >= self.budget:
                break  # no budget left to act on a triage decision

            action = "new_proposal"
            if self.judge is not None:
                action = self._triage(task, round_no, best, variants, can_escalate=bool(ladder) or bool(self.fleet))
            if action == "human_review":
                needs_human_review = True
                break
            if action == "escalate_model":
                model = self.next_fleet_model() or (ladder.pop(0) if ladder else None)
                if model and hasattr(self.executor, "model"):
                    self.executor.model = model
                    # _assign_fleet() rewrites variant.model each round; without this
                    # pin the escalation was a no-op whenever --fleet was set (M2).
                    self._pinned_model = model
                    self._used_models.add(model)
            if action == "retry_same":
                reprobe = list(variants)

        runlog.record({
            "type": "run",
            "task": task,
            "elapsed_s": round(time.perf_counter() - run_started, 3),
            "rounds": rounds_run,
            "passed": bool(best.probe and best.probe.critique_passed),
            "needs_human_review": needs_human_review,
            "best_id": best.id,
            "best_score": best.score,
            "models": sorted({v.model for v in all_variants if v.model}),
        })
        return HarnessResult(
            task=task,
            rounds=rounds_run,
            best=best,
            variants=all_variants,
            passed=bool(best.probe and best.probe.critique_passed),
            needs_human_review=needs_human_review,
        )

    def _triage(
        self,
        task: str,
        round_no: int,
        best: HarnessVariant,
        variants: list[HarnessVariant],
        *,
        can_escalate: bool,
    ) -> str:
        """Ask the judge for the next action after a failed round.

        Returns one of ``new_proposal`` / ``retry_same`` / ``escalate_model``
        / ``human_review``. Any failure or unconfident answer defaults to
        ``new_proposal`` — the loop's previous behaviour.
        """
        table = "\n".join(
            f"- id={v.id} label={v.label!r} score={v.score} "
            f"tests={v.probe.tests_passed if v.probe else '?'}/"
            f"{v.probe.tests_total if v.probe else '?'} "
            f"passed={v.probe.critique_passed if v.probe else '?'}"
            for v in variants
        )
        criteria = {
            "new_proposal": "Propose fresh harness variants and probe them — the normal next round.",
            "retry_same": (
                "Re-probe this round's variants unchanged — only for transient "
                "failures (network, rate limits), not for bad code."
            ),
            "human_review": (
                "Stop and hand the task to a human — the current approach is "
                "unlikely to succeed."
            ),
        }
        if can_escalate:
            criteria["escalate_model"] = (
                "Switch the executor to a stronger model, then propose again."
            )
        probe = best.probe
        state = (
            f"Task:\n{task}\n\n"
            f"Round {round_no} of {self.budget} finished; no variant passed the gate.\n"
            f"Best so far: id={best.id} label={best.label!r} score={best.score} "
            f"tests={probe.tests_passed if probe else '?'}/"
            f"{probe.tests_total if probe else '?'} "
            f"critique_passed={probe.critique_passed if probe else '?'}\n\n"
            f"This round's variants:\n{table}\n"
        )
        try:
            action, _confidence = self.judge.choice(
                state=state,
                instructions=(
                    "The gate did not pass this round. Choose the single next "
                    "action for the rig. Answer with exactly one action id."
                ),
                criteria=criteria,
                min_confidence=0.5,
            )
        except Exception as exc:  # noqa: BLE001 - a judge failure must not kill the run
            print(f"[judge] triage failed: {exc}", file=sys.stderr)
            return "new_proposal"
        return action or "new_proposal"

    def _cleanup_losers(
        self,
        variants: list[HarnessVariant],
        *,
        keep: HarnessVariant,
        also: list[HarnessVariant] | None = None,
    ) -> None:
        if self.keep_winning_worktree:
            return
        for variant in [*variants, *(also or [])]:
            if variant is keep or variant is None:
                continue
            trial_dir = variant.probe.workspace if variant.probe else None
            if not trial_dir:
                continue
            task_id = _trial_id_from_path(trial_dir)
            if task_id:
                try:
                    self.sandbox.remove(task_id, delete_branch=True)
                except Exception:  # noqa: BLE001 - cleanup is best-effort
                    pass


# -- helpers ---------------------------------------------------------------


def _gate_evidence(report: object) -> str:
    """The gate's own output, for the evidence trail.

    ``unittest`` suites print their summary to stderr and a gate that timed out
    (124), was refused by isolation (125) or found no command (127) writes only
    to stderr — stdout alone reads as "no gate output" and the executor's
    chatter gets reported as evidence instead. Both streams belong here.
    """
    stdout = (getattr(report, "stdout", "") or "").strip()
    stderr = (getattr(report, "stderr", "") or "").strip()
    return "\n".join(part for part in (stdout, stderr) if part)[-4000:]


def _variant_score(result: ProbeResult) -> float:
    """Rank a trial: passing tests dominate, critique score breaks ties."""
    if result.tests_total == 0:
        base = 0.0
    else:
        base = result.tests_passed / result.tests_total
    bonus = 0.5 if result.critique_passed else 0.0
    exit_penalty = 0.0 if result.exit_code == 0 else 0.25
    score = round(max(0.0, min(1.0, base + bonus - exit_penalty)), 3)
    # A gate failure must never tie a gate pass: the audit showed an earlier
    # failing variant keeping 1.0 and winning the stable sort, after which the
    # passing worktree was cleaned up.
    if not result.critique_passed:
        score = min(score, 0.999)
    return score


def _family(model: str) -> str:
    """``anthropic/claude-sonnet-4`` and ``claude-sonnet-4`` share a family."""
    head = model.split("/")[0] if "/" in model else model
    return head.split("-")[0].lower()


_CONTROL_SEQ_RE = re.compile(r"\x1b(?:\][^\x07\x1b]*(?:\x07|\x1b\\)|\[[0-9;?]*[A-Za-z])|[\x00-\x1f\x7f]")


def _sanitize_label(text: str) -> str:
    """Strip terminal escape sequences and control characters from a variant label."""
    return _CONTROL_SEQ_RE.sub("", text).strip()


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", text.strip().lower()).strip("-")
    return (slug or "task")[:40]


def _trial_id_from_path(path: str) -> str | None:
    from pathlib import Path

    name = Path(path).name
    return name or None


def _parse_json_array(raw: str) -> list:
    data = _extract_json(raw, expect="[")
    return data if isinstance(data, list) else []


def _parse_json_object(raw: str) -> dict:
    data = _extract_json(raw, expect="{")
    return data if isinstance(data, dict) else {}


def _extract_json(raw: str, *, expect: str):
    """Pull a JSON value out of an LLM reply that may include prose/fences."""
    if not raw:
        return [] if expect == "[" else {}
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    closer = "]" if expect == "[" else "}"
    start = text.find(expect)
    end = text.rfind(closer)
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return [] if expect == "[" else {}
    return [] if expect == "[" else {}
