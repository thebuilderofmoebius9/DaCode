"""DaCode command-line interface.

Usage:
  dacode run "<task>" [--budget N] [--llm KIND] [--executor pi|script] ...
  dacode kanban work <task_id> [--board SLUG] [run options]
  dacode memory search "<query>"
  dacode sandbox list | clean
  dacode layers
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dacode.config import DaCodeConfig
from dacode.llm import provider_names
from dacode.env import DEFAULT_ENV_PATHS, load_env
from dacode.harness.models import HarnessResult
from dacode.memory import VaultMemory, status
from dacode.memory.backends.local import LocalStoreBackend
from dacode.orchestrator import DaCodeRig
from dacode.sandbox import GitSandbox

__all__ = ["main", "build_parser"]


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from None
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"--ttl must be positive, got {parsed}")
    return parsed


def _add_run_options(p: argparse.ArgumentParser) -> None:
    """Options shared by `run` and `kanban work` — one card is one rig run."""
    p.add_argument("--budget", type=int, default=2, help="Harness² recursion rounds")
    p.add_argument("--variants", type=int, default=2, help="variants proposed per round")
    p.add_argument(
        "--llm",
        default=None,
        help="stub | local | " + " | ".join(provider_names()),
    )
    p.add_argument("--model", default=None, help="model id for the chosen backend")
    p.add_argument("--executor", default=None, choices=["pi", "script"])
    p.add_argument(
        "--executor-provider",
        default=None,
        help='provider id the pi executor logs in with, e.g. "zai" (default: openrouter)',
    )
    p.add_argument("--script-command", default=None, help="command for --executor script")
    p.add_argument(
        "--test-command",
        default=None,
        help="shell command that runs the project's tests "
        "(default: python -m pytest -q); run via bash -c inside the trial worktree",
    )
    p.add_argument(
        "--test-glob",
        default=None,
        help="glob for static test scanning (default: test_*.py)",
    )
    p.add_argument("--keep-worktree", action="store_true", help="keep losing trial worktrees")
    p.add_argument(
        "--variant-timeout",
        type=float,
        default=None,
        help="seconds a single variant may spend executing (default: 900)",
    )
    p.add_argument(
        "--isolate",
        choices=["none", "bwrap"],
        default=None,
        help="sandbox trials and the test gate with bubblewrap (default: bwrap; "
        "pass --isolate none to run on the host)",
    )
    p.add_argument(
        "--isolate-network",
        action="store_true",
        help="allow the sandboxed trial to open sockets (an agent needs this)",
    )
    p.add_argument(
        "--fleet",
        default=None,
        help="comma-separated models probed side by side, e.g. "
        "'deepseek/deepseek-v4.1-flash,anthropic/claude-sonnet-4'",
    )
    p.add_argument(
        "--parallel",
        type=int,
        default=None,
        help="probe this many variants at once (default: 1; worktree creation stays serialised)",
    )
    p.add_argument(
        "--escalate-models",
        default=None,
        help="comma-separated model ladder for the judge's escalate_model "
        "triage action (e.g. 'claude-sonnet-4-5,gpt-4o')",
    )
    p.add_argument(
        "--judge",
        default=None,
        help="bounded-decision judge for compose: 'jev' = TypeSafe System One "
        "(unset = off)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dacode",
        description="DaCode — The Ultimate Autonomous Engineering Rig",
    )
    parser.add_argument("--repo", default=None, help="target repository (default: cwd)")
    parser.add_argument("--vault", default=None, help="Obsidian vault path for memory")
    parser.add_argument("--sandbox-root", default=None, help="where trial worktrees live")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a task through the full 5-layer rig")
    run.add_argument("task", help="the engineering task, in plain language")
    _add_run_options(run)

    kanban = sub.add_parser("kanban", help="work cards from a Hermes Kanban board")
    kanban_sub = kanban.add_subparsers(dest="kanban_command", required=True)
    kwork = kanban_sub.add_parser(
        "work", help="claim one card, run it through the rig, complete or block it"
    )
    kwork.add_argument("task_id", help="Kanban card id, e.g. t_1485ddf4")
    kwork.add_argument("--board", default=None, help="board slug (default: DACODE_KANBAN_BOARD or the current board)")
    kwork.add_argument("--ttl", type=_positive_int, default=None, help="claim TTL in seconds (default: covers the whole run)")
    _add_run_options(kwork)

    mcp = sub.add_parser("mcp", help="serve memory + run history over MCP (stdio)")
    mcp.add_argument(
        "--keyword-only",
        action="store_true",
        help="skip the semantic index (no embedding calls; keyword scoring only)",
    )

    watch = sub.add_parser("watch", help="live read-only view of the run log")
    watch.add_argument("--runlog", default=None,
                       help="runlog path (default: DACODE_RUNLOG or ~/.local/state/dacode/runlog.jsonl)")
    watch.add_argument("--interval", type=float, default=2.0, help="repaint period in seconds")
    watch.add_argument("--history", type=int, default=10, help="finished runs to show below the live one")
    watch.add_argument("--once", action="store_true", help="render one frame and exit (no follow)")

    retro = sub.add_parser("retro", help="read the runlog and suggest harness changes")
    retro.add_argument("--days", type=int, default=14, help="look back this many days (default: 14)")
    retro.add_argument("--task", default=None, help="filter by task substring")
    retro.add_argument("--json", action="store_true", help="machine-readable output")

    mem = sub.add_parser("memory", help="query the vault memory layer")
    mem_sub = mem.add_subparsers(dest="mem_command", required=True)
    search = mem_sub.add_parser("search", help="retrieve relevant lessons for a query")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=5)
    search.add_argument(
        "--backend",
        default=None,
        help="memory backend: auto (default) | local | hermes | off | module:factory",
    )
    mem_sub.add_parser("backends", help="show which memory backends are usable")
    mem_index = mem_sub.add_parser("index", help="embed the vault into the local vector store")
    mem_index.add_argument("--rebuild", action="store_true", help="re-embed every note")

    sand = sub.add_parser("sandbox", help="inspect or clean trial worktrees")
    sand_sub = sand.add_subparsers(dest="sand_command", required=True)
    sand_sub.add_parser("list", help="list registered worktrees")
    sand_sub.add_parser("clean", help="remove all dacode trial worktrees")

    jev = sub.add_parser(
        "judge-eval",
        help="score Jev's Compose picks against recorded gate-verified winners",
    )
    jev.add_argument("--limit", type=int, default=None, help="max cases to evaluate")
    jev.add_argument("--model", default=None, help="Jev model id (default: the pinned jev-1.13.0)")
    jev.add_argument(
        "--log",
        default=None,
        help="runlog path (default: DACODE_RUNLOG or ~/.local/state/dacode/runlog.jsonl)",
    )
    st = sub.add_parser("state", help="query the local SQLite state store (memory.db)")
    st_sub = st.add_subparsers(dest="state_command", required=True)
    st_search = st_sub.add_parser("search", help="search recorded runs, variants, and picks")
    st_search.add_argument("query")
    st_search.add_argument("--limit", type=int, default=10)
    st_sub.add_parser("stats", help="counts and freshness of the local store")
    st_fail = st_sub.add_parser("failures", help="approaches that failed the gate")
    st_fail.add_argument("--task", default=None, help="filter by task substring")
    st_fail.add_argument("--limit", type=int, default=20)
    st_sub.add_parser("rebuild", help="rebuild the DB from the JSONL run log")
    sub.add_parser("layers", help="show the status of the five layers")
    return parser


def _make_config(args) -> DaCodeConfig:
    cfg = DaCodeConfig.from_env(args.repo)
    if args.vault:
        cfg.vault = Path(args.vault).expanduser().resolve()
    if args.sandbox_root:
        cfg.sandbox_root = Path(args.sandbox_root).expanduser().resolve()
    for attr in ("budget", "variants"):
        value = getattr(args, attr, None)
        if value is not None:
            setattr(cfg, "variants_per_round" if attr == "variants" else attr, value)
    if getattr(args, "llm", None):
        cfg.llm_kind = args.llm
    if getattr(args, "model", None):
        cfg.model = args.model
    if getattr(args, "executor", None):
        cfg.executor = args.executor
    if getattr(args, "executor_provider", None):
        cfg.executor_provider = args.executor_provider
    if getattr(args, "script_command", None):
        cfg.script_command = args.script_command
    if getattr(args, "test_command", None):
        # Run the user's test command through a shell so pipes/`&&` work, and
        # so a non-Python suite (e.g. `make -C port/renderer test`) is one argv.
        cfg.test_command = ("bash", "-c", args.test_command)
    if getattr(args, "test_glob", None):
        cfg.test_glob = args.test_glob
    if getattr(args, "variant_timeout", None) is not None:
        cfg.variant_timeout = args.variant_timeout
    if getattr(args, "isolate", None):
        cfg.isolate = args.isolate
    if getattr(args, "isolate_network", False):
        cfg.isolate_network = True
    if getattr(args, "fleet", None):
        cfg.fleet = [m.strip() for m in args.fleet.split(",") if m.strip()]
    if getattr(args, "parallel", None) is not None:
        cfg.parallel = max(1, args.parallel)
    if getattr(args, "keep_worktree", False):
        cfg.keep_winning_worktree = True
    if getattr(args, "judge", None):
        cfg.judge = args.judge
    if getattr(args, "escalate_models", None):
        cfg.escalate_models = [m.strip() for m in args.escalate_models.split(",") if m.strip()]
    return cfg


def _print_result(result: HarnessResult) -> None:
    best = result.best
    print()
    print("=" * 62)
    print(f"  DaCode result — {'PASS ✅' if result.passed else 'FAIL ❌'}")
    print("=" * 62)
    print(f"  task      : {result.task}")
    print(f"  rounds    : {result.rounds}")
    print(f"  variants  : {len(result.variants)} probed")
    print(f"  best      : {best.id} ({best.label})  score={best.score}")
    if best.probe:
        p = best.probe
        print(f"  tests     : {p.tests_passed}/{p.tests_total} pass  (exit {p.exit_code})")
        print(f"  workspace : {p.workspace}")
    if result.lesson_path:
        print(f"  lesson    : {result.lesson_path}")
    if result.needs_human_review:
        print("  note      : needs human review — the judge stopped the run")
    print("=" * 62)
    for v in sorted(result.variants, key=lambda x: x.score or 0, reverse=True):
        mark = "★" if v is best else " "
        t = f"{v.probe.tests_passed}/{v.probe.tests_total}" if v.probe else "?"
        print(f"  {mark} {v.id:<6} {v.label:<20} score={v.score}  tests={t}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Load secrets from the target repo's own .env before building the rig.
    # DaCode never reads ~/.hermes/.env or ~/.pi/agent/auth.json.
    repo_hint = Path(args.repo).expanduser() if getattr(args, "repo", None) else Path.cwd()
    load_env(*[p.replace("{repo}", str(repo_hint)) for p in DEFAULT_ENV_PATHS])

    if args.command == "judge-eval":
        from dacode.jeval import collect_cases, evaluate, summarize
        from dacode.judgment import JevJudge
        from dacode.runlog import load_events

        cases = collect_cases(load_events(args.log))
        if not cases:
            print(
                "no gate-verified compose cases in the run log yet — run tasks "
                "and let the recorder accumulate (DACODE_RUNLOG, default "
                "~/.local/state/dacode/runlog.jsonl)"
            )
            return 1
        judge = JevJudge(model=args.model) if args.model else JevJudge()
        print(summarize(evaluate(cases, judge, limit=args.limit)))
        return 0

    if args.command == "mcp":
        from dacode.mcp import MemoryServer

        cfg = _make_config(args)
        if not cfg.vault:
            print("error: no vault configured; pass --vault PATH", file=sys.stderr)
            return 2
        server = MemoryServer(vault=cfg.vault, semantic=False if args.keyword_only else "auto")
        server.serve()
        return 0

    if args.command == "retro":
        from dacode import retro

        report = retro.build_report(since_days=args.days, task_filter=args.task)
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(retro.render_text(report))
        return 0

    if args.command == "watch":
        from dacode.watch import watch
        frame = watch(path=Path(args.runlog).expanduser() if args.runlog else None,
                      interval=args.interval, history=args.history, once=args.once)
        if args.once:
            print(frame)
        return 0

    if args.command == "layers":
        cfg = _make_config(args)
        rig = DaCodeRig(cfg)
        print(f"DaCode rig @ {cfg.repo}")
        for layer in rig.layers():
            icon = "✅" if layer.ready else "⚠️ "
            print(f"  {icon} {layer.name:<22} {layer.detail}")
        return 0

    if args.command == "memory":
        cfg = _make_config(args)
        if cfg.vault is None or not Path(cfg.vault).is_dir():
            print(f"error: vault not found ({cfg.vault}); pass --vault", file=sys.stderr)
            return 2
        if args.mem_command == "backends":
            for row in status(cfg.vault):
                mark = "OK  " if row["available"] else "off "
                print(f"  {mark} {row['name']:<8} {row['detail']}")
            return 0
        if args.mem_command == "index":
            index = LocalStoreBackend(cfg.vault)
            if not index.embedder.configured:
                print(
                    "error: no embedding endpoint configured — set DACODE_EMBED_BASE_URL "
                    "and DACODE_EMBED_MODEL (see README 'Memory backends')",
                    file=sys.stderr,
                )
                return 2
            counts = index.store.index(cfg.vault, index.embedder, rebuild=args.rebuild)
            print(
                f"indexed: {counts['embedded']} embedded, {counts['skipped']} unchanged, "
                f"{counts['pruned']} pruned → {counts['total']} vectors "
                f"({counts['model']})"
            )
            return 0
        mode = args.backend if args.backend is not None else cfg.semantic
        mem = VaultMemory(cfg.vault, semantic=mode)
        backend = mem._semantic()
        label = f"{backend.name}+keyword" if backend is not None else "keyword only"
        print(f"# retrieval: {label}")
        hits = mem.retrieve(args.query, limit=args.limit)
        if not hits:
            print("no matching notes")
            return 1
        from dacode.redact import redact

        for hit in hits:
            # A vault note can contain a key someone pasted into a lesson; the
            # terminal is an output boundary like any other (audit, Oct 2026).
            print(f"[{hit.score:>6}] {redact(hit.title)}")
            print(f"          {hit.path}")
            print(f"          {redact(hit.snippet[:160])}…")
        return 0

    if args.command == "sandbox":
        cfg = _make_config(args)
        sandbox = GitSandbox(cfg.repo, cfg.sandbox_root)
        if args.sand_command == "list":
            for info in sandbox.list():
                print(f"  {info.path}  [{info.branch or 'detached'}]  {info.head or ''}")
            return 0
        if args.sand_command == "clean":
            removed = 0
            repo_path = Path(cfg.repo).resolve()
            root = Path(sandbox.sandbox_root).resolve()
            for info in sandbox.list():
                path = Path(info.path).resolve()
                # Never touch the main checkout: a repo named `dacode*` or a branch
                # containing "dacode" used to match its own clean rule.
                if path == repo_path:
                    continue
                if not path.is_relative_to(root):
                    continue
                sandbox.remove(path.name, delete_branch=True, force=True)
                removed += 1
            print(f"removed {removed} trial worktree(s)")
            return 0

    if args.command == "state":
        from dacode import state

        if args.state_command == "search":
            hits = state.search(args.query, limit=args.limit)
            if not hits:
                print("no matching history")
                return 1
            for hit in hits:
                when = time.strftime("%Y-%m-%d %H:%M", time.localtime(hit["ts"]))
                print(f"[{when}] {hit['kind']:<7} {hit['task'][:90]}")
                if hit["label"] or hit["detail"]:
                    print(f"           {hit['label']}  {hit['detail']}")
            return 0
        if args.state_command == "stats":
            info = state.stats()
            if not info.get("enabled"):
                print("state store disabled (DACODE_MEMORY_DB=off)")
                return 1
            for key, value in info.items():
                print(f"  {key:<15}: {value}")
            return 0
        if args.state_command == "failures":
            rows = state.failures(args.task, limit=args.limit)
            if not rows:
                print("no failed approaches recorded")
                return 1
            for row in rows:
                print(
                    f"  {row['tests_passed']}/{row['tests_total']}  "
                    f"{str(row['label']):<24} {row['task'][:80]}"
                )
            return 0
        if args.state_command == "rebuild":
            count = state.rebuild()
            print(f"rebuilt from {count} runlog event(s)")
            return 0

    if args.command == "kanban":
        from dacode.kanban import HermesKanban, make_rig_runner, work

        cfg = _make_config(args)
        # The claim must outlive every round, or a second worker can take the card.
        # Each variant can burn a full timeout in the executor AND another in the
        # critic's gate (plus the baseline measurement before round one) — a TTL
        # sized for the executor alone loses the claim mid-gate on slow suites.
        per_variant = 2 * cfg.variant_timeout
        ttl = args.ttl or int(cfg.budget * cfg.variants_per_round * per_variant + cfg.variant_timeout) + 600
        return work(
            args.task_id,
            HermesKanban(args.board),
            make_rig_runner(cfg, explicit_repo=getattr(args, "repo", None)),
            ttl=ttl,
        )

    if args.command == "run":
        cfg = _make_config(args)
        rig = DaCodeRig(cfg)
        result = rig.run_task(args.task)
        _print_result(result)
        return 0 if result.passed else 1

    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
