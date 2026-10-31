"""`dacode kanban work` CLI wiring — parser shape and dispatch, no board, no rig."""

from __future__ import annotations

import pytest

from dacode import cli


def test_kanban_work_parses_card_board_and_run_options():
    args = cli.build_parser().parse_args(
        ["kanban", "work", "t_1", "--board", "b", "--budget", "1", "--isolate-network"]
    )
    assert (args.command, args.kanban_command, args.task_id, args.board) == ("kanban", "work", "t_1", "b")
    assert args.budget == 1 and args.isolate_network is True


def test_kanban_work_dispatches_to_work_with_a_ttl_covering_the_run(monkeypatch, tmp_path):
    seen = {}

    def fake_work(task_id, board, run_card, *, ttl):
        seen.update(task_id=task_id, board=board.board, ttl=ttl, run_card=run_card)
        return 3

    import dacode.kanban as kanban

    monkeypatch.setattr(kanban, "work", fake_work)
    rc = cli.main(
        ["--repo", str(tmp_path), "kanban", "work", "t_9", "--board", "b",
         "--budget", "2", "--variants", "2", "--variant-timeout", "100"]
    )
    assert rc == 3
    assert seen["task_id"] == "t_9" and seen["board"] == "b"
    # Each variant can burn a full timeout in the executor AND another in the
    # critic's gate; the baseline spends one more before round one.
    assert seen["ttl"] == 2 * 2 * (2 * 100) + 100 + 600
    assert callable(seen["run_card"])


def test_kanban_work_honours_an_explicit_ttl(monkeypatch, tmp_path):
    seen = {}
    import dacode.kanban as kanban

    monkeypatch.setattr(kanban, "work", lambda *a, ttl: seen.setdefault("ttl", ttl) and 0)
    cli.main(["--repo", str(tmp_path), "kanban", "work", "t_9", "--ttl", "77"])
    assert seen["ttl"] == 77


def test_kanban_work_refuses_a_non_positive_ttl(tmp_path):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            ["--repo", str(tmp_path), "kanban", "work", "t_9", "--ttl", "0"]
        )
