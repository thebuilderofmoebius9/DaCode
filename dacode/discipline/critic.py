"""The pstack Critic — combines static checks with a real test run."""

from __future__ import annotations

import fnmatch
import re
import subprocess
import tempfile
import xml.etree.ElementTree as xml
import os
import sys
from dataclasses import dataclass, field, replace as dataclass_replace
from pathlib import Path

from dacode.discipline.checks import Finding, check_python_test
from dacode.sandbox.bwrap import (
    IsolationPolicy,
    IsolationUnavailable,
    interpreter_binds,
    wrap_command,
)

__all__ = ["CritiqueReport", "PstackCritic"]


@dataclass
class CritiqueReport:
    passed: bool
    findings: list[Finding] = field(default_factory=list)
    tests_total: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    tests_skipped: int = 0
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    score: float = 0.0
    ran: bool = False

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    def summary(self) -> str:
        if not self.ran:
            return "not run"
        bits = [f"tests {self.tests_passed}/{self.tests_total} pass"]
        if self.errors:
            bits.append(f"{len(self.errors)} error(s)")
        if self.warnings:
            bits.append(f"{len(self.warnings)} warning(s)")
        return ", ".join(bits)


class PstackCritic:
    """Review a workspace against pstack principles.

    * static scan of every ``test_*.py`` for weak/tautological assertions
    * run the configured test command; a clean exit is required to pass
    """

    def __init__(
        self,
        *,
        test_command: tuple[str, ...] | list[str] | None = None,
        test_glob: str = "test_*.py",
        timeout: float = 300.0,
        min_score: float = 0.6,
        isolation: IsolationPolicy | None = None,
        gate_infra: tuple[str, ...] | None = None,
    ) -> None:
        # The gate must own its output contract: a repo pyproject that sets
        # `addopts = "-q"` compounds with our own `-q` into `-qq` (verbosity −2),
        # which suppresses the `N passed` summary line _parse_test_summary needs —
        # a fully passing suite then reads as 0/0 collected ("no tests"). Clearing
        # ini addopts keeps the gate's summary deterministic regardless of repo
        # config; plugins still autoload, only addopts flags are cleared.
        self.test_command = tuple(test_command) if test_command else (sys.executable, "-m", "pytest", "-q", "-o", "addopts=")
        self.test_glob = test_glob
        self.timeout = timeout
        self.min_score = min_score
        self.isolation = isolation
        # Files that decide what the gate sees and what it reports. A trial that
        # touches them can pass without doing the work (verified: a conftest that
        # prints a fake summary and calls os._exit(0)).
        self.gate_infra = tuple(gate_infra or DEFAULT_GATE_INFRA)

    def _scan_tests(self, workspace: Path) -> tuple[list[Finding], int]:
        findings: list[Finding] = []
        count = 0
        for path in sorted(workspace.rglob(self.test_glob)):
            if any(part in {".git", ".venv", "node_modules", "__pycache__"} for part in path.parts):
                continue
            # The static checks are AST-based and Python-only.  A non-Python
            # suite (C, JS, …) still gets the real test gate below, but must
            # not be fed to the Python parser.
            if path.suffix != ".py":
                continue
            found = check_python_test(path)
            count += 1
            findings.extend(found)
        return findings, count

    def _static_test_count(self, repo: Path) -> int:
        """Count `def test_*` functions by reading the files.

        A contract-first suite that imports a module which does not exist yet fails
        at collection, so the dynamic count is 1 error (or nothing) and the baseline
        would silently switch the lost/added-test guards off. The static count does
        not care whether the code runs (audit H1).
        """
        import ast

        from dacode.discipline.checks import _TestVisitor

        total = 0
        for path in Path(repo).rglob("*.py"):
            try:
                relative = path.relative_to(repo)
            except ValueError:
                continue
            if any(part in {".git", ".venv", "node_modules", "__pycache__"} for part in relative.parts):
                continue
            if not fnmatch.fnmatch(path.name, self.test_glob):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
            except (OSError, SyntaxError):
                continue
            visitor = _TestVisitor(str(relative))
            visitor.visit(tree)
            total += len(visitor.test_functions)
        return total

    def measure_baseline(self, repo: Path) -> tuple[int, int] | None:
        """Count (total, skipped) on the untouched repo, once per run.

        A trial that deletes tests or silences them with skips keeps the suite
        green without proving anything; the baseline is what makes that visible.
        The gate runs with the repo bound read-only: this is a measurement on
        the live main checkout, not a trial worktree — the suite under test is
        still untrusted code.
        """
        try:
            _exit, _out, _err, total, _failed, skipped, _note, _source = self._run_tests(
                Path(repo), read_only=True
            )
        except Exception:  # noqa: BLE001 - a missing baseline must never block a run
            return None
        static = self._static_test_count(Path(repo))
        if total <= 0 and static <= 0:
            return None
        # A collection error under-counts; trust whichever number is higher so the
        # guards never switch themselves off by accident.
        return (max(total, static), skipped)

    def _wrap(
        self, cmd: list[str], workspace: Path, *, policy=None, read_only: bool = False
    ) -> list[str]:
        """Run the gate under bwrap when isolation is on.

        A suite is model-written code too: it must not reach the network, the
        user's home, or anything outside its own worktree. ``read_only`` binds
        the workspace read-only — for the baseline, which measures the live
        main checkout instead of a disposable trial worktree.
        """
        active = policy if policy is not None else self.isolation
        if active is None or not active.enabled:
            return cmd
        policy = active.with_paths(
            *interpreter_binds(self.test_command),
            os.environ.get("VIRTUAL_ENV"),
        )
        # Fail closed: an isolated run that silently fell back to the host would be
        # worse than a failed gate. The exception is re-raised as a failed probe by
        # _run_tests (exit 125) so the caller sees a verdict, not a traceback.
        return wrap_command(
            cmd, workspace=workspace, policy=policy, read_only_workspace=read_only
        )

    def _run_once(
        self, command: Sequence[str], workspace: Path, *, read_only: bool = False
    ) -> tuple[int, str, str, int, int, int, str | None, bool]:
        """One gate run. Counts come from a JUnit report when pytest can write one.

        The last element is ``from_report``: the counts were read from the JUnit
        report written outside the workspace, not parsed from the run's own
        stdout. Callers use it to refuse stdout-only counts where a real report
        is always available (see review_workspace).
        """
        import shutil

        junit_dir: Path | None = None
        junit: Path | None = None
        cmd = list(command)
        policy = self.isolation
        if _pytest_like(cmd):
            junit_dir = Path(tempfile.mkdtemp(prefix="dacode-junit-"))
            junit = junit_dir / "report.xml"
            if len(cmd) >= 3 and cmd[0] == "bash" and cmd[1] == "-c":
                # `bash -c '<script>' --junitxml=…` hands the flag to bash as a
                # positional, not to pytest — the report never gets written and
                # every green run reads as counts_unavailable. Put the flag
                # inside the script where the test runner can see it.
                cmd[2] = f"{cmd[2]} --junitxml={junit}"
            else:
                cmd = [*cmd, f"--junitxml={junit}"]
            if policy is not None and policy.enabled:
                policy = dataclass_replace(
                    policy, writable_paths=(*policy.writable_paths, junit_dir)
                )
        try:
            proc = subprocess.run(
                self._wrap(cmd, workspace, policy=policy, read_only=read_only),
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=_gate_env(),
            )
        except IsolationUnavailable as exc:
            return 125, "", f"isolation unavailable: {exc}", 0, 0, 0, None, False
        except FileNotFoundError as exc:
            return 127, "", f"test command not found: {exc}", 0, 0, 0, None, False
        except subprocess.TimeoutExpired as exc:
            return 124, exc.stdout or "", f"timed out after {self.timeout}s", 0, 0, 0, None, False
        finally:
            pass

        xml_counts = None
        junit_shaped = False
        counts_source = "stdout"
        if junit is not None:
            parsed = _junit_counts(junit, stdout=proc.stdout)
            if parsed is not None:
                xml_counts, junit_shaped = parsed
        stdout_counts = _parse_test_counts(proc.stdout)
        if stdout_counts[0] == 0 and proc.stderr:
            # unittest writes its progress dots and summary to stderr, not
            # stdout: a fully green `python -m unittest` suite used to read as
            # "no tests were collected" (0/0) and every honest trial failed.
            # Only fall back when stdout is silent, so pytest summaries — and
            # anything printed around them — keep meaning exactly what they meant.
            stderr_counts = _parse_test_counts(proc.stderr)
            if stderr_counts[0] > 0:
                stdout_counts = stderr_counts
                if _parse_unittest_summary(_strip_ansi(proc.stderr).splitlines()) is not None:
                    counts_source = "unittest_summary"
        note = None
        if (
            xml_counts is not None
            and xml_counts[0] > 0
            and stdout_counts[0] > 0        # silence is not a disagreement
            and not junit_shaped            # subTest totals count units, not elements
            and xml_counts != stdout_counts
        ):
            # The report is written by pytest itself outside the workspace; a
            # disagreement means something is printing summaries of its own.
            note = f"counts_disagree:junit={xml_counts} stdout={stdout_counts}"
        if (
            junit is not None
            and proc.returncode == 0
            and (xml_counts is None or xml_counts[0] == 0)
        ):
            # pytest was asked for a report and a "green" run did not produce one:
            # something ended the process before pytest could write it.
            note = (note + " " if note else "") + "counts_unavailable:no junit report"
        counts = xml_counts if (xml_counts and xml_counts[0] > 0) else stdout_counts
        if xml_counts and xml_counts[0] > 0:
            counts_source = "report"
        if junit_dir is not None:
            shutil.rmtree(junit_dir, ignore_errors=True)
        return proc.returncode, proc.stdout, proc.stderr, *counts, note, counts_source

    def _run_tests(
        self, workspace: Path, *, read_only: bool = False
    ) -> tuple[int, str, str, int, int, int, str | None, bool]:
        result = self._run_once(list(self.test_command), workspace, read_only=read_only)
        if result[3] == 0 and result[0] == 0:
            # A green run with no counts usually means the project's own addopts
            # already contain -q ("-qq" prints nothing). Re-run with them neutralised.
            retried = self._neutralised_run(workspace, read_only=read_only)
            if retried is not None:
                return retried
        return result

    def _neutralised_run(
        self, workspace: Path, *, read_only: bool = False
    ) -> tuple[int, str, str, int, int, int, str | None, bool] | None:
        """Re-run with ``-o addopts=`` so a quiet project config cannot look like 0 tests."""
        if not self.test_command or not _pytest_like(self.test_command):
            # ``-o addopts=`` is a pytest flag: appending it to a unittest or
            # custom command breaks the re-run instead of un-quieting it.
            return None
        cmd = [*self.test_command, "-o", "addopts="]
        result = self._run_once(cmd, workspace, read_only=read_only)
        if result[3] == 0:
            return None
        return result

    def _is_contract_path(self, relative: Path, *, test_dirs: set[str]) -> bool:
        """Is this path part of what decides the gate's answer?

        Basenames alone were not enough (verification audit, Oct 2026): a trial can
        add ``pytest/__init__.py`` to shadow the runner, or rewrite
        ``tests/helpers.py`` to make every assertion trivially true. So: gate-infra
        basenames, a shadowing path component, a conftest anywhere, and *any* file
        inside a directory that holds tests.
        """
        name = relative.name
        if name in self.gate_infra:
            return True
        if (
            name.startswith("conftest")
            or name.startswith("test_")
            or name.startswith("_test")
            or name.endswith("_test.py")
        ):
            return True
        if fnmatch.fnmatch(name, self.test_glob):
            return True
        if name.endswith(".py") and relative.stem in SHADOW_NAMES:
            return True
        if any(part in SHADOW_NAMES for part in relative.parts[:-1]):
            return True
        if name == "entry_points.txt" or any(
            part.endswith((".dist-info", ".egg-info")) for part in relative.parts
        ):
            return True
        if name.endswith(".py") and str(relative.parent) in test_dirs:
            return True
        return False

    def _test_dirs(self, root: Path) -> set[str]:
        dirs: set[str] = set()
        for path in root.rglob("*"):
            if not path.is_file() or not fnmatch.fnmatch(path.name, self.test_glob):
                continue
            relative = path.parent.relative_to(root)
            # A dedicated test directory is contract; the repository root is not,
            # or every source file a trial writes there would look like a test
            # helper (it cannot be told apart from one by path alone).
            if str(relative) != ".":
                dirs.add(str(relative))
        return dirs

    def snapshot_contract(self, workspace: str | Path) -> dict[str, str]:
        """Hash the test files and gate infrastructure as they are *now*.

        Taken before the executor runs, on the untouched worktree. Comparing hashes
        afterwards is the only check a trial cannot reach around by committing: the
        git-status route compares against the worktree's own index, which the trial
        can move.
        """
        import hashlib

        root = Path(workspace)
        contract: dict[str, str] = {}
        for path in sorted(root.rglob("*")):
            is_link = path.is_symlink()
            if not is_link and not path.is_file():
                continue
            relative = path.relative_to(root)
            # Interpreter caches appear the moment *any* pytest runs — including
            # the trial's own — and are not suite logic: a stale .pyc is ignored
            # while its source differs, and source-less imports do not load from
            # __pycache__. Contract-hashing them fails every honest trial.
            if any(part in {".git", "__pycache__", ".pytest_cache"} for part in relative.parts):
                continue
            if not self._is_contract_path(relative, test_dirs=self._test_dirs(root)):
                continue
            try:
                raw = (
                    f"symlink:{os.readlink(path)}".encode("utf-8")
                    if is_link
                    else path.read_bytes()
                )
                contract[str(relative)] = hashlib.sha256(raw).hexdigest()
            except OSError:
                continue
        return contract

    def _check_contract(
        self, workspace: Path, contract: dict[str, str]
    ) -> list[Finding]:
        """Compare the workspace against a snapshot taken before the trial ran."""
        import hashlib

        findings: list[Finding] = []
        current = self.snapshot_contract(workspace)
        test_dirs = self._test_dirs(workspace)

        def _check_for(relative: str) -> str:
            path = Path(relative)
            # a file inside a test directory, or a test module, is the suite itself
            if fnmatch.fnmatch(path.name, self.test_glob) or str(path.parent) in test_dirs:
                return "tests_modified"
            return "tests_infra_modified"

        for path, digest in contract.items():
            check = _check_for(path)
            if path not in current:
                findings.append(
                    Finding(
                        check=check,
                        severity="error",
                        message=f"trial removed {path} — the gate must judge the suite it was given",
                        location=str(workspace / path),
                    )
                )
            elif current[path] != digest:
                findings.append(
                    Finding(
                        check=check,
                        severity="error",
                        message=f"trial changed {path} — the gate must judge unedited tests",
                        location=str(workspace / path),
                    )
                )

        for path in sorted(set(current) - set(contract)):
            check = _check_for(path)
            findings.append(
                Finding(
                    check=check,
                    severity="error",
                    message=(
                        f"trial added {path} — a new test or config file changes what the gate "
                        "sees; a human must review it"
                    ),
                    location=str(workspace / path),
                )
            )
        return findings

    def _check_test_integrity(self, workspace: Path) -> list[Finding]:
        """A trial may not edit, add or delete a test file.

        The gate's ground truth is the test suite it was handed. A trial that
        rewrites an assertion can turn a red gate green without changing
        behaviour — that is a false PASS, and it happened once (a trial changed
        ``approx(100.0)`` to ``approx(105.0)``). Only git workspaces can be
        checked; elsewhere this is a no-op.
        """
        try:
            proc = subprocess.run(
                [
                    "git",
                    "-c", "core.hooksPath=/dev/null",
                    "-c", "core.fsmonitor=false",
                    "-c", "core.attributesFile=/dev/null",
                    "status", "--porcelain", "--untracked-files=all",
                ],
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return [
                Finding(
                    check="integrity_unavailable",
                    severity="error",
                    message="could not read the worktree's git status — refusing to certify unedited tests",
                    location=str(workspace),
                )
            ]
        if proc.returncode != 0:
            if "not a git repository" in (proc.stderr or "").lower():
                return []  # standalone use on a plain directory: nothing to compare
            return [
                Finding(
                    check="integrity_unavailable",
                    severity="error",
                    message="git status failed in the worktree — refusing to certify unedited tests",
                    location=str(workspace),
                )
            ]

        findings: list[Finding] = []
        test_dirs = self._test_dirs(workspace)
        for line in proc.stdout.splitlines():
            if len(line) < 4:
                continue
            path = line[3:].strip().strip('"')
            # same carve-out as snapshot_contract: bytecode caches appear the
            # moment any pytest (the trial's own included) touches the tree
            if any(part in {"__pycache__", ".pytest_cache"} for part in Path(path).parts):
                continue
            name = Path(path).name
            in_test_dir = str(Path(path).parent) in test_dirs
            if fnmatch.fnmatch(name, self.test_glob) or in_test_dir:
                findings.append(
                    Finding(
                        check="tests_modified",
                        severity="error",
                        message=f"trial modified a test file ({path}) — the gate must judge unedited tests",
                        location=str(workspace / path),
                    )
                )
            elif name in self.gate_infra:
                findings.append(
                    Finding(
                        check="tests_infra_modified",
                        severity="error",
                        message=(
                            f"trial modified gate infrastructure ({path}) — a conftest or a "
                            "pytest config can silence or spoof the gate itself; a human must "
                            "review that change, not a trial"
                        ),
                        location=str(workspace / path),
                    )
                )
        return findings

    def review_workspace(
        self,
        workspace: str | Path,
        *,
        baseline: tuple[int, int] | None = None,
        contract: dict[str, str] | None = None,
    ) -> CritiqueReport:
        workspace = Path(workspace)
        findings, py_test_files = self._scan_tests(workspace)
        if contract is not None:
            findings.extend(self._check_contract(workspace, contract))
        else:
            findings.extend(self._check_test_integrity(workspace))
        exit_code, stdout, stderr, total, failed, skipped, counts_note, counts_source = (
            self._run_tests(workspace)
        )
        if counts_note and "counts_unavailable" in counts_note:
            findings.append(
                Finding(
                    check="counts_unavailable",
                    severity="error",
                    message=(
                        "pytest was asked for a JUnit report and produced none on a green "
                        f"run — the process was ended early ({counts_note})"
                    ),
                    location=str(workspace),
                )
            )
        elif counts_note:
            findings.append(
                Finding(
                    check="counts_disagree",
                    severity="error",
                    message=(
                        "the test runner's own report disagrees with its stdout — "
                        f"something is printing a summary ({counts_note})"
                    ),
                    location=str(workspace),
                )
            )

        # A sandbox hides $HOME by design, so an interpreter whose packages are
        # a pip --user install gets an opaque "No module named X" instead of a
        # verdict. Name the cause and the two clean ways out — the sandbox does
        # not open for this (the home is hidden on purpose), but the message
        # should not be a riddle (issue #9).
        # pytest reports collection-time import errors on stdout, not stderr —
        # scan both.
        run_text = f"{stdout}\n{stderr}"
        if exit_code != 0 and self.isolation is not None and self.isolation.enabled:
            match = re.search(r"No module named (?:'([^']+)'|([\w.]+))", run_text)
            if match:
                module = (match.group(1) or match.group(2)).split(".")[0]
                home = Path.home()
                if (
                    module
                    and _module_lives_under_home(module, home)
                    and not (workspace / module).exists()
                ):
                    findings.append(
                        Finding(
                            check="hidden_home_packages",
                            severity="error",
                            message=(
                                f"'{module}' is installed under your home directory "
                                "(pip --user), which the sandbox hides by design — the "
                                "suite cannot see it. Point the test command at a "
                                "workspace-local venv (the workspace is bound) or a "
                                "system-installed runner."
                            ),
                            location=str(workspace),
                        )
                    )

        # A printed summary is the suite grading its own homework. For a Python
        # suite that is never necessary: pytest can always emit the JUnit report
        # the gate already attaches, so a green run counted only from stdout is
        # refused outright (SECURITY.md open item 1, enforced). Two exemptions,
        # both about what a suite *can* emit, not what it prints:
        # - a report-backed run is the authoritative source;
        # - a stdlib unittest suite has no report at all — its deterministic
        #   two-line stderr summary is the only count that exists (residual 7
        #   covers the forgery shape). Non-Python suites (C/JS/PHPUnit) keep
        #   the stdout count; the integrity and infrastructure guards still apply.
        if (
            counts_source == "stdout"
            and exit_code == 0
            and total > 0
            and py_test_files > 0
        ):
            findings.append(
                Finding(
                    check="counts_from_stdout",
                    severity="error",
                    message=(
                        "the only counts are the suite's own printed summary — a "
                        "Python suite must emit a real report: make the command "
                        "name pytest (the gate attaches --junitxml itself, also "
                        "inside `bash -c` scripts), or wrap the runner so pytest "
                        "writes one"
                    ),
                    location=str(workspace),
                )
            )

        if baseline is not None:
            base_total, base_skipped = baseline
            if total < base_total:
                findings.append(
                    Finding(
                        check="tests_lost",
                        severity="error",
                        message=(
                            f"the suite shrank: {total} test(s) now vs {base_total} at baseline "
                            "— deleting or disabling tests is not a fix"
                        ),
                        location=str(workspace),
                    )
                )
            if skipped > base_skipped:
                findings.append(
                    Finding(
                        check="tests_skipped",
                        severity="error",
                        message=(
                            f"{skipped} test(s) skipped vs {base_skipped} at baseline "
                            "— a skipped test proves nothing"
                        ),
                        location=str(workspace),
                    )
                )
        passed_count = max(0, total - failed)

        # pstack #9 — a report without a run is not a report.
        if total == 0:
            findings.append(
                Finding(
                    check="no_tests",
                    severity="error",
                    message="no tests were collected — the change is unverified",
                    location=str(workspace),
                )
            )

        penalty = 0.25 * len([f for f in findings if f.severity == "error"])
        penalty += 0.05 * len([f for f in findings if f.severity == "warning"])
        if exit_code not in (0, None):
            penalty += 0.35
        score = max(0.0, min(1.0, 1.0 - penalty))

        # `failed == 0` matters: counts come from the runner's stdout, so an exit
        # code of 0 is not enough on its own (see _check_test_integrity).
        passed = (
            exit_code == 0
            and total > 0
            and failed == 0
            and not [f for f in findings if f.severity == "error"]
        )
        passed = passed and score >= self.min_score

        return CritiqueReport(
            passed=passed,
            findings=findings,
            tests_total=total,
            tests_passed=passed_count,
            tests_failed=failed,
            tests_skipped=skipped,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            score=round(score, 3),
            ran=True,
        )


#: Path components that shadow or hook the runner when the workspace is `sys.path[0]`.
SHADOW_NAMES: tuple[str, ...] = (
    "pytest", "_pytest", "pluggy", "py", "unittest", "xml", "sitecustomize", "usercustomize",
)

DEFAULT_GATE_INFRA: tuple[str, ...] = (
    "conftest.py",
    "pytest.ini",
    ".pytest.ini",
    "tox.ini",
    "setup.cfg",
    "pyproject.toml",
    "setup.py",
    ".coveragerc",
    # modules that shadow or hook the runner inside the workspace
    "pytest.py",
    "_pytest.py",
    "pluggy.py",
    "py.py",
    "unittest.py",
    "sitecustomize.py",
    "usercustomize.py",
)


def _pytest_like(command: Sequence[str]) -> bool:
    """Is this command pytest? Only then can we ask for a JUnit report."""
    return any("pytest" in str(token) for token in command)


_PYTEST_SUMMARY_UNIT_RE = re.compile(
    r"(\d+)\s+(subtests?\s+(?:passed|failed)|xpassed|xfailed|errors?|passed|failed|skipped)\b"
)
_PYTEST_SUMMARY_LINE_RE = re.compile(r"\bin\s+\d+(?:\.\d+)?s\s*$")


def _pytest_summary_units(stdout: str) -> int | None:
    """Sum the per-outcome unit counts on pytest's final summary line, or None.

    ``unittest.subTest`` makes pytest report more units than it emits ``<testcase>``
    elements for (passing subtests are folded into the declared total with no element
    of their own, measured against pytest 9.1.1), so an element-only reading can never
    reach the declared ``tests=``. This parses the summary line the gate captured
    itself from the suite's stdout — "4 failed, 2 passed, 2 skipped, 11 subtests passed
    in 0.3s" — and sums the units, ``warning`` tokens excluded. Returns None when the
    output has no pytest summary line at all (silenced or non-pytest runner), so the
    caller cannot accept a shaped report on the strength of nothing.
    """
    for line in reversed(_strip_ansi(stdout).splitlines()):
        stripped = line.strip().strip("=").strip()
        if not stripped:
            continue
        if not _PYTEST_SUMMARY_LINE_RE.search(stripped):
            continue
        units = _PYTEST_SUMMARY_UNIT_RE.findall(stripped)
        if not units:
            continue
        return sum(int(count) for count, _outcome in units)
    return None


def _junit_counts(
    path: Path, stdout: str = ""
) -> tuple[tuple[int, int, int], bool] | None:
    """``((total, failed, skipped), subtest_shaped)`` from a JUnit XML report, or None.

    The report is written outside the workspace and parsed here on the host, so the
    numbers cannot be printed by the code under test. Two independent readings must
    agree: the counts are taken from the ``<testcase>`` elements (one element per
    test, which is what pytest emits) and cross-checked against the attributes the
    suite declares. A report whose declared totals do not match its own test cases is
    not a report — that is exactly the shape of a rewritten one (audit, Oct 2026:
    a module imported by the suite rewrote tests="1" to tests="999" at exit).

    One honest exception exists. ``unittest.subTest`` folds every passing subtest into
    the declared total without emitting an element for it (measured: a suite declaring
    tests="11" yields 6 elements, while failures and skips still reconcile exactly
    through ``<failure>``/``<skipped>`` children). A report may exceed its element
    total only when the failures and skips match exactly AND pytest's own summary
    line, captured by the gate from the run's stdout, accounts for the declared total
    unit by unit. The audited rewrite keeps failing that check — bumping an attribute
    produces no matching summary line — and a silenced stdout cannot corroborate
    anything, so both stay fail-closed.
    """
    try:
        tree = xml.parse(path)
    except (OSError, xml.ParseError):
        return None
    root = tree.getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    if not suites:
        return None

    declared_total = declared_failed = declared_skipped = 0
    for suite in suites:
        declared_total += int(suite.get("tests") or 0)
        declared_failed += int(suite.get("failures") or 0) + int(suite.get("errors") or 0)
        declared_skipped += int(suite.get("skipped") or 0)

    total = failed = skipped = 0
    for case in root.iter("testcase"):
        total += 1
        if case.find("failure") is not None or case.find("error") is not None:
            failed += 1
        if case.find("skipped") is not None:
            skipped += 1

    if (declared_total, declared_failed, declared_skipped) != (total, failed, skipped):
        if (
            declared_failed == failed
            and declared_skipped == skipped
            and declared_total > total
        ):
            units = _pytest_summary_units(stdout)
            if units is not None and units == declared_total:
                return (total, failed, skipped), True
        return None
    return (total, failed, skipped), False


def _gate_env() -> dict[str, str]:
    """Environment for the gate run.

    Plugin autoloading is off by default: a trial can drop a
    ``*.dist-info/entry_points.txt`` declaring a ``pytest11`` plugin and have it
    imported before collection, without touching any file the contract lists
    (external audit, Oct 2026). A repository that genuinely needs an installed
    plugin autoloaded can set ``DACODE_GATE_PLUGIN_AUTOLOAD=1``.
    """
    import os

    env = dict(os.environ)
    if os.environ.get("DACODE_GATE_PLUGIN_AUTOLOAD", "").lower() not in {"1", "true", "yes"}:
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return env


def _module_lives_under_home(module: str, home: Path) -> bool:
    """Does the host keep this package under ``$HOME`` (a pip --user install)?

    A user-site package is invisible to a sandboxed run (the home tmpfs hides
    it by design), which is a plausible cause of an otherwise opaque
    ``No module named X`` from a sandboxed suite. Layouts covered: pip's
    ``~/.local/lib/python*/site-packages`` (and lib64). Bounded on purpose —
    no whole-home scan.
    """
    parts = ("site-packages",)
    for lib in (home / ".local" / "lib", home / ".local" / "lib64"):
        if not lib.is_dir():
            continue
        for interp in lib.iterdir():
            if not interp.is_dir() or not interp.name.startswith("python"):
                continue
            site = interp / parts[0]
            if (site / module).is_dir() or (site / f"{module}.py").is_file():
                return True
    return False


def _parse_test_counts(stdout: str) -> tuple[int, int, int]:
    """``(total, failed, skipped)`` — see :func:`_parse_test_summary`.

    The skip count must come from the *same* summary line as the pass/fail counts.
    Searching the whole stdout let a trial print ``0 skipped`` early and mask real
    skips (external audit, verified).
    """
    import re

    total, failed = _parse_test_summary(stdout)
    skipped = 0
    for line in reversed(_strip_ansi(stdout).splitlines()):
        stripped = line.strip().strip("=").strip()
        if not stripped:
            continue
        if re.search(r"\d+\s+(passed|failed|checks?\b)", stripped, re.I) or "skipped" in stripped.lower():
            match = re.search(r"(\d+)\s+skipped", stripped)
            if match:
                skipped = int(match.group(1))
            else:
                # PHPUnit spells it "Skipped: 1", not "1 skipped"; unittest
                # puts the count after an equals sign: "OK (skipped=1)".
                match = re.search(r"\bSkipped:\s*(\d+)", stripped) or re.search(
                    r"\bskipped=(\d+)", stripped, re.I
                )
                if match:
                    skipped = int(match.group(1))
            break
    return total, failed, skipped


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _strip_ansi(text: str) -> str:
    """Remove colour escapes before parsing.

    PHPUnit paints its summary line: ``\\x1b[37;41mTests: 23\\x1b[0m...``. The escape
    ends in a letter, so ``\\bTests:`` never matches and a fully reported suite counts
    as 0 tests (verified against a real PHP project, Oct 2026).
    """
    return _ANSI_RE.sub("", text)


def _parse_unittest_summary(lines: list[str]) -> tuple[int, int] | None:
    """stdlib unittest's two-line summary: ``(total, failed)``, or None.

        ----------------------------------------------------------------------
        Ran 25 tests in 0.123s

        OK (skipped=1)                          green
        FAILED (failures=2, errors=1)           red

    The verdict line carries the failure counts, so it only counts together with
    its ``Ran`` line; a ``Ran`` with no verdict below means the run was cut short
    and stays uncounted — the zero-total "no tests" verdict is honest there.
    """
    import re

    for index in range(len(lines) - 1, -1, -1):
        match = re.search(r"\bRan\s+(\d+)\s+tests?\b", lines[index])
        if match is None:
            continue
        total = int(match.group(1))
        for verdict in lines[index + 1 : index + 4]:
            verdict = verdict.strip()
            if not verdict or set(verdict) <= {"-"}:
                continue
            if verdict.startswith("OK"):
                return total, 0
            if verdict.startswith("FAILED"):
                failed = 0
                for key in ("failures", "errors"):
                    sub = re.search(rf"\b{key}=(\d+)", verdict)
                    if sub:
                        failed += int(sub.group(1))
                return total, failed
            break  # there is a line, but it is not a verdict: not this shape
    return None


def _parse_test_summary(stdout: str) -> tuple[int, int]:
    """Extract ``(total, failed)`` from a test run's stdout.

    Understands pytest's summary line, stdlib unittest's two-line summary *and*
    the common custom-runner shapes, so a non-Python suite (Makefile/C, JS, …)
    is counted correctly too::

        "3 passed, 1 failed in 0.2s"        pytest
        "2 passed in 0.15s"                 pytest, all green
        "Ran 25 tests in 0.123s" + "OK"     unittest
        "90 checks, 0 failed"               Makefile / C suite
        "OK: 66 checks passed (headless)"   Makefile / C suite

    Returns ``(0, 0)`` when nothing recognisable is found; the caller treats a
    zero total as "no tests were collected".
    """
    import re

    unittest_counts = _parse_unittest_summary(_strip_ansi(stdout).splitlines())
    if unittest_counts is not None:
        return unittest_counts

    for line in reversed(_strip_ansi(stdout).splitlines()):
        stripped = line.strip().strip("=").strip()
        if not stripped:
            continue
        # PHPUnit: "Tests: 23, Assertions: 65, Failures: 2, Errors: 0, Skipped: 1."
        m = re.search(r"\bTests:\s*(\d+)", stripped)
        if m:
            total = int(m.group(1))
            failed = 0
            for key in ("Failures", "Errors"):
                sub = re.search(rf"\b{key}:\s*(\d+)", stripped)
                if sub:
                    failed += int(sub.group(1))
            return total, failed
        # PHPUnit, everything green: "OK (17 tests, 46 assertions)"
        m = re.search(r"\bOK\b.*?\((\d+)\s+tests?", stripped)
        if m:
            return int(m.group(1)), 0
        # "N checks, M failed"
        m = re.search(r"(\d+)\s+checks?\s*,\s*(\d+)\s+failed", stripped)
        if m:
            return int(m.group(1)), int(m.group(2))
        # "N checks passed" / "OK: N checks passed (headless)"
        m = re.search(r"(\d+)\s+checks?\s+passed", stripped)
        if m:
            return int(m.group(1)), 0
        # pytest: "3 passed, 1 failed in 0.2s"
        passed_m = re.search(r"(\d+)\s+passed", stripped)
        failed_m = re.search(r"(\d+)\s+failed", stripped)
        error_m = re.search(r"(\d+)\s+errors?", stripped)
        if passed_m or failed_m or error_m:
            total = 0
            failed = 0
            if passed_m:
                total += int(passed_m.group(1))
            if failed_m:
                total += int(failed_m.group(1))
                failed += int(failed_m.group(1))
            if error_m:
                total += int(error_m.group(1))
                failed += int(error_m.group(1))
            return total, failed
    return 0, 0


# Backwards-compatible alias — the old name was pytest-specific.
_parse_pytest_summary = _parse_test_summary
