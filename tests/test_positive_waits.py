"""A wait a test expects to succeed gets seconds of margin, not a guess.

CI's runners are slow (AGENTS.md, Tests): a Windows setup once took 13 s,
and a positive wait - an event expected to be set, a future expected to
answer, a thread expected to end, a poll expected to see its condition -
that allows 2 s fails there for no fault of the code. A positive wait
costs its timeout only when the test is already failing, so it allows 30 s
(docs/rules.md, Tests). A negative wait - one that proves something does
*not* happen yet - stays short, and says so: inside `assert not ...` or
`pytest.raises(...)`, which this scan reads as negative by itself, or named
in `NEGATIVE` below with the reason it is short.

The scan reads every Python file under tests/ and flags a literal timeout
under `FLOOR` seconds in:

- a call's `timeout=` keyword (any keyword naming a timeout);
- the first positional argument of `.wait()`, `.join()`, `.result()` and
  `.exception()`, and the second of `asyncio.wait_for()`;
- the positional timeout of a helper defined under tests/ with a parameter
  named for one (`until(predicate, 5)`), and that parameter's default;
- a deadline, `time.monotonic()` (or `perf_counter()`, `time()`) plus a
  literal;
- a `for _ in range(N)` loop around `sleep(s)`, whose poll lasts N x s.
"""
from __future__ import annotations

import ast
from pathlib import Path

TESTS = Path(__file__).resolve().parent
FLOOR = 10.0
POSITIVE = 30.0

# Waits that are short on purpose, by file, enclosing function and the
# flagged source, with the reason. A new short wait that proves a negative
# joins here; a positive one gets `POSITIVE` seconds instead.
NEGATIVE: dict[tuple[str, str, str], str] = {
    ("fakes.py", "FakeBackend._wait_for_hold", "hold.wait(0.01)"):
        "one slice of a polled hold; the loop around it waits on the event",
    ("test_conditioning.py", "test_two_meta_builds_never_overlap", "second.join(0.5)"):
        "proves the second build is still waiting its turn",
    ("test_converter_cache.py", "test_the_sweep_waits_for_a_conversion_holding_the_lock", "worker.join(0.5)"):
        "proves the resolve is still waiting on the held lock",
    ("test_critic_install.py", "test_the_sweep_waits_for_an_install_holding_the_lock", "worker.join(0.5)"):
        "proves the install is still waiting on the held lock",
    ("test_single_file.py", "test_the_sweep_waits_for_a_conversion_holding_the_lock", "worker.join(0.5)"):
        "proves the resolve is still waiting on the held lock",
    ("test_critic_process.py", "_listening",
     "socket.create_connection(('127.0.0.1', port), timeout=0.5)"):
        "a probe that may be refused: Windows retries a refused loopback connect for 2 s",
    ("test_cli.py", "test_a_server_that_would_not_say_is_refused_in_its_own_words", "2.0"):
        "a stand-in's signature mirroring the probe it replaces; it never waits",
}

_POSITIONAL = {"wait": 0, "join": 0, "result": 0, "exception": 0}
_CLOCKS = {"monotonic", "perf_counter", "time"}


def _seconds(node: ast.AST | None) -> float | None:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return float(node.value)
    return None


def _helpers(trees: dict[Path, ast.Module]) -> dict[str, int]:
    """Each helper under tests/ taking a timeout, and its positional index."""
    found: dict[str, int] = {}
    for tree in trees.values():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names = [arg.arg for arg in node.args.posonlyargs + node.args.args]
                if names[:1] == ["self"]:
                    names = names[1:]
                for index, name in enumerate(names):
                    if "timeout" in name:
                        found.setdefault(node.name, index)
    return found


class _Scan(ast.NodeVisitor):
    def __init__(self, path: str, helpers: dict[str, int]):
        self.path, self.helpers = path, helpers
        self.scope: list[str] = []
        self.negative = 0
        self.found: list[tuple[str, int, str, str, float]] = []

    def flag(self, node: ast.AST, literal: ast.AST, seconds: float | None = None) -> None:
        if seconds is None:
            seconds = _seconds(literal)
        if seconds is None or seconds <= 0 or seconds >= FLOOR or self.negative:
            return
        self.found.append((
            self.path, node.lineno, ".".join(self.scope) or "<module>", ast.unparse(node), seconds,
        ))

    def visit_FunctionDef(self, node):
        args = node.args
        positional = args.posonlyargs + args.args
        for arg, default in zip(positional[len(positional) - len(args.defaults):], args.defaults):
            if "timeout" in arg.arg:
                self.flag(default, default)
        for arg, default in zip(args.kwonlyargs, args.kw_defaults):
            if "timeout" in arg.arg:
                self.flag(default, default)
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_UnaryOp(self, node):
        negated = isinstance(node.op, ast.Not)
        self.negative += negated
        self.generic_visit(node)
        self.negative -= negated

    def visit_With(self, node):
        raises = any(
            isinstance(item.context_expr, ast.Call)
            and ast.unparse(item.context_expr.func).rpartition(".")[2] == "raises"
            for item in node.items
        )
        self.negative += raises
        self.generic_visit(node)
        self.negative -= raises

    visit_AsyncWith = visit_With

    def visit_Call(self, node):
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        for keyword in node.keywords:
            if keyword.arg and "timeout" in keyword.arg:
                self.flag(node, keyword.value)
        index = None
        if name == "wait_for" and ast.unparse(func) == "asyncio.wait_for":
            index = 1
        elif name in _POSITIONAL and isinstance(func, ast.Attribute):
            index = _POSITIONAL[name]
        elif name in self.helpers:
            index = self.helpers[name]
        if index is not None and len(node.args) > index:
            self.flag(node, node.args[index])
        self.generic_visit(node)

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Add):
            for clock, other in ((node.left, node.right), (node.right, node.left)):
                if (
                    isinstance(clock, ast.Call) and isinstance(clock.func, ast.Attribute)
                    and clock.func.attr in _CLOCKS
                ):
                    self.flag(node, other)
        self.generic_visit(node)

    def visit_For(self, node):
        loop = node.iter
        if (
            isinstance(loop, ast.Call) and getattr(loop.func, "id", "") == "range"
            and len(loop.args) == 1 and _seconds(loop.args[0])
        ):
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute)
                    and inner.func.attr == "sleep" and inner.args and _seconds(inner.args[0])
                ):
                    self.flag(loop, loop, _seconds(loop.args[0]) * _seconds(inner.args[0]))
                    break
        self.generic_visit(node)


def short_waits(root: Path = TESTS) -> list[tuple[str, int, str, str, float]]:
    """Every literal timeout under FLOOR the scan does not read as negative."""
    trees = {
        path: ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for path in sorted(root.rglob("*.py"))
    }
    helpers = _helpers(trees)
    found = []
    for path, tree in trees.items():
        scan = _Scan(path.relative_to(root).as_posix(), helpers)
        scan.visit(tree)
        found.extend(scan.found)
    return found


def test_every_positive_wait_has_seconds_of_margin():
    found = short_waits()
    flagged = [
        f"tests/{path}:{line} in {scope}: {source} ({seconds:g} s)"
        for path, line, scope, source, seconds in found
        if (path, scope, source) not in NEGATIVE
    ]
    assert not flagged, (
        f"A wait expected to succeed allows {POSITIVE:g} s (docs/rules.md, Tests); a short "
        "one that proves a negative joins NEGATIVE with its reason:\n" + "\n".join(flagged)
    )
    # And every entry there still names a wait, so the list cannot go stale.
    gone = set(NEGATIVE) - {(path, scope, source) for path, _, scope, source, _ in found}
    assert not gone, f"NEGATIVE names waits that are gone: {sorted(gone)}"


def test_the_scan_reads_each_form_and_its_negatives(tmp_path):
    (tmp_path / "test_sample.py").write_text(
        "import asyncio, time\n"
        "def until(predicate, timeout=5.0):\n"
        "    pass\n"
        "def test_forms(event, future, thread, queue):\n"
        "    event.wait(2)\n"
        "    future.result(timeout=3)\n"
        "    thread.join(1)\n"
        "    queue.get(timeout=0.5)\n"
        "    asyncio.wait_for(future, 2)\n"
        "    until(lambda: True, 4)\n"
        "    deadline = time.monotonic() + 3\n"
        "    for _ in range(100):\n"
        "        time.sleep(0.02)\n"
        "    event.wait(30)\n"
        "    assert not event.wait(0.1)\n"
        "    event.wait(0)\n",
        encoding="utf-8",
    )
    sources = [source for _, _, _, source, _ in short_waits(tmp_path)]
    assert sources == [
        "5.0", "event.wait(2)", "future.result(timeout=3)", "thread.join(1)",
        "queue.get(timeout=0.5)", "asyncio.wait_for(future, 2)", "until(lambda: True, 4)",
        "time.monotonic() + 3", "range(100)",
    ]
