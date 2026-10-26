"""Run the pinned Python Pyright wrapper over the explicit repository scope."""
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
PYRIGHT_VERSION = "1.1.407"


def _scope(project: Path) -> None:
    config = json.loads(project.read_text(encoding="utf-8"))
    includes = config.get("include") if isinstance(config, dict) else None
    if not isinstance(includes, list) or not includes or any(not isinstance(path, str) or not path.strip() for path in includes):
        raise ValueError("the Pyright configuration needs an explicit, nonempty include list")
    for include in includes:
        source = project.parent / include
        if not ((source.is_file() and source.suffix == ".py") or (
            source.is_dir() and any(path.is_file() for path in source.rglob("*.py"))
        )):
            raise ValueError(f"the configured Pyright include has no Python sources: {include}")


def _run(arguments: list[str], root: Path) -> subprocess.CompletedProcess[str]:
    # pyright 1.1.407's Python wrapper otherwise checks PyPI for updates,
    # honors npm/Pylance version overrides, and can provision Node or Pyright.
    # These are its own controls (pyright/_utils.py and pyright/node.py).
    env = {
        **os.environ, "PYRIGHT_PYTHON_FORCE_VERSION": PYRIGHT_VERSION,
        "PYRIGHT_PYTHON_USE_BUNDLED_PYRIGHT": "1", "PYRIGHT_PYTHON_IGNORE_WARNINGS": "1",
        "PYRIGHT_PYTHON_GLOBAL_NODE": "1", "PYRIGHT_PYTHON_NODEJS_WHEEL": "0",
    }
    env.pop("PYRIGHT_PYTHON_PYLANCE_VERSION", None)
    return subprocess.run(
        [sys.executable, "-m", "pyright", *arguments], cwd=root,
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180, env=env,
    )


def _output(done: subprocess.CompletedProcess[str]) -> None:
    if done.stdout:
        print(done.stdout, end="" if done.stdout.endswith("\n") else "\n")
    if done.stderr:
        print(done.stderr, file=sys.stderr, end="" if done.stderr.endswith("\n") else "\n")


def main(project: Path | None = None) -> int:
    project = (project if project is not None else ROOT / "pyrightconfig.json").resolve()
    try:
        _scope(project)
        # The Python wrapper may provision Node when it is absent. Refuse
        # before invoking it: this gate must use the already-installed runtime.
        node_target = "node.exe" if sys.platform == "win32" else "node"
        node = shutil.which(node_target)
        if node is None or not Path(node).is_file():
            raise ValueError(f"Node is required on PATH ({node_target}); type checking never installs it")
        spec = importlib.util.find_spec("pyright")
        if spec is None or spec.origin is None:
            raise ValueError("Pyright is not installed in this Python environment; install the locked dev group")
        if importlib.metadata.version("pyright") != PYRIGHT_VERSION:
            raise ValueError(f"type checking requires Python package pyright=={PYRIGHT_VERSION}")
        bundled = Path(spec.origin).parent / "dist"
        package = json.loads((bundled / "package.json").read_text(encoding="utf-8"))
        if not (bundled / "index.js").is_file() or not isinstance(package, dict) or package.get("version") != PYRIGHT_VERSION:
            raise ValueError(f"the installed Python package needs its bundled Pyright {PYRIGHT_VERSION}; no npm install is attempted")
        version = _run(["--version"], project.parent)
        _output(version)
        if version.returncode:
            return version.returncode
        if version.stdout.strip() != f"pyright {PYRIGHT_VERSION}":
            raise ValueError(f"type checking requires Pyright {PYRIGHT_VERSION}")
        checked = _run(["--project", str(project), "--outputjson"], project.parent)
        _output(checked)
        try:
            report = json.loads(checked.stdout)
            summary = report["summary"]
            if not isinstance(report["generalDiagnostics"], list) or not isinstance(summary, dict):
                raise ValueError
            files, errors = summary["filesAnalyzed"], summary["errorCount"]
            if type(files) is not int or files <= 0 or type(errors) is not int or errors < 0:
                raise ValueError
        except (ValueError, KeyError, TypeError):
            print("Typecheck failed: Pyright returned malformed output or analyzed no files", file=sys.stderr)
            return checked.returncode or 2
        return checked.returncode or (1 if errors else 0)
    except (OSError, ValueError, importlib.metadata.PackageNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"Typecheck failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
