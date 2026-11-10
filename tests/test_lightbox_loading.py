"""Current-image ownership and bounded neighbour requests on cold DOM ports."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_current_thumbnail_loading_errors_and_bounded_neighbours():
    node = shutil.which("node")
    if node is None:
        pytest.fail("Node is required for the frontend checks")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([node, str(root / "tests/lightbox_loading_fixture.mjs"), str(root)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
