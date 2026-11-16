"""Cold refusal boundaries; actual process/restart evidence is recorded separately."""
import socket

import pytest

from tools.recovery_restart_probe import run


@pytest.mark.parametrize("browser_finish,history_review", [(False, None), (True, None), (False, "desktop"), (False, "phone")])
def test_existing_path_refused_without_touching(tmp_path, browser_finish, history_review):
    marker = tmp_path / "keep.txt"
    marker.write_text("untouched")
    with pytest.raises(ValueError, match="new scratch"):
        run(tmp_path, 8346, browser_finish=browser_finish, history_review=history_review)
    assert marker.read_text() == "untouched"
    assert list(tmp_path.iterdir()) == [marker]


@pytest.mark.parametrize("port", [0, 1023, 65536])
def test_invalid_port_refused_before_creation(tmp_path, port):
    data = tmp_path / "new"
    with pytest.raises(ValueError):
        run(data, port)
    assert not data.exists()


def test_occupied_port_refused_before_creation(tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        data = tmp_path / "new"
        with pytest.raises(ValueError, match="occupied"):
            run(data, listener.getsockname()[1])
        assert not data.exists()


def test_open_tab_handshakes_require_truthful_state_before_restart():
    from tools.recovery_restart_probe import open_tab_steps
    steps = open_tab_steps(8357)
    assert sum("navigate" in step for step in steps) == 1
    assert steps[0]["navigate"] == "http://127.0.0.1:8357/"
    assert all(step.get("fatal") is True for step in steps if "expect" in step)
    offline = next(i for i, step in enumerate(steps) if step.get("log") == "'OPEN_TAB_OFFLINE'")
    assert "serverOffline && streamLost" in steps[offline - 2]["expect"]
    assert "els.stepRow.hidden" in steps[offline - 1]["expect"]
    assert "jobEta==='" in steps[offline - 1]["expect"]
    assert "serverStartedAt!==__oldStarted" in steps[offline + 1]["expect"]
    assert "jobs.length===0" in steps[-2]["expect"]
    assert steps[-1]["log"] == "'OPEN_TAB_RECOVERED'"
