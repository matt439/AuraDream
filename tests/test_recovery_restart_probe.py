"""Cold refusal boundaries; actual process/restart evidence is recorded separately."""
import socket

import pytest

from tools.recovery_restart_probe import run


@pytest.mark.parametrize("browser_finish", [False, True])
def test_existing_path_refused_without_touching(tmp_path, browser_finish):
    marker = tmp_path / "keep.txt"
    marker.write_text("untouched")
    with pytest.raises(ValueError, match="new scratch"):
        run(tmp_path, 8346, browser_finish=browser_finish)
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
