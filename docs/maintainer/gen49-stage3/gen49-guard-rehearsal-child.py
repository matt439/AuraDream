"""Disposable CPU-only IPC target for the resource supervisor."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

attempt = Path(sys.argv[1])


def wait(name):
    end = time.monotonic() + 15.
    while not (attempt / name).exists():
        if time.monotonic() > end:
            raise RuntimeError("IPC timeout: " + name)
        time.sleep(.02)


wait("guard-admitted")
(attempt / "phase.json").write_text(json.dumps({"name": "cpu-only", "started": time.time()}))
descendant = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"])
(attempt / "descendant-pid").write_text(str(descendant.pid))
(attempt / "load-request").touch()
wait("load-admitted")
(attempt / "unload-done").touch()
wait("unload-observed")
print(os.getpid(), "CPU markers complete; deliberately leaving owned descendant for job close")
