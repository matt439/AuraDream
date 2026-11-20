"""No-model control for A4's post-unload CUDA residency, under the same external guard."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading
import time

OUT = Path(sys.argv[1]).resolve()
WORK = next(p / "work" for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())


def wait(marker):
    deadline = time.monotonic() + 60
    while not (OUT / marker).is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(marker)
        time.sleep(0.05)


def write(name, data):
    path = OUT / name
    part = path.with_suffix(".part")
    part.write_text(json.dumps(data, indent=2), encoding="utf-8")
    part.replace(path)


def main():
    if not __debug__ or not OUT.is_relative_to(WORK) or OUT == WORK:
        raise ValueError("Use the trial supervisor and an owned scratch directory.")
    wait("guard-admitted")
    import torch

    write("phase.json", {"name": "control", "started": time.time()})
    (OUT / "load-request").touch(exist_ok=False)
    wait("load-admitted")
    assert torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 5080"
    receipt = dict(
        torch=torch.__version__,
        torch_commit=torch.version.git_version,
        card=torch.cuda.get_device_name(0),
        observations={},
    )

    def read(label):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        active = [
            {k: block[k] for k in ("address", "size", "requested_size", "state")}
            for segment in torch.cuda.memory_snapshot()
            for block in segment["blocks"]
            if block["state"] != "inactive"
        ]
        receipt["observations"][label] = dict(
            allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(), active_blocks=active
        )
        write("control.json", receipt)

    def multiply():
        with torch.no_grad():
            a = torch.ones((512, 512), device="cuda:0", dtype=torch.float32)
            b = torch.mm(a, a)
            assert b[0, 0].item() == 512
        return threading.get_native_id()

    read("initial")
    receipt["workspace_size"] = torch.backends.cuda.cublas_workspace_size()
    receipt["main_thread"] = multiply()
    read("main_only")
    with ThreadPoolExecutor(max_workers=1) as worker:
        receipt["worker_thread"] = worker.submit(multiply).result(timeout=45)
        read("main_and_worker")
    read("worker_exited")
    # Diagnostic intervention isolated to this process. No model, app or production unload.
    torch._C._cuda_clearCublasWorkspaces()
    read("after_cublas_workspace_clear")
    (OUT / "unload-done").touch(exist_ok=False)
    wait("unload-observed")
    obs = receipt["observations"]
    ws = receipt["workspace_size"]
    assert receipt["main_thread"] != receipt["worker_thread"]
    assert ws == 32 * 2**20
    assert [
        obs[k]["allocated"]
        for k in ("initial", "main_only", "main_and_worker", "worker_exited", "after_cublas_workspace_clear")
    ] == [0, ws, 2 * ws, 2 * ws, 0]
    assert [x["size"] for x in obs["worker_exited"]["active_blocks"]] == [ws, ws]
    assert obs["after_cublas_workspace_clear"]["reserved"] == 0
    receipt["status"] = "complete"
    write("control.json", receipt)


if __name__ == "__main__":
    main()
