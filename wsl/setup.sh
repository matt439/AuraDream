#!/usr/bin/env bash
# One-time WSL 2 setup: uv and Python 3.13 if missing, a Linux environment on
# the Linux filesystem, and a check that CUDA and the Nunchaku packages are
# there. Safe to re-run. From the repo root, inside WSL:
#
#     bash wsl/setup.sh
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=env.sh
source wsl/env.sh

if ! grep -qi microsoft /proc/version 2>/dev/null; then
    echo "warning: not a WSL kernel; this assumes the Windows NVIDIA driver is mapped in" >&2
fi
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi does not work here. Update the Windows NVIDIA driver; do not install" >&2
    echo "a Linux driver inside WSL (docs.nvidia.com/cuda/wsl-user-guide)." >&2
    exit 1
fi

if ! command -v cc >/dev/null 2>&1 && ! command -v gcc >/dev/null 2>&1; then
    echo "note: no C compiler here. env.sh sets TORCH_DISABLE_NATIVE_JIT=1 so torch does not" >&2
    echo "      need one; 'sudo apt install gcc' lifts that if you ever want its Triton kernels." >&2
fi

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

mkdir -p "$KREA_OUTPUTS_DIR" "$(dirname "$KREA_DB_PATH")"
uv python install 3.13
uv sync                        # Linux cu130 torch, plus kernels and bitsandbytes

uv run python - <<'PY'
import sys
from importlib.metadata import version

import torch

ok = torch.cuda.is_available()
print(f"python {sys.version.split()[0]}  torch {torch.__version__}  cuda {'ok' if ok else 'MISSING'}")
if ok:
    cap = torch.cuda.get_device_capability()
    fp4 = "nvfp4 ok" if cap >= (10, 0) else "pre-Blackwell: use the int4 repos"
    print(f"{torch.cuda.get_device_name(0)}  sm_{cap[0]}{cap[1]}  {fp4}")
print(f"kernels {version('kernels')}  bitsandbytes {version('bitsandbytes')}")
PY

cat <<MSG

Ready. Every new WSL shell needs the environment first:

    source wsl/env.sh          # or put that line in ~/.bashrc
    bash wsl/serve.sh          # then open http://127.0.0.1:7860 in a Windows browser

The gated official repos (krea/Krea-2-Turbo, FLUX.1-Krea-dev) also need
'uv run hf auth login' in here; the lite-infer re-uploads do not.
MSG
