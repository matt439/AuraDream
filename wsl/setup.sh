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

installed_uv=0
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # For the rest of *this* script only: the export dies with it, and the
    # installer's rc-file edit reaches only future shells - so the shell that
    # ran setup.sh has to be told below, or the very next command it is
    # given, serve.sh, ends in "uv: command not found".
    export PATH="$HOME/.local/bin:$PATH"
    installed_uv=1
fi

mkdir -p "$AURADREAM_OUTPUTS_DIR" "$(dirname "$AURADREAM_DB_PATH")"
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

# The prompt critic's pinned llama.cpp build (backend/critic.py) takes glibc
# 2.38 or newer and three libraries from the system rather than its own tree;
# a fresh WSL Ubuntu has no libgomp1. The app checks the same before the
# critic's first 5 GB download, and refuses there without them.
glibc=$(getconf GNU_LIBC_VERSION 2>/dev/null || true)
glibc=${glibc#glibc }
if [ -n "$glibc" ] && ! printf '%s\n' 2.38 "$glibc" | sort -V -C; then
    echo "note: glibc $glibc is older than the 2.38 the prompt critic's llama.cpp build" >&2
    echo "      needs (Ubuntu 24.04 or newer), so the page will not offer the critic." >&2
else
    libraries=$("$(command -v ldconfig || echo /sbin/ldconfig)" -p 2>/dev/null || true)
    critic_packages=""
    for need in libgomp.so.1:libgomp1 libssl.so.3:libssl3 libcrypto.so.3:libssl3; do
        if [ -n "$libraries" ] && ! grep -q "[[:space:]]${need%%:*} (.*x86-64" <<<"$libraries"; then
            case " $critic_packages " in
                *" ${need#*:} "*) ;;
                *) critic_packages="$critic_packages ${need#*:}" ;;
            esac
        fi
    done
    if [ -n "$critic_packages" ]; then
        echo "note: the prompt critic's llama.cpp needs$critic_packages, not installed here;" >&2
        echo "      'sudo apt install$critic_packages' before its first use." >&2
    fi
fi

if [ "$installed_uv" = 1 ]; then
    cat <<MSG

uv was just installed. This shell cannot see it yet; run this first:

    . "\$HOME/.local/bin/env"
MSG
fi

# An absolute path: .bashrc runs at shell start, where a WSL shell begins in
# \$HOME and a relative `source wsl/env.sh` fails - leaving UV_PROJECT_ENVIRONMENT
# unset, so the next `uv run` from the checkout finds the Windows .venv, judges
# it broken and replaces it.
cat <<MSG

Ready. Every new WSL shell needs the environment first:

    source $PWD/wsl/env.sh     # or put that line in ~/.bashrc
    bash wsl/serve.sh          # then open http://127.0.0.1:7860 in a Windows browser

The gated official repos (krea/Krea-2-Turbo, FLUX.1-Krea-dev) also need
'uv run hf auth login' in here; the lite-infer re-uploads do not.
MSG
