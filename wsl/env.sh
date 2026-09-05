# Source this in a WSL 2 shell before any `uv` command for this project:
#
#     source wsl/env.sh
#
# The checkout can stay on the Windows drive (/mnt/c/...). Three things must
# not: the virtualenv, the HuggingFace cache and the gallery. /mnt/c is a 9P
# bridge - a 25 GB weight load over it crawls, and SQLite cannot lock across it.
# uv also has to be told where its environment is: left alone it would find the
# Windows .venv in the project directory, judge it broken, and replace it.

_krea_home="${KREA_WSL_HOME:-$HOME/krea-local}"

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$_krea_home/.venv}"
export KREA_OUTPUTS_DIR="${KREA_OUTPUTS_DIR:-$_krea_home/outputs}"
export KREA_DB_PATH="${KREA_DB_PATH:-$_krea_home/krea.db}"
# HF_HOME is left alone: its default, ~/.cache/huggingface, is already ext4.

# The reason to run here at all: Nunchaku Lite's 4-bit kernels are Linux-only.
# These repos are the official weights re-uploaded pre-quantized, and not gated.
# nvfp4 needs an RTX 50 (Blackwell); older cards use the same names with int4.
export KREA_QUANT="${KREA_QUANT:-nunchaku}"
export KREA_MODEL_ID="${KREA_MODEL_ID:-lite-infer/krea-2-turbo-nunchaku-lite-nvfp4_r32-bnb4-text-encoder}"
# FLUX.1 Krea dev:  lite-infer/flux.1-krea-dev-nunchaku-lite-nvfp4_r32-bnb4-text-encoder

# torch 2.13 swaps some CUDA ops for Triton kernels even in eager mode, and
# Triton compiles a small C module the first time it runs - which needs a C
# compiler a fresh Ubuntu does not have. The Qwen3-VL text encoder hits one of
# those ops (a batched outer product in its rotary embedding), so without this
# the first step of the first image dies with "Failed to find C compiler".
# This makes torch use its ordinary ATen kernels instead; the transformer's
# matmuls run in Nunchaku's own kernels regardless. After `sudo apt install gcc`
# the line can go.
export TORCH_DISABLE_NATIVE_JIT="${TORCH_DISABLE_NATIVE_JIT:-1}"

# The Hub's xet downloader tunes its connection count adaptively, and on this
# connection that tuning collapsed to a single connection after one failed
# request and sat there - a 12 GB checkpoint stalled at 4.7 GB for twenty
# minutes, twice. A fixed count downloaded the same files in twenty minutes.
export HF_XET_CLIENT_ENABLE_ADAPTIVE_CONCURRENCY="${HF_XET_CLIENT_ENABLE_ADAPTIVE_CONCURRENCY:-false}"
export HF_XET_FIXED_DOWNLOAD_CONCURRENCY="${HF_XET_FIXED_DOWNLOAD_CONCURRENCY:-32}"

unset _krea_home
