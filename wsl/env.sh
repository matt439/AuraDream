# Source this in a WSL 2 shell before any `uv` command for this project:
#
#     source wsl/env.sh
#
# The checkout can stay on the Windows drive (/mnt/c/...). Three things must
# not: the virtualenv, the HuggingFace cache and the gallery. /mnt/c is a 9P
# bridge - a 25 GB weight load over it crawls, and SQLite cannot lock across it.
# uv also has to be told where its environment is: left alone it would find the
# Windows .venv in the project directory, judge it broken, and replace it.

_local_diffusion_home="${LOCAL_DIFFUSION_WSL_HOME:-$HOME/local-diffusion}"

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$_local_diffusion_home/.venv}"
export LOCAL_DIFFUSION_OUTPUTS_DIR="${LOCAL_DIFFUSION_OUTPUTS_DIR:-$_local_diffusion_home/outputs}"
export LOCAL_DIFFUSION_DB_PATH="${LOCAL_DIFFUSION_DB_PATH:-$_local_diffusion_home/local-diffusion.db}"
# HF_HOME is left alone: its default, ~/.cache/huggingface, is already ext4.

# The reason to run here at all: Nunchaku Lite's 4-bit kernels are Linux-only.
# These repos are the official weights re-uploaded pre-quantized, and not gated.
# nvfp4 needs an RTX 50 (Blackwell); older cards use the same names with int4.
if [ -z "${LOCAL_DIFFUSION_QUANT:-}" ]; then
  export LOCAL_DIFFUSION_QUANT="nunchaku"
  export LOCAL_DIFFUSION_WSL_DEFAULT_QUANT="$LOCAL_DIFFUSION_QUANT"
elif [ "${LOCAL_DIFFUSION_WSL_DEFAULT_QUANT:-}" != "$LOCAL_DIFFUSION_QUANT" ]; then
  unset LOCAL_DIFFUSION_WSL_DEFAULT_QUANT
fi
# The checkpoint default is tied to the quantization: a pre-quantized repo
# under fp8 dies deep in transformers after the whole download ("quantized
# with BitsAndBytesConfig but you are passing a TorchAoConfig"). Any other
# LOCAL_DIFFUSION_QUANT falls through to config.py's own default checkpoint. The 4-bit
# format defaults to nvfp4 (RTX 50); LOCAL_DIFFUSION_NUNCHAKU_FORMAT=int4 for older cards,
# as setup.sh reports.
if [ "$LOCAL_DIFFUSION_QUANT" = "nunchaku" ]; then
  _local_diffusion_fmt="${LOCAL_DIFFUSION_NUNCHAKU_FORMAT:-nvfp4}"
  if [ -z "${LOCAL_DIFFUSION_MODEL_ID:-}" ]; then
    export LOCAL_DIFFUSION_MODEL_ID="lite-infer/krea-2-turbo-nunchaku-lite-${_local_diffusion_fmt}_r32-bnb4-text-encoder"
    export LOCAL_DIFFUSION_WSL_DEFAULT_MODEL_ID="$LOCAL_DIFFUSION_MODEL_ID"
  elif [ "${LOCAL_DIFFUSION_WSL_DEFAULT_MODEL_ID:-}" != "$LOCAL_DIFFUSION_MODEL_ID" ]; then
    unset LOCAL_DIFFUSION_WSL_DEFAULT_MODEL_ID
  fi
  # FLUX.1 Krea dev:  lite-infer/flux.1-krea-dev-nunchaku-lite-${_local_diffusion_fmt}_r32-bnb4-text-encoder
  # ERNIE-Image Turbo: lite-infer/ERNIE-Image-Turbo-nunchaku-lite-${_local_diffusion_fmt}-bnb4-text-encoder (no _r32)
  unset _local_diffusion_fmt
fi

# A bare server restores its browser choice over these marked defaults. The
# standalone CLI still gets Nunchaku above, and an explicitly exported value
# (including one changed after this file was sourced) keeps taking precedence.

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

unset _local_diffusion_home
