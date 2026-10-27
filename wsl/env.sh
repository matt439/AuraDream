# Source this in a WSL 2 shell before any `uv` command for this project:
#
#     source wsl/env.sh
#
# The checkout can stay on the Windows drive (/mnt/c/...). Three things must
# not: the virtualenv, the HuggingFace cache and the gallery. /mnt/c is a 9P
# bridge - a 25 GB weight load over it crawls, and SQLite cannot lock across it.
# uv also has to be told where its environment is: left alone it would find the
# Windows .venv in the project directory, judge it broken, and replace it.

# Before the rename to AuraDream the home was ~/local-diffusion and its gallery
# local-diffusion.db. One already there is used where it is, never moved: a
# virtualenv does not survive a move, and the gallery's files sit beside it.
if [ -n "${AURADREAM_WSL_HOME:-}" ]; then
  _auradream_home="$AURADREAM_WSL_HOME"
elif [ ! -e "$HOME/auradream" ] && [ -d "$HOME/local-diffusion" ]; then
  _auradream_home="$HOME/local-diffusion"
else
  _auradream_home="$HOME/auradream"
fi
_auradream_db="$_auradream_home/auradream.db"
if [ ! -e "$_auradream_db" ] && [ -e "$_auradream_home/local-diffusion.db" ]; then
  _auradream_db="$_auradream_home/local-diffusion.db"
fi

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$_auradream_home/.venv}"
export AURADREAM_OUTPUTS_DIR="${AURADREAM_OUTPUTS_DIR:-$_auradream_home/outputs}"
export AURADREAM_DB_PATH="${AURADREAM_DB_PATH:-$_auradream_db}"
# HF_HOME is left alone: its default, ~/.cache/huggingface, is already ext4.

# The reason to run here at all: Nunchaku Lite's 4-bit kernels are Linux-only.
# These repos are the official weights re-uploaded pre-quantized, and not gated.
# nvfp4 needs an RTX 50 (Blackwell); older cards use the same names with int4.
if [ -z "${AURADREAM_QUANT:-}" ]; then
  export AURADREAM_QUANT="nunchaku"
  export AURADREAM_WSL_DEFAULT_QUANT="$AURADREAM_QUANT"
elif [ "${AURADREAM_WSL_DEFAULT_QUANT:-}" != "$AURADREAM_QUANT" ]; then
  unset AURADREAM_WSL_DEFAULT_QUANT
fi
# The checkpoint default is tied to the quantization: a pre-quantized repo
# under fp8 dies deep in transformers after the whole download ("quantized
# with BitsAndBytesConfig but you are passing a TorchAoConfig"). Any other
# AURADREAM_QUANT falls through to config.py's own default checkpoint. The 4-bit
# format defaults to nvfp4 (RTX 50); AURADREAM_NUNCHAKU_FORMAT=int4 for older cards,
# as setup.sh reports.
if [ "$AURADREAM_QUANT" = "nunchaku" ]; then
  _auradream_fmt="${AURADREAM_NUNCHAKU_FORMAT:-nvfp4}"
  if [ -z "${AURADREAM_MODEL_ID:-}" ]; then
    export AURADREAM_MODEL_ID="lite-infer/krea-2-turbo-nunchaku-lite-${_auradream_fmt}_r32-bnb4-text-encoder"
    export AURADREAM_WSL_DEFAULT_MODEL_ID="$AURADREAM_MODEL_ID"
  elif [ "${AURADREAM_WSL_DEFAULT_MODEL_ID:-}" != "$AURADREAM_MODEL_ID" ]; then
    unset AURADREAM_WSL_DEFAULT_MODEL_ID
  fi
  # FLUX.1 Krea dev:  lite-infer/flux.1-krea-dev-nunchaku-lite-${_auradream_fmt}_r32-bnb4-text-encoder
  # ERNIE-Image Turbo: lite-infer/ERNIE-Image-Turbo-nunchaku-lite-${_auradream_fmt}-bnb4-text-encoder (no _r32)
  unset _auradream_fmt
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

unset _auradream_home _auradream_db
