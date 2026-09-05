#!/usr/bin/env bash
# Start the server from WSL 2 with the Linux environment and data paths set.
# Arguments pass through: bare, the server shows its start-up menu with env.sh's
# choices marked as defaults; `--defaults` (or any --model/--quant/--offload)
# starts without asking.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=env.sh
source wsl/env.sh
exec uv run python -m backend.main "$@"
