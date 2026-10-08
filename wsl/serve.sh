#!/usr/bin/env bash
# Start the server from WSL 2 with the Linux environment and data paths set.
# Arguments pass through: bare, the server restores browser choices without
# asking. --defaults keeps env.sh's defaults; --menu opens the optional picker.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=env.sh
source wsl/env.sh
exec uv run python -m backend.main "$@"
