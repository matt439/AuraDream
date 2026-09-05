#!/usr/bin/env bash
# Start the server from WSL 2 with the Linux environment and data paths set.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=env.sh
source wsl/env.sh
exec uv run python -m backend.main "$@"
