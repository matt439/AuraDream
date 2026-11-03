# Shared by linux/setup.sh, linux/install-service.sh and linux/update.sh;
# sourced, never run on its own. Plain bash, ASCII only, no sudo.
#
# Test seams (tests/test_linux_scripts.py), as in install.ps1:
#   AURADREAM_INSTALL_TEST_ROOT        the project folder, instead of this
#                                      script's own checkout
#   AURADREAM_INSTALL_TEST_PROC_VERSION a file read instead of /proc/version
# Everything else a test changes comes from PATH (stub nvidia-smi, uv, git,
# systemctl, loginctl, hostname, id) and a temporary HOME/XDG_CONFIG_HOME.

# The variables are set for the scripts that source this file.
# shellcheck shell=bash disable=SC2034

if [ -n "${AURADREAM_INSTALL_TEST_ROOT:-}" ]; then
    ROOT=$(cd "$AURADREAM_INSTALL_TEST_ROOT" && pwd)
else
    ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fi
CONFIG_HOME=${XDG_CONFIG_HOME:-$HOME/.config}
UNIT_NAME=auradream.service
UNIT_FILE=$CONFIG_HOME/systemd/user/$UNIT_NAME
ENV_FILE=$CONFIG_HOME/auradream/auradream.env
VENV=$ROOT/.venv

say() { printf '%s\n' "$*"; }
ok() { printf '  [ok]   %s\n' "$*"; }
note() { printf '  [note] %s\n' "$*"; }
warn() { printf '  [warn] %s\n' "$*"; }
would() { printf '  [dry run] would %s\n' "$*"; }
# $1 without leading or trailing blanks.
trim() {
    local text=$1
    text=${text#"${text%%[![:space:]]*}"}
    text=${text%"${text##*[![:space:]]}"}
    printf '%s' "$text"
}
what_to_do() { [ -z "${1:-}" ] || printf '         What to do: %s\n' "$1"; }

# The user's numeric id, from `id` (a test puts a stub first on PATH).
is_root() { [ "$(id -u)" = 0 ]; }

# uv: on PATH, or where its own installer puts it (a non-login shell, such
# as one from ssh or systemd, may not have ~/.local/bin on PATH yet).
find_uv() {
    local found
    if found=$(command -v uv 2>/dev/null) && [ -n "$found" ]; then
        :
    elif [ -x "$HOME/.local/bin/uv" ]; then
        found=$HOME/.local/bin/uv
    else
        return 1
    fi
    # Absolute, symlinks kept: a unit's ExecStart needs a full path.
    printf '%s/%s\n' "$(cd "$(dirname "$found")" && pwd)" "$(basename "$found")"
}

# The port the server uses: AURADREAM_PORT when it is a number, else 7860.
server_port() {
    case "${AURADREAM_PORT:-}" in
        '' | *[!0-9]*) printf '7860\n' ;;
        *) printf '%s\n' "$AURADREAM_PORT" ;;
    esac
}

# Whether AuraDream answers on loopback at $1: prints free, auradream or
# other, as update.ps1's test and the server's own second-start check
# (backend/server_probe.py) decide it - something listens, and its page has
# AuraDream's title. Through the checkout's .venv python with server_probe
# (standard library only, so it works before anything is loaded; -B writes
# no bytecode into the checkout), else curl. Never through a proxy, which
# would answer for loopback instead.
auradream_state() {
    local port=$1 answer page code
    if [ -x "$VENV/bin/python" ] && [ -f "$ROOT/backend/server_probe.py" ]; then
        if answer=$("$VENV/bin/python" -I -B - "$ROOT" "$port" 2>/dev/null <<'PY'
import socket
import sys

sys.path.insert(0, sys.argv[1])
from backend.server_probe import auradream_at

port = int(sys.argv[2])
try:
    socket.create_connection(("127.0.0.1", port), timeout=1).close()
except OSError:
    print("free")
else:
    print("auradream" if auradream_at(f"http://127.0.0.1:{port}/") else "other")
PY
        ); then
            case "$answer" in
                free | auradream | other) printf '%s\n' "$answer"; return 0 ;;
            esac
        fi
    fi
    if command -v curl >/dev/null 2>&1; then
        code=0
        page=$(curl --noproxy '*' --silent --max-time 2 "http://127.0.0.1:$port/" 2>/dev/null) || code=$?
        if [ "$code" = 7 ]; then
            printf 'free\n'          # connection refused: nothing listens
        elif [[ "$page" == *"<title>AuraDream</title>"* ]]; then
            printf 'auradream\n'
        else
            printf 'other\n'
        fi
        return 0
    fi
    printf 'unknown\n'
}

# Whether the user service is running (no systemctl, or no user manager to
# ask, counts as not running).
service_active() {
    command -v systemctl >/dev/null 2>&1 || return 1
    systemctl --user is-active --quiet "$UNIT_NAME" 2>/dev/null
}

# Programs started from this checkout's .venv, one "name (process N)" per
# line. .venv/bin/python is a symlink to uv's own Python, so the executable
# itself is never under .venv: a process counts when the path it was started
# by (its first argument) is, or when it is a python whose environment names
# this .venv as VIRTUAL_ENV (as uv run leaves it). This shell is skipped.
venv_processes() {
    local dir pid argv0 comm environ
    for dir in /proc/[0-9]*; do
        pid=${dir#/proc/}
        [ "$pid" = "$$" ] && continue
        [ "$pid" = "${BASHPID:-$$}" ] && continue
        # Only this user's programs: another user's (a root daemon's) environ
        # cannot be read. Each read's errors are silenced around it, since a
        # redirection that fails (a process that ended mid-scan) reports
        # before a 2>/dev/null after it applies.
        [ -O "$dir" ] || continue
        argv0=
        { IFS= read -r -d '' argv0 <"$dir/cmdline"; } 2>/dev/null || true
        [ -n "$argv0" ] || continue
        comm=
        { IFS= read -r comm <"$dir/comm"; } 2>/dev/null || true
        if [[ "$argv0" == "$VENV/"* ]]; then
            printf '%s (process %s)\n' "${comm:-${argv0##*/}}" "$pid"
        elif [[ "$comm" == python* ]]; then
            # Read whole, then searched: grep -q stopping early under
            # pipefail would fail the pipe it reads from.
            { environ=$(tr '\0' '\n' <"$dir/environ"); } 2>/dev/null || environ=
            if grep -qxF "VIRTUAL_ENV=$VENV" <<<"$environ"; then
                printf '%s (process %s)\n' "$comm" "$pid"
            fi
        fi
    done
}

# The service's environment file, read into this shell when it exists, so a
# script sees the AURADREAM_DB_PATH and AURADREAM_PORT the service runs with.
# It holds NAME=value lines only (systemd reads the same file).
load_env_file() {
    if [ -f "$ENV_FILE" ]; then
        set -a
        # shellcheck disable=SC1090
        . "$ENV_FILE"
        set +a
        return 0
    fi
    return 1
}
