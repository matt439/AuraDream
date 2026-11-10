# Shared by linux/setup.sh, linux/install-service.sh, linux/update.sh and
# linux/control.sh; sourced, never run on its own. Plain bash, ASCII only,
# no sudo.
#
# Test seams (tests/test_linux_scripts.py), as in install.ps1:
#   AURADREAM_INSTALL_TEST_ROOT        the project folder, instead of this
#                                      script's own checkout
#   AURADREAM_INSTALL_TEST_PROC_VERSION a file read instead of /proc/version
#   AURADREAM_INSTALL_TEST_CGROUP_ROOT  the cgroup v2 mount, instead of
#                                      /sys/fs/cgroup
# Everything else a test changes comes from PATH (stub nvidia-smi, uv, git,
# systemctl, journalctl, systemd-inhibit, loginctl, hostname, id) and a
# temporary HOME/XDG_CONFIG_HOME.

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

# A Python to ask the running server with: the checkout's .venv one, else
# the system's python3. Fails when there is neither.
status_python() {
    if [ -x "$VENV/bin/python" ]; then printf '%s\n' "$VENV/bin/python"
    elif command -v python3 >/dev/null 2>&1; then command -v python3
    else return 1
    fi
}

# Python every script asks the running server with, put before each
# script's own lines (ask_server_python): ask(port, path, timeout) answers
# the JSON at that path on loopback, never through a proxy. Where
# AURADREAM_PASSCODE is in the environment (load_env_file exports the
# service's) each request carries it as Authorization: Bearer, read from the
# environment and never an argument, and then follows no redirect, which
# would carry the header elsewhere. The passcode's gate refusing is Refused,
# in words that never hold the passcode: a 401 is a passcode the
# environment file lacks (or one the server does not accept), a 429 the
# lock after too many wrong ones. Neither is a server that is down.
IFS= read -r -d '' ASK_SERVER_PY <<'PY' || true
import json
import os
import sys
import urllib.error
import urllib.request


class Refused(Exception):
    """The passcode's gate said no; its text is what to print."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _passcode():
    # Bytes as they are in the environment, sent as they are.
    raw = os.environb.get(b"AURADREAM_PASSCODE", b"").strip()
    if any((byte < 32 and byte != 9) or byte == 127 for byte in raw):
        # Not Refused: no request is made, so nothing has answered.
        raise ValueError("it was not asked: AURADREAM_PASSCODE holds a control character, "
                         "which a request cannot carry")
    return raw.decode("latin-1")


def ask(port, path, timeout):
    passcode = _passcode()
    handlers = [urllib.request.ProxyHandler({})]
    headers = {}
    if passcode:
        handlers.append(_NoRedirect())
        headers["Authorization"] = "Bearer " + passcode
    request = urllib.request.Request(f"http://127.0.0.1:{int(port)}{path}", headers=headers)
    try:
        with urllib.request.build_opener(*handlers).open(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            wait = str(exc.headers.get("Retry-After") or "")
            wait = f"{wait} s" if wait.isdigit() else "a minute"
            raise Refused(
                "it is refusing every passcode for now, after too many wrong ones just now: "
                f"wait {wait} and run this again"
            ) from None
        if exc.code == 401 and passcode:
            raise Refused(
                "it did not accept the AURADREAM_PASSCODE these scripts read: the service's environment "
                "file must hold the passcode the server was started with (a restart applies a changed one)"
            ) from None
        if exc.code == 401:
            raise Refused(
                "it asks for a passcode, and AURADREAM_PASSCODE is missing from the service's environment "
                "file (or from this shell, where there is no such file)"
            ) from None
        raise
PY

# Run $1 (a python) on ASK_SERVER_PY and then the lines on standard input,
# with the remaining arguments as that program's own.
ask_server_python() {
    local python=$1
    shift
    { printf '%s\n' "$ASK_SERVER_PY"; cat; } | "$python" -I -B - "$@"
}

# The running server's own answer to whether it is busy: a job queued or
# running, a model loading or switching, something holding the card, or a
# gallery bundle operation. Prints the reason, or nothing when idle; fails
# when the server cannot be asked, which a server that asks for a passcode
# these scripts do not have is too: one that would not say is never taken
# for an idle one. Both reads are the page's own polls, which the keep-awake
# hold does not count as someone using it.
server_busy() {
    local python
    python=$(status_python) || return 2
    ask_server_python "$python" "$1" <<'PY'
port = sys.argv[1]
try:
    status = ask(port, "/api/status", 10)
    bundles = ask(port, "/api/gallery/bundles/status", 10)
except Exception as exc:  # noqa: BLE001 - any failure means "cannot tell"
    print(f"could not ask it ({exc})")
    sys.exit(1)
if status.get("active_jobs"):
    print(f"{len(status['active_jobs'])} job(s) queued or running")
elif status.get("state") == "loading":
    print("a model is loading (or downloading)")
elif status.get("switching"):
    print("a model switch is under way")
elif status.get("card_holder"):
    print(f"the card is in use ({status['card_holder']})")
elif bundles.get("state") == "running":
    print(f"a gallery bundle operation is running ({bundles.get('operation') or 'unnamed'})")
PY
}

# Programs started from this checkout's .venv, one "name (process N)" per
# line. .venv/bin/python is a symlink to uv's own Python, so the executable
# itself is never under .venv: a process counts when the path it was started
# by (its first argument) is, or when it is a python whose environment names
# this .venv as VIRTUAL_ENV (as uv run leaves it). This shell is skipped.
# The processes in the service's own control group, one per line (nothing
# when it has none or it cannot be read): what stopping it would end.
service_pids() {
    local cgroup root=${AURADREAM_INSTALL_TEST_CGROUP_ROOT:-/sys/fs/cgroup}
    cgroup=$(systemctl --user show -p ControlGroup --value "$UNIT_NAME" 2>/dev/null) || return 0
    [ -n "$cgroup" ] && [ -r "$root$cgroup/cgroup.procs" ] || return 0
    cat "$root$cgroup/cgroup.procs" 2>/dev/null || true
}

# VENV_SKIP_PIDS, one per line, are left out (a dry run of --restart: the
# service's own python, which the real run would stop first).
venv_processes() {
    local dir pid argv0 comm environ skip=$'
'"${VENV_SKIP_PIDS:-}"$'
'
    for dir in /proc/[0-9]*; do
        pid=${dir#/proc/}
        [ "$pid" = "$$" ] && continue
        [ "$pid" = "${BASHPID:-$$}" ] && continue
        [[ "$skip" == *$'
'"$pid"$'
'* ]] && continue
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

# Whether the environment file's AURADREAM_PASSCODE is one both of the
# file's readers take alike, read as text before anything sources it.
# systemd takes a line to its end; bash, sourcing it, would set
# "AURADREAM_PASSCODE=correct horse" to "correct" and run "horse", and
# reads $, quotes, #, ; and a backslash each its own way. So only letters,
# digits and . _ - + @ % : , / pass (and no value at all). The limit is this
# file's: a passcode set in a shell may hold any character. Never prints
# the value.
env_file_passcode_plain() {
    # Spaces and tabs only: a carriage return (a file saved with Windows
    # line ends) stays in the value, and is refused.
    local LC_ALL=C line value blank=$' \t'
    [ -f "$ENV_FILE" ] || return 0
    while IFS= read -r line || [ -n "$line" ]; do
        line=${line#"${line%%[!$blank]*}"}
        case "$line" in
            AURADREAM_PASSCODE=*) ;;
            *) continue ;;
        esac
        value=${line#AURADREAM_PASSCODE=}
        value=${value%"${value##*[!$blank]}"}
        case "$value" in *[!A-Za-z0-9._+@%:,/-]*) return 1 ;; esac
    done <"$ENV_FILE"
    return 0
}

# Stop, in plain words, on an environment file whose passcode the scripts
# cannot read as systemd does. Before every read of the file.
require_plain_env_passcode() {
    env_file_passcode_plain && return 0
    say "  [stop] The AURADREAM_PASSCODE line in $ENV_FILE has a character these scripts cannot read the way systemd does (a space, a quote, a dollar sign, #, ;, a backslash or another symbol), so the file was not read and nothing was changed."
    what_to_do 'In that file, use a passcode made only of letters, digits and . _ - + @ % : , / (no quotes round it), then apply it with: systemctl --user restart auradream'
    exit 1
}

# The service's environment file, read into this shell when it exists, so a
# script sees the AURADREAM_DB_PATH and AURADREAM_PORT the service runs with
# (and its AURADREAM_PASSCODE, which the scripts' requests then carry).
# It holds NAME=value lines only (systemd reads the same file).
load_env_file() {
    if [ -f "$ENV_FILE" ]; then
        require_plain_env_passcode
        set -a
        # shellcheck disable=SC1090
        . "$ENV_FILE"
        set +a
        return 0
    fi
    return 1
}
