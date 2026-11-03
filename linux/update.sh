#!/usr/bin/env bash
# Update AuraDream to the latest version on native Linux. Safe to run again.
# From the checkout:
#
#     bash linux/update.sh             # while AuraDream is stopped
#     bash linux/update.sh --restart   # stop the service, update, start it again
#     bash linux/update.sh --dry-run   # print what it would do; change nothing
#
# The same steps as update.ps1: it refuses while AuraDream is running, copies
# the gallery database to its before-update name (the way back, since a new
# version may add to the database on its first start), then runs
# `git pull --ff-only` and `uv sync`. It never discards a change: a pull Git
# refuses is explained, and nothing is reset, stashed or checked out.
# --restart stops the service only when it is idle (no job, load or gallery
# bundle operation), and starts it again afterwards, even if the update fails.
set -euo pipefail

# shellcheck source=linux/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

dry_run=0
restart=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) dry_run=1 ;;
        --restart) restart=1 ;;
        -h | --help)
            sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            say "Unknown option: $arg (try --help)" >&2
            exit 2
            ;;
    esac
done

stopped_service=0
pulled=0
# With --restart, a service this script stopped is started again on every
# way out, a failed update included. The start comes before any message, and
# nothing after it is fatal: the script's output may be gone by then (an ssh
# session that dropped, or a pipe through tee closed by Ctrl-C), and a failed
# write must not leave the service stopped.
# shellcheck disable=SC2329 # run by the trap below
start_again() {
    local code=$? started=0
    set +e
    trap '' HUP PIPE INT TERM
    if [ "$stopped_service" = 1 ]; then
        stopped_service=0
        systemctl --user start "$UNIT_NAME" && started=1
        {
            say ''
            if [ "$code" != 0 ] && [ "$pulled" = 1 ]; then
                say '  The code is updated but uv sync did not finish; starting the service on it anyway. If it fails to start, run bash linux/update.sh again.'
            elif [ "$code" != 0 ]; then
                say '  The update did not finish; starting the AuraDream service again as it was.'
            fi
            if [ "$started" = 1 ]; then
                ok 'The AuraDream service is started again.'
            else
                warn 'systemctl --user start auradream failed. Read its log: journalctl --user -u auradream -e'
            fi
        } 2>/dev/null
        [ "$started" = 1 ] || code=1
    fi
    exit "$code"
}
trap start_again EXIT

stop_update() {
    say "  [stop] $1"
    what_to_do "${2:-}"
    say ''
    if [ "$stopped_service" = 1 ]; then
        say 'Update stopped.'
    else
        say 'Update stopped. Nothing was changed by this step.'
    fi
    exit 1
}

say 'AuraDream update'
say "Project folder: $ROOT"
[ "$dry_run" = 0 ] || say 'Dry run: nothing will be changed; every action is printed instead.'
if is_root; then
    stop_update 'This is running as root (or under sudo), which would leave files in the checkout only root can change.' 'Run it as your own user: bash linux/update.sh'
fi
# The service's settings, so the database and port here are the ones it uses.
if load_env_file; then note "Read the service's settings from $ENV_FILE."; fi
cd "$ROOT"
port=$(server_port)

# The running server's own answer to whether it is busy: a job queued or
# running, a model loading or switching, something holding the card, or a
# gallery bundle operation. Prints the reason, or nothing when idle; fails
# when the server cannot be asked.
server_busy() {
    local python
    if [ -x "$VENV/bin/python" ]; then python=$VENV/bin/python
    elif command -v python3 >/dev/null 2>&1; then python=python3
    else return 2
    fi
    "$python" -I -B - "$1" <<'PY'
import json
import sys
import urllib.request

base = f"http://127.0.0.1:{int(sys.argv[1])}"
direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def get(path):
    with direct.open(base + path, timeout=10) as response:
        return json.load(response)


try:
    status = get("/api/status")
    bundles = get("/api/gallery/bundles/status")
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

# 1. Not while AuraDream runs -----------------------------------------------------
if service_active; then
    if [ "$restart" = 0 ]; then
        stop_update 'The AuraDream service is running. Updating under it could break the gallery database copy and the packages it is using.' 'Stop it with: systemctl --user stop auradream, run bash linux/update.sh again, then: systemctl --user start auradream (or run bash linux/update.sh --restart, which does all three when it is idle).'
    fi
    busy_code=0
    busy=$(server_busy "$port") || busy_code=$?
    if [ "$busy_code" != 0 ]; then
        stop_update "The AuraDream service is running, and whether it is busy could not be asked at http://127.0.0.1:$port/: ${busy:-no python to ask with}." 'Wait for it to finish its work, stop it with: systemctl --user stop auradream, then run bash linux/update.sh again.'
    elif [ -n "$busy" ]; then
        stop_update "The AuraDream service is busy: $busy." 'Let it finish (or cancel it in the page), then run bash linux/update.sh --restart again.'
    fi
    if [ "$dry_run" = 1 ]; then
        would 'run: systemctl --user stop auradream (it is idle), and start it again after the update'
    else
        say '  The service is idle; stopping it for the update (systemctl --user stop auradream)...'
        stopped_service=1
        systemctl --user stop "$UNIT_NAME" || stop_update 'systemctl --user stop auradream failed.' 'Read its message above; journalctl --user -u auradream -e shows the log.'
        ok 'The AuraDream service is stopped; it starts again when the update ends.'
    fi
elif [ "$restart" = 1 ]; then
    note 'The AuraDream service is not running, so there is nothing to restart; updating only.'
fi
state=$(auradream_state "$port")
if [ "$state" = auradream ] && ! { [ "$dry_run" = 1 ] && [ "$restart" = 1 ]; }; then
    stop_update "AuraDream is running at http://127.0.0.1:$port/. Updating under it could break the gallery database copy and the packages it is using." 'Stop it first: press Ctrl-C in its terminal (or systemctl --user stop auradream for the service), wait for it to finish, then run bash linux/update.sh again.'
fi
[ "$state" != other ] || note "Another program is using port $port; it is not AuraDream, so carrying on."
[ "$state" != unknown ] || note "Could not check port $port (no .venv python and no curl); the process check below still runs."
running=$(venv_processes)
if [ -n "$running" ]; then
    stop_update "A program from this folder's .venv is still running: $(paste -sd ',' <<<"$running" | sed 's/,/, /g'). It may be AuraDream on another port, or a command such as generate." 'Stop it (Ctrl-C in its terminal), then run bash linux/update.sh again.'
fi
ok 'AuraDream is not running.'

# 2. Git ---------------------------------------------------------------------------
if [ ! -e "$ROOT/.git" ]; then
    stop_update 'This folder has no .git: it came from a ZIP download, which cannot update itself.' 'Clone the project with Git instead (git clone https://github.com/matt439/AuraDream.git), or download the new ZIP (docs/setup.md, Updating).'
fi
command -v git >/dev/null 2>&1 || stop_update 'Git is not installed (or not on PATH in this shell).' 'Install it: sudo apt install git'
git=$(command -v git)
uv=$(find_uv) || stop_update 'uv is not installed (or not on PATH in this shell).' 'Run bash linux/setup.sh, which installs it; then open a new shell.'
dirty_code=0
dirty=$("$git" -C "$ROOT" status --porcelain --untracked-files=no 2>&1) || dirty_code=$?
if [ "$dirty_code" = 0 ] && [ -n "$(trim "$dirty")" ]; then
    note "Some code files in this folder were changed. Git keeps them; the update goes ahead unless the new version changes the same files:"
    while IFS= read -r line; do say "         $line"; done <<<"$dirty"
fi

# 3. The gallery database copy -------------------------------------------------------
# The default database, as backend/legacy_names.py chooses it: auradream.db,
# or local-diffusion.db while that is the only one; AURADREAM_DB_PATH wins
# (relative to the checkout, where the service runs, and ~ expanded as
# backend/config.py expands it).
if [ -n "${AURADREAM_DB_PATH:-}" ]; then
    db=$AURADREAM_DB_PATH
    # shellcheck disable=SC2088 # a literal ~ from the env file, expanded here
    case "$db" in
        "~") db=$HOME ;;
        "~/"*) db=$HOME/${db#"~/"} ;;
    esac
else
    db=$ROOT/auradream.db
    legacy=$ROOT/local-diffusion.db
    if [ ! -e "$db" ] && [ -e "$legacy" ]; then db=$legacy; fi
fi
if [ -e "$db" ]; then
    name=${db##*/}
    copy=$(dirname "$db")/${name%.*}-before-update.db
    wal=$db-wal
    has_wal=0
    [ ! -s "$wal" ] || has_wal=1
    if [ "$dry_run" = 1 ]; then
        would "copy $db to $copy (replacing an older copy)"
        [ "$has_wal" = 0 ] || would "copy $wal to $copy-wal"
    else
        cp -f -- "$db" "$copy"
        # A journal left from an older copy would be applied to this one.
        rm -f -- "$copy-wal" "$copy-shm"
        [ "$has_wal" = 0 ] || cp -f -- "$wal" "$copy-wal"
        ok "Gallery database copied to $copy. If the new version misbehaves, that copy is the way back."
    fi
else
    note "No gallery database yet ($db), so there is nothing to copy."
fi

# 4. git pull --ff-only ----------------------------------------------------------------
if [ "$dry_run" = 1 ]; then
    would "run in $ROOT: git pull --ff-only"
else
    say '  Getting the latest version (git pull --ff-only)...'
    pull_code=0
    pull=$("$git" -C "$ROOT" pull --ff-only 2>&1) || pull_code=$?
    if [ -n "$(trim "$pull")" ]; then
        while IFS= read -r line; do say "    $line"; done <<<"$pull"
    fi
    if [ "$pull_code" != 0 ]; then
        if grep -q 'untracked working tree files would be overwritten' <<<"$pull"; then
            stop_update 'Git did not update: files in this folder that are not part of the code would be overwritten by files the new version adds. Nothing was discarded.' 'Git names the files above. Move or rename them (somewhere outside this folder if you want to keep them), then run bash linux/update.sh again.'
        elif grep -q 'would be overwritten' <<<"$pull"; then
            stop_update 'Git did not update: code files changed in this folder would be overwritten by the new version. Nothing was discarded.' 'Git names the files above. To keep your change, copy those files somewhere else first. Then put back the original with: git restore <file> (this discards your change to that file) and run bash linux/update.sh again.'
        elif grep -qE 'Not possible to fast-forward|diverging|non-fast-forward|fatal: Need to specify how to reconcile' <<<"$pull"; then
            stop_update 'Git did not update: this folder has commits of its own that the new version does not include. Nothing was discarded.' 'If you made those commits, merge or rebase them yourself (git pull). If you did not, ask on the project page before changing anything.'
        elif grep -qE 'no tracking information|There is no tracking' <<<"$pull"; then
            stop_update 'Git did not update: this branch does not follow one on GitHub.' 'Switch to main with: git checkout main, then run bash linux/update.sh again.'
        elif grep -qE 'Could not resolve host|unable to access|Failed to connect|timed out' <<<"$pull"; then
            stop_update 'Git could not reach GitHub.' 'Check the internet connection (and any proxy or firewall), then run bash linux/update.sh again.'
        else
            stop_update "git pull --ff-only failed (exit $pull_code). Nothing was discarded." "Read Git's message above; git status shows the state of the folder."
        fi
    fi
    pulled=1
    ok 'The code is up to date.'
fi

# 5. uv sync -----------------------------------------------------------------------------
if [ "$dry_run" = 1 ]; then
    would "run in $ROOT: uv sync"
else
    say '  Installing any changed packages (uv sync); this can take a while when PyTorch changes...'
    sync_code=0
    "$uv" sync || sync_code=$?
    if [ "$sync_code" != 0 ]; then
        say "  [warn] uv sync failed (exit $sync_code). The code is updated; run bash linux/update.sh again."
        say '         A proxy or firewall in the way of the download can break it.'
        exit 1
    fi
    ok 'Packages up to date.'
fi

say ''
if [ "$dry_run" = 1 ]; then say 'Dry run finished. Nothing was changed.'; exit 0; fi
say 'Update finished.'
if [ "$stopped_service" = 0 ]; then
    say 'Start AuraDream again: systemctl --user start auradream (the service), or uv run python -m backend.main.'
fi
exit 0
