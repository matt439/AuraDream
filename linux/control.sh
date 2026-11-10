#!/usr/bin/env bash
# Control the AuraDream user service on native Linux, from the checkout:
#
#     bash linux/control.sh status            # the service, the checkout, the server's own status
#     bash linux/control.sh start             # start it and wait for the page to answer
#     bash linux/control.sh stop [--force]    # stop it, only while it is idle unless --force
#     bash linux/control.sh restart [--force] # stop it (same check), start it, wait for the page
#     bash linux/control.sh logs [-f] [-n N]  # its log (journalctl), the last 100 lines by default
#     bash linux/control.sh doctor            # the read-only machine check, with the service's settings
#
# bin/auradream-server.ps1 runs these over ssh from a Windows PC. status
# exits 0 when the service is active and the server answers, 1 otherwise;
# stop and restart refuse while the server is busy (a job, a load, a switch,
# something holding the card, a gallery bundle operation), as
# update.sh --restart does. It never uses sudo.
set -euo pipefail

# shellcheck source=linux/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

usage() {
    sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

refuse() {
    say "  [stop] $1"
    what_to_do "${2:-}"
    exit 1
}

command=${1:-help}
[ "$#" = 0 ] || shift
case "$command" in
    help | -h | --help) usage; exit 0 ;;
    status | start | stop | restart | logs | doctor) ;;
    *)
        say "Unknown command: $command" >&2
        usage >&2
        exit 2
        ;;
esac

if is_root; then
    refuse 'This is running as root (or under sudo); the AuraDream service is a user service of your own.' 'Run it as your own user: bash linux/control.sh '"$command"
fi
# The service's settings, so the port asked is the one it uses.
load_env_file || true
port=$(server_port)
scheme=$(server_scheme)
# How long start and restart wait for the page (a test shortens it).
wait_seconds=${AURADREAM_INSTALL_TEST_WAIT:-60}

bad_option() {
    say "Unknown option for $command: $1 (try: bash linux/control.sh help)" >&2
    exit 2
}

# Whether the server answers /api/status on loopback (one quick try). A
# server that answers by asking for a passcode these scripts do not have
# (or by its lock on wrong ones) is up, not starting: exit 3, with the
# reason printed.
server_answers() {
    local python
    python=$(status_python) || return 1
    ask_server_python "$python" "$port" <<'PY' 2>/dev/null
try:
    sys.exit(0 if isinstance(ask(sys.argv[1], "/api/status", 2), dict) else 1)
except Refused as exc:
    print(exc)
    sys.exit(3)
except Exception:  # noqa: BLE001 - any failure means "not yet"
    sys.exit(1)
PY
}

# The unit's SubState and NRestarts, as "<sub> <count>" ("-" when unknown).
unit_restart_state() {
    local key value sub='' restarts=''
    while IFS='=' read -r key value; do
        case "$key" in
            SubState) sub=$value ;;
            NRestarts) restarts=$value ;;
        esac
    done < <(systemctl --user show -p SubState -p NRestarts "$UNIT_NAME" 2>/dev/null || true)
    printf '%s %s\n' "${sub:--}" "${restarts:--}"
}

# After a start: wait for the page, or for the service to fail. The unit has
# Restart=on-failure, so a crash while starting shows as auto-restart (or a
# grown NRestarts) rather than failed, until systemd's start limit trips.
wait_for_page() {
    local waited=0 sub restarts first_restarts said answers
    read -r sub first_restarts < <(unit_restart_state)
    say "  Waiting for the page at $scheme://127.0.0.1:$port/ (up to $wait_seconds s)..."
    while :; do
        answers=0
        said=$(server_answers) || answers=$?
        if [ "$answers" = 0 ]; then
            ok "AuraDream answers at $scheme://127.0.0.1:$port/ (after about $waited s)."
            return 0
        elif [ "$answers" = 3 ]; then
            ok "AuraDream answers at $scheme://127.0.0.1:$port/ (after about $waited s)."
            warn "These scripts cannot ask it anything more: $said."
            return 0
        fi
        if systemctl --user is-failed --quiet "$UNIT_NAME" 2>/dev/null; then
            refuse 'The AuraDream service failed while starting.' 'Read its log: bash linux/control.sh logs (or journalctl --user -u auradream -e).'
        fi
        read -r sub restarts < <(unit_restart_state)
        if [ "$sub" = auto-restart ] || { [[ $restarts =~ ^[0-9]+$ ]] && [[ $first_restarts =~ ^[0-9]+$ ]] && [ "$restarts" -gt "$first_restarts" ]; }; then
            refuse 'The AuraDream service exited while starting, and systemd is starting it again (Restart=on-failure): it is crashing, not slow.' 'Read its log: bash linux/control.sh logs (or journalctl --user -u auradream -e).'
        fi
        if [ "$waited" -ge "$wait_seconds" ]; then
            refuse "The service is started, but the page did not answer within $wait_seconds s." 'It may still be starting; run bash linux/control.sh status in a minute, and read its log: bash linux/control.sh logs.'
        fi
        sleep 1
        waited=$((waited + 1))
    done
}

need_systemctl() {
    command -v systemctl >/dev/null 2>&1 || refuse 'systemctl is not here: this machine has no systemd to run the service.' 'Run AuraDream by hand instead: uv run python -m backend.main'
}

# The busy check stop and restart share. Without --force a busy server, or
# one that cannot be asked, is refused; with it, says what it interrupts.
check_idle() {
    local force=$1 verb=$2 doing=$3 busy busy_code=0
    busy=$(server_busy "$port") || busy_code=$?
    if [ "$force" = 1 ]; then
        if [ "$busy_code" != 0 ]; then
            warn "--force: whether it is busy could not be asked (${busy:-no python to ask with}); $doing it anyway."
        elif [ -n "$busy" ]; then
            warn "--force: $doing it although it is busy: $busy. That work is interrupted."
        else
            note "--force: it is idle, so nothing is interrupted."
        fi
        return 0
    fi
    if [ "$busy_code" != 0 ]; then
        refuse "Whether the AuraDream service is busy could not be asked at $scheme://127.0.0.1:$port/: ${busy:-no python to ask with}." "Wait for it to answer (bash linux/control.sh status), or $verb it anyway with --force, which interrupts any work it has."
    elif [ -n "$busy" ]; then
        refuse "The AuraDream service is busy: $busy." "Let it finish (or cancel it in the page), then run bash linux/control.sh $verb again; --force ${verb}s it anyway and interrupts that work."
    fi
    ok 'It is idle.'
}

# --- status ------------------------------------------------------------------------
do_status() {
    [ "$#" = 0 ] || bad_option "$1"
    local healthy=1 key value active='' sub='' since='' enabled='' loaded='' linger user checkout=''
    say 'AuraDream server status'
    say "Project folder: $ROOT"

    say "== The service ($UNIT_NAME)"
    if ! command -v systemctl >/dev/null 2>&1; then
        warn 'systemctl is not here: this machine has no systemd to run the service.'
        healthy=0
    else
        while IFS='=' read -r key value; do
            case "$key" in
                LoadState) loaded=$value ;;
                ActiveState) active=$value ;;
                SubState) sub=$value ;;
                StateChangeTimestamp) since=$value ;;
                UnitFileState) enabled=$value ;;
            esac
        done < <(systemctl --user show "$UNIT_NAME" -p LoadState -p ActiveState -p SubState \
            -p StateChangeTimestamp -p UnitFileState 2>/dev/null || true)
        if [ -z "$active" ]; then
            warn 'systemctl --user could not be asked (no user manager in this session?).'
            healthy=0
        elif [ "$loaded" = not-found ]; then
            warn 'No auradream.service is installed for this user.'
            what_to_do 'bash linux/install-service.sh'
            healthy=0
        else
            [ "$active" = active ] || healthy=0
            local state=$active
            [ -z "$sub" ] || state="$active ($sub)"
            [ -z "$since" ] || state="$state since $since"
            if [ "$active" = active ]; then ok "$state"
            elif [ "$active" = failed ]; then warn "$state. Its log: bash linux/control.sh logs"
            else note "$state"
            fi
            case "$enabled" in
                enabled) ok 'Enabled: it starts with your user manager.' ;;
                '') note 'Whether it is enabled could not be read.' ;;
                *) note "Not enabled ($enabled): it does not start by itself. systemctl --user enable auradream" ;;
            esac
        fi
        user=${USER:-$(id -un)}
        linger=$(loginctl show-user "$user" -p Linger 2>/dev/null || true)
        if [ "$linger" = Linger=yes ]; then
            ok 'Lingering is on: it runs from boot, with nobody logged in.'
        elif [ -n "$linger" ]; then
            note "Lingering is off: it runs only while you are logged in (sudo loginctl enable-linger $user)."
        else
            note 'Whether lingering is on could not be asked (loginctl).'
        fi
    fi

    say '== The checkout'
    if [ ! -e "$ROOT/.git" ]; then
        note 'Not a Git checkout (no .git), so it cannot be updated with Git.'
    elif ! command -v git >/dev/null 2>&1; then
        note 'Git is not installed (or not on PATH in this shell).'
    else
        local branch commit upstream fetch_error='' counts ahead behind
        branch=$(git -C "$ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || printf '?')
        commit=$(git -C "$ROOT" log -1 --format='%h %s (%cr)' 2>/dev/null || printf '?')
        say "         Branch $branch at $commit"
        # The short commit alone, for the running server's own below.
        [ "$commit" = '?' ] || checkout=${commit%% *}
        if upstream=$(git -C "$ROOT" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null); then
            local fetch_code=0
            fetch_error=$(GIT_TERMINAL_PROMPT=0 git -C "$ROOT" fetch --quiet 2>&1) || fetch_code=$?
            if counts=$(git -C "$ROOT" rev-list --left-right --count "HEAD...@{u}" 2>/dev/null); then
                read -r ahead behind <<<"$counts"
                if [ "$ahead" = 0 ] && [ "$behind" = 0 ]; then
                    ok "Up to date with $upstream."
                elif [ "$ahead" = 0 ]; then
                    note "$behind commit(s) behind $upstream: auradream-server update (or bash linux/update.sh --restart) brings them."
                elif [ "$behind" = 0 ]; then
                    note "$ahead commit(s) ahead of $upstream (commits of this checkout's own)."
                else
                    note "$ahead commit(s) ahead of and $behind behind $upstream: it has commits of its own, so update.sh cannot fast-forward it."
                fi
            fi
            if [ "$fetch_code" != 0 ]; then
                note "git fetch failed, so that is as of the last fetch: $(trim "${fetch_error:-exit $fetch_code}" | head -n 1)"
            fi
        else
            note 'This branch follows no branch on GitHub, so ahead/behind cannot be told.'
        fi
        local dirty
        dirty=$(git -C "$ROOT" status --porcelain --untracked-files=no 2>/dev/null || true)
        [ -z "$(trim "$dirty")" ] || note 'Some code files in this folder are changed (git status lists them).'
    fi

    say "== The server at $scheme://127.0.0.1:$port/"
    local python summary_code=0
    if python=$(status_python); then
        ask_server_python "$python" "$port" "$checkout" <<'PY' || summary_code=$?
port = int(sys.argv[1])
checkout = sys.argv[2] if len(sys.argv) > 2 else ""
try:
    status = ask(port, "/api/status", 10)
except Refused as exc:
    # It is running; only its status is closed to these scripts.
    print("  [ok]   It answers.")
    print(f"  [warn] Its status could not be read: {exc}.")
    sys.exit(3)
except Exception as exc:  # noqa: BLE001 - any failure means "not answering"
    print(f"  [warn] It does not answer ({exc}).")
    sys.exit(1)


def line(label, text):
    print(f"         {label + ':':<13} {text}")


print("  [ok]   It answers.")
# The commit the process started from: the checkout's own, above, moves with
# a pull, and this one only with a restart.
commit = status.get("commit")
if isinstance(commit, str) and commit:
    if checkout and not (commit.startswith(checkout) or checkout.startswith(commit)):
        line("Commit", f"{commit}, and the checkout is at {checkout}: restart it to run the checkout's code")
    else:
        line("Commit", commit)
label = status.get("model_label") or status.get("model_id") or "?"
model_id = status.get("model_id")
if model_id and model_id != label:
    label = f"{label} ({model_id})"
line("Checkpoint", label)
offload = status.get("offload")
line("Loading", f"{status.get('quantization') or '?'}, "
     + (f"{offload} offload" if offload and offload != "none" else "no offload"))
state = status.get("state") or "?"
if status.get("detail"):
    state = f"{state}: {status['detail']}"
line("Model", state)
if status.get("switching"):
    line("Switching", "a model switch is under way")
jobs = status.get("active_jobs") or []
line("Jobs", f"{len(jobs)} queued or running" if jobs else "none")
line("Card", f"in use ({status['card_holder']})" if status.get("card_holder") else "free")
awake = status.get("keep_awake")
if isinstance(awake, dict):
    # "wanted" False is idle, not switched off: only kind "off" is
    # AURADREAM_KEEP_AWAKE=false.
    kind = awake.get("kind") or "none"
    if kind == "off":
        text = "off (AURADREAM_KEEP_AWAKE=false)"
    elif not awake.get("wanted"):
        text = f"{kind}, not holding: idle, so the PC may sleep"
    else:
        text = f"{kind}, " + ("holding" if awake.get("holding") else "not holding")
        if awake.get("reasons"):
            text += ": " + "; ".join(str(reason) for reason in awake["reasons"])
        if awake.get("detail"):
            text += f"; {awake['detail']}"
    line("Keep awake", text)
used, total = status.get("vram_used_mb"), status.get("vram_total_mb")
if total:
    device = f" ({status['device_name']})" if status.get("device_name") else ""
    line("GPU memory", f"{used if used is not None else '?'} of {total} MiB in use{device}")
if status.get("disk_free_mb") is not None:
    line("Disk free", f"{status['disk_free_mb'] / 1024:.1f} GiB")
PY
    else
        warn 'No python to ask it with (no .venv here and no python3).'
        summary_code=1
    fi
    # 3: it answers, by asking for a passcode these scripts do not have (or
    # by its lock on wrong ones). Running, so not unhealthy.
    [ "$summary_code" = 0 ] || [ "$summary_code" = 3 ] || healthy=0

    say '== Sleep holds (systemd-inhibit --list)'
    local inhibitors held=0 line
    if inhibitors=$(systemd-inhibit --list --no-pager 2>/dev/null); then
        while IFS= read -r line; do
            if [[ "$line" == *AuraDream* ]]; then say "         $line"; held=1; fi
        done <<<"$inhibitors"
        [ "$held" = 1 ] || say '         None from AuraDream.'
    else
        note 'systemd-inhibit --list could not be read.'
    fi

    say ''
    if [ "$healthy" = 1 ]; then
        say 'The service is running and AuraDream answers.'
        exit 0
    fi
    say 'The service is not running, or AuraDream does not answer (see above).'
    exit 1
}

# --- start / stop / restart -----------------------------------------------------------
force_option() {
    force=0
    for arg in "$@"; do
        case "$arg" in
            --force) force=1 ;;
            *) bad_option "$arg" ;;
        esac
    done
}

do_start() {
    [ "$#" = 0 ] || bad_option "$1"
    need_systemctl
    if service_active; then
        note 'The AuraDream service is already running.'
    else
        say '  Starting the AuraDream service (systemctl --user start auradream)...'
        systemctl --user start "$UNIT_NAME" || refuse 'systemctl --user start auradream failed (its message is above).' 'Read its log: bash linux/control.sh logs'
        ok 'Started.'
    fi
    wait_for_page
}

do_stop() {
    force_option "$@"
    need_systemctl
    if ! service_active; then
        note 'The AuraDream service is not running; nothing to stop.'
        exit 0
    fi
    check_idle "$force" stop stopping
    say '  Stopping the AuraDream service (systemctl --user stop auradream); it finishes the current step first, which can take up to two minutes...'
    systemctl --user stop "$UNIT_NAME" || refuse 'systemctl --user stop auradream failed (its message is above).' 'Read its log: bash linux/control.sh logs'
    ok 'The AuraDream service is stopped. Start it again with: bash linux/control.sh start'
}

do_restart() {
    force_option "$@"
    need_systemctl
    if service_active; then
        check_idle "$force" restart restarting
        say '  Restarting the AuraDream service (systemctl --user restart auradream)...'
    else
        note 'The AuraDream service is not running; starting it.'
    fi
    systemctl --user restart "$UNIT_NAME" || refuse 'systemctl --user restart auradream failed (its message is above).' 'Read its log: bash linux/control.sh logs'
    ok 'Restarted.'
    wait_for_page
}

# --- logs / doctor ----------------------------------------------------------------------
do_logs() {
    local lines=100 follow=0
    while [ "$#" -gt 0 ]; do
        case "$1" in
            -f | --follow) follow=1 ;;
            -n | --lines)
                [ "$#" -ge 2 ] || bad_option "$1 (it needs a number)"
                lines=$2
                shift
                ;;
            --lines=*) lines=${1#--lines=} ;;
            *) bad_option "$1" ;;
        esac
        shift
    done
    case "$lines" in '' | *[!0-9]*) bad_option "--lines $lines (a whole number)" ;; esac
    command -v journalctl >/dev/null 2>&1 || refuse 'journalctl is not here: this machine has no systemd journal.'
    if [ "$follow" = 1 ]; then
        exec journalctl --user -u "$UNIT_NAME" -n "$lines" -f
    fi
    exec journalctl --user -u "$UNIT_NAME" -n "$lines" --no-pager
}

do_doctor() {
    [ "$#" = 0 ] || bad_option "$1"
    local uv
    uv=$(find_uv) || refuse 'uv is not installed (or not on PATH in this shell).' 'Run bash linux/setup.sh, which installs it.'
    [ ! -f "$ENV_FILE" ] || note "With the service's settings from $ENV_FILE."
    cd "$ROOT"
    exec "$uv" run --no-sync python -m backend.doctor
}

case "$command" in
    status) do_status "$@" ;;
    start) do_start "$@" ;;
    stop) do_stop "$@" ;;
    restart) do_restart "$@" ;;
    logs) do_logs "$@" ;;
    doctor) do_doctor "$@" ;;
esac
