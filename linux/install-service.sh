#!/usr/bin/env bash
# Run AuraDream as a systemd user service other PCs on the network can open.
# Safe to run again. From the checkout, after bash linux/setup.sh:
#
#     bash linux/install-service.sh              # write the unit, enable and start it
#     bash linux/install-service.sh --no-start   # write and enable; start it later
#     bash linux/install-service.sh --dry-run    # print the unit and settings; change nothing
#     bash linux/install-service.sh --uninstall  # stop, disable and remove the unit
#
# It writes ~/.config/systemd/user/auradream.service (under XDG_CONFIG_HOME
# when that is set) and, only when there is none yet, the service's
# environment file ~/.config/auradream/auradream.env, which it never
# rewrites. It never uses sudo: what needs it (lingering) is printed.
set -euo pipefail

# shellcheck source=linux/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

dry_run=0
uninstall=0
start=1
for arg in "$@"; do
    case "$arg" in
        --dry-run) dry_run=1 ;;
        --uninstall) uninstall=1 ;;
        --no-start) start=0 ;;
        -h | --help)
            sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            say "Unknown option: $arg (try --help)" >&2
            exit 2
            ;;
    esac
done

refuse() {
    say "  [stop] $1"
    what_to_do "${2:-}"
    exit 1
}

say 'AuraDream service'
say "Project folder: $ROOT"
[ "$dry_run" = 0 ] || say 'Dry run: nothing will be changed; every action is printed instead.'

if is_root; then
    refuse 'This is running as root (or under sudo). The service is a user service, run by your own user.' 'Run it as that user: bash linux/install-service.sh'
fi
user=${USER:-$(id -un)}

# Runs a systemctl --user command, or in a dry run prints it.
user_systemctl() {
    if [ "$dry_run" = 1 ]; then
        would "run: systemctl --user $*"
        return 0
    fi
    systemctl --user "$@"
}

# --- Uninstall -------------------------------------------------------------------
if [ "$uninstall" = 1 ]; then
    if [ -f "$UNIT_FILE" ]; then
        user_systemctl disable --now "$UNIT_NAME" || warn 'systemctl could not stop or disable it; carrying on.'
        if [ "$dry_run" = 1 ]; then
            would "remove $UNIT_FILE"
        else
            rm -f -- "$UNIT_FILE"
            ok "Removed $UNIT_FILE."
        fi
        user_systemctl daemon-reload || true
    else
        note "No unit at $UNIT_FILE, so there is nothing to remove."
    fi
    if [ -f "$ENV_FILE" ]; then
        note "Kept the environment file $ENV_FILE (your settings); delete it yourself if you no longer want it."
    fi
    note 'The gallery, the downloaded checkpoints and the checkout are untouched.'
    say ''
    if [ "$dry_run" = 1 ]; then say 'Dry run finished. Nothing was changed.'; else say 'The service is removed.'; fi
    exit 0
fi

# --- Checks ----------------------------------------------------------------------
if [ ! -d "$VENV" ]; then
    refuse "This checkout has no .venv yet ($VENV)." 'Run bash linux/setup.sh first, then this again.'
fi
if ! uv=$(find_uv); then
    refuse 'uv is not installed (or not on PATH in this shell).' 'Run bash linux/setup.sh first, then this again.'
fi
case "$ROOT$uv$ENV_FILE" in
    *$'\n'* | *'"'* | *\\*) refuse 'A path here has a newline, a double quote or a backslash, which a systemd unit cannot hold as written.' 'Move the checkout to a plain folder such as ~/AuraDream.' ;;
esac
ok "uv: $uv"

# --- The environment file ----------------------------------------------------------
# The names the page may be opened by: loopback first, this machine's name,
# its .local name (mDNS) and each IPv4 address it has now; lowercase, as
# browsers send them, and exact (the server compares names exactly).
allowed_hosts() {
    local name list address
    list='127.0.0.1,localhost'
    name=$(hostname 2>/dev/null || true)
    name=$(trim "${name,,}")
    if [ -n "$name" ] && [ "$name" != localhost ]; then
        name=${name%%.*}
        list="$list,$name,$name.local"
    fi
    for address in $(hostname -I 2>/dev/null || true); do
        if [[ "$address" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] && [[ ",$list," != *",$address,"* ]]; then
            list="$list,$address"
        fi
    done
    printf '%s\n' "$list"
}

env_text() {
    cat <<ENV
# AuraDream's service environment: read by systemd (EnvironmentFile= in
# $UNIT_NAME) and by linux/update.sh. linux/install-service.sh writes it
# once and never rewrites it, so edit it freely, then apply a change with:
#     systemctl --user restart auradream
# One NAME=value per line, no quotes. systemd expands nothing here (no \$HOME,
# no ~): every path must be absolute.

# Listen on every network interface, so other PCs can open the page. There is
# no login: anyone who can reach this port can use AuraDream.
AURADREAM_HOST=0.0.0.0

# The names the page may be opened by, comma-separated, written when the
# service was installed: this machine's name, its .local name and its IPv4
# addresses then. Add a name or address another PC uses (after a DHCP change,
# say). Exact lowercase names only: a wildcard (* or *.local) would load the
# page and then refuse every change the browser asks for.
# 127.0.0.1 and localhost must stay: the second-start check, the doctor
# (python -m backend.doctor) and the generate CLI all ask the server with
# Host: 127.0.0.1, and without it they take AuraDream for another program.
AURADREAM_ALLOWED_HOSTS=$(allowed_hosts)

# The Hub's xet downloader tunes its connection count adaptively, and on the
# measured connection that tuning collapsed to a single connection after one
# failed request and sat there - a 12 GB checkpoint stalled at 4.7 GB for
# twenty minutes, twice. A fixed count downloaded the same files in twenty
# minutes. (The same as wsl/env.sh.)
HF_XET_CLIENT_ENABLE_ADAPTIVE_CONCURRENCY=false
HF_XET_FIXED_DOWNLOAD_CONCURRENCY=32

# Examples; remove the # to use one.
# Checkpoint downloads on another disk (an absolute path; move an existing
# ~/.cache/huggingface there first to keep what is downloaded):
#HF_HOME=/mnt/data/huggingface
# Another port (the default is 7860):
#AURADREAM_PORT=7860
# Hold the PC awake (systemd-inhibit) while a job, load or download runs;
# false lets it suspend whenever the system decides:
#AURADREAM_KEEP_AWAKE=true
# How long after the last request it stays held awake, in minutes:
#AURADREAM_KEEP_AWAKE_IDLE_MINUTES=20
ENV
}

# --- The unit ---------------------------------------------------------------------
# systemd treats % as a specifier: a literal one is written %%.
unit_path() { printf '%s' "${1//%/%%}"; }

unit_text() {
    cat <<UNIT
# Written by linux/install-service.sh; run it again to rewrite this file.
# Settings go in $(unit_path "$ENV_FILE"), not here.
[Unit]
Description=AuraDream (local text-to-image server)
StartLimitIntervalSec=900
StartLimitBurst=3

[Service]
Type=simple
WorkingDirectory=$(unit_path "$ROOT")
EnvironmentFile=$(unit_path "$ENV_FILE")
Environment=PYTHONUTF8=1
# No --open (there is no desktop to open), and no systemd-inhibit wrapper:
# AuraDream holds the PC awake itself, only while it is busy.
ExecStart="$(unit_path "$uv")" run --no-sync python -m backend.main
Restart=on-failure
RestartSec=10
# A stop finishes the current step and any gallery bundle operation first.
TimeoutStopSec=120
UMask=0077
# KillMode is left at its default, control-group: a stop also ends the prompt
# critic's llama-server, which runs in a session of its own.

[Install]
WantedBy=default.target
UNIT
}

say ''
say "== The environment file ($ENV_FILE)"
if [ -f "$ENV_FILE" ]; then
    ok "Kept as it is: $ENV_FILE (it is never rewritten)."
elif [ "$dry_run" = 1 ]; then
    would "write $ENV_FILE (readable by you only):"
    env_text | sed 's/^/    /'
else
    mkdir -p -- "$(dirname "$ENV_FILE")"
    (
        umask 077
        env_text >"$ENV_FILE.part"
        chmod 600 "$ENV_FILE.part"
        mv -f -- "$ENV_FILE.part" "$ENV_FILE"
    )
    ok "Wrote $ENV_FILE (readable by you only)."
fi

say ''
say "== The unit ($UNIT_FILE)"
if [ "$dry_run" = 1 ]; then
    would "write $UNIT_FILE:"
    unit_text | sed 's/^/    /'
else
    mkdir -p -- "$(dirname "$UNIT_FILE")"
    unit_text >"$UNIT_FILE.part"
    mv -f -- "$UNIT_FILE.part" "$UNIT_FILE"
    ok "Wrote $UNIT_FILE."
fi

# --- systemd ---------------------------------------------------------------------
say ''
say '== systemd'
was_active=0
if [ "$dry_run" = 0 ] && service_active; then was_active=1; fi
no_manager='Run it from a login of your own (ssh or the console), not su or sudo; systemctl --user needs your user manager.'
user_systemctl daemon-reload || refuse 'systemctl --user daemon-reload failed: the unit is written, but your user manager did not read it.' "$no_manager"
if [ "$start" = 1 ]; then
    user_systemctl enable --now "$UNIT_NAME" || refuse 'systemctl --user enable --now failed (its message is above).' "Read its log: journalctl --user -u auradream -e. $no_manager"
    if [ "$was_active" = 1 ]; then
        note 'It was already running, and keeps the old settings until: systemctl --user restart auradream'
    elif [ "$dry_run" = 0 ]; then
        ok 'Enabled and started.'
    fi
else
    user_systemctl enable "$UNIT_NAME" || refuse 'systemctl --user enable failed (its message is above).' "$no_manager"
    [ "$dry_run" = 1 ] || ok 'Enabled (it starts with your user manager); not started now.'
fi

linger=$(loginctl show-user "$user" -p Linger 2>/dev/null || true)
if [ "$linger" = Linger=yes ]; then
    ok 'Lingering is on: the service starts at boot and keeps running with nobody logged in.'
elif [ -n "$linger" ]; then
    note 'Lingering is off: the service starts only when you log in, and stops when you log out.'
    say "         To run it from boot with nobody logged in: sudo loginctl enable-linger $user"
else
    note "Could not ask loginctl whether lingering is on. Without it, the service runs only while you are logged in: sudo loginctl enable-linger $user"
fi

# --- How to reach it ------------------------------------------------------------------
# The port and names as the service will read them: the env file's, which a
# dry run before the first install has not written yet.
if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    service_env=$(unset AURADREAM_PORT AURADREAM_ALLOWED_HOSTS && set -a && . "$ENV_FILE" && printf '%s\n%s\n' "${AURADREAM_PORT:-7860}" "${AURADREAM_ALLOWED_HOSTS:-}")
else
    service_env=$(printf '7860\n%s\n' "$(allowed_hosts)")
fi
port=$(sed -n 1p <<<"$service_env")
hosts=$(sed -n 2p <<<"$service_env")
say ''
say 'Open it from another PC on the network at:'
IFS=',' read -r -a names <<<"$hosts"
for name in "${names[@]}"; do
    name=$(trim "$name")
    case "$name" in
        '' | 127.0.0.1 | localhost) ;;
        *) say "    http://$name:$port/" ;;
    esac
done
say "  (On this machine: http://127.0.0.1:$port/. A .local name needs mDNS, such as avahi-daemon."
say "  If a firewall is on, allow the port: sudo ufw allow $port/tcp)"
cat <<MSG

Follow its log:   journalctl --user -u auradream -f
Stop / start:     systemctl --user stop auradream / systemctl --user start auradream
Its state:        systemctl --user status auradream
Update:           bash linux/update.sh --restart
Settings:         $ENV_FILE (then systemctl --user restart auradream)
MSG
if [ "$dry_run" = 1 ]; then
    say ''
    say 'Dry run finished. Nothing was changed.'
fi
