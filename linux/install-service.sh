#!/usr/bin/env bash
# Run AuraDream as a systemd user service other PCs on the network can open.
# Safe to run again. From the checkout, after bash linux/setup.sh:
#
#     bash linux/install-service.sh              # write the unit, enable and start it
#     bash linux/install-service.sh --no-start   # write and enable; start it later
#     bash linux/install-service.sh --dry-run    # print the unit and settings; change nothing
#     bash linux/install-service.sh --uninstall  # stop, disable and remove the unit
#     bash linux/install-service.sh --print-unit # print only the unit it would write
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
print_unit=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) dry_run=1 ;;
        --uninstall) uninstall=1 ;;
        --no-start) start=0 ;;
        --print-unit) print_unit=1 ;;
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

# --- The unit ---------------------------------------------------------------------
# Defined before anything runs: linux/update.sh compares an installed unit
# with --print-unit's.
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
# 78: AuraDream refused its own settings (a passcode or TLS pair it cannot
# use); starting again cannot help, so the unit fails at once instead.
RestartPreventExitStatus=78
# A stop finishes the current step and any gallery bundle operation first.
TimeoutStopSec=120
UMask=0077
# KillMode is left at its default, control-group: a stop also ends the prompt
# critic's llama-server, which runs in a session of its own.

[Install]
WantedBy=default.target
UNIT
}

# --print-unit: the unit this version writes, on stdout and nothing else
# (linux/update.sh compares the installed one with it); changes nothing.
if [ "$print_unit" = 1 ]; then
    uv=$(find_uv) || exit 1
    unit_text
    exit 0
fi

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
# An environment file kept from before is sourced below for its port and
# names: not one whose passcode bash would read differently from systemd.
require_plain_env_passcode

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

# Listen on every network interface, so other PCs can open the page. Without
# AURADREAM_PASSCODE (below) there is no login: anyone who can reach this
# port can use AuraDream.
AURADREAM_HOST=0.0.0.0

# A passcode every browser, and every script that asks the server, must give:
# remove the # and write one of 8 characters or more after the = sign. In this
# file it may hold only letters, digits and . _ - + @ % : , / (no spaces, no
# quotes, no dollar sign, # or backslash): systemd and AuraDream's own scripts
# both read this file and agree only on such plain values. It sits here, and
# crosses the network over plain http, unencrypted: it keeps out people who can
# merely reach the address, so never reuse a password from anywhere else.
# Changing it and restarting signs every device out.
#AURADREAM_PASSCODE=

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
# Optional direct HTTPS: both existing PEM files, with a trusted certificate
# including the names used by browsers and 127.0.0.1 for these scripts:
#AURADREAM_SSL_CERTFILE=/absolute/path/cert.pem
#AURADREAM_SSL_KEYFILE=/absolute/path/key.pem
# Hold the PC awake (systemd-inhibit) while a job, load or download runs;
# false lets it suspend whenever the system decides:
#AURADREAM_KEEP_AWAKE=true
# How long after the last request it stays held awake, in minutes:
#AURADREAM_KEEP_AWAKE_IDLE_MINUTES=20
ENV
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
    service_env=$(unset AURADREAM_PORT AURADREAM_ALLOWED_HOSTS AURADREAM_PASSCODE AURADREAM_SSL_CERTFILE && set -a && . "$ENV_FILE" && printf '%s\n%s\n%s\n%s\n' "${AURADREAM_PORT:-7860}" "${AURADREAM_ALLOWED_HOSTS:-}" "${AURADREAM_PASSCODE:+set}" "$(server_scheme)")
else
    service_env=$(printf '7860\n%s\n\nhttp\n' "$(allowed_hosts)")
fi
port=$(sed -n 1p <<<"$service_env")
hosts=$(sed -n 2p <<<"$service_env")
scheme=$(sed -n 4p <<<"$service_env")
if [ -z "$(sed -n 3p <<<"$service_env")" ]; then
    warn 'No passcode is set: anyone who can reach this server can use it.'
    warn "Edit $ENV_FILE: set AURADREAM_PASSCODE to a new passcode of 8 characters or more, then run: systemctl --user restart auradream"
    if [ "$scheme" = http ]; then note 'Plain http leaves the passcode and pictures readable to anyone who can capture network traffic; never reuse another password.'; fi
fi
say ''
say 'Open it from another PC on the network at:'
IFS=',' read -r -a names <<<"$hosts"
for name in "${names[@]}"; do
    name=$(trim "$name")
    case "$name" in
        '' | 127.0.0.1 | localhost) ;;
        *) say "    $scheme://$name:$port/" ;;
    esac
done
say "  (On this machine: $scheme://127.0.0.1:$port/. A .local name needs mDNS, such as avahi-daemon."
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
