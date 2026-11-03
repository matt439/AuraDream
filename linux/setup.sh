#!/usr/bin/env bash
# Set up AuraDream on native Linux (not WSL: that is wsl/setup.sh). Safe to
# run again. From the checkout:
#
#     bash linux/setup.sh            # check the machine, then install
#     bash linux/setup.sh --dry-run  # check and print; change nothing
#
# It checks the NVIDIA driver and the card, installs uv only if it is
# missing (uv's own installer, at the version install.ps1 pins), then runs
# `uv python install 3.13`, `uv sync`, a CUDA check and
# `python -m backend.doctor`. It never uses sudo (a command that needs it is
# printed for you to run), never downloads model weights, never writes
# auradream.settings.json and never moves the gallery.
set -euo pipefail

# shellcheck source=linux/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# uv's own installer, pinned: the version install.ps1 pins.
UV_VERSION=0.12.7
UV_INSTALLER="https://astral.sh/uv/$UV_VERSION/install.sh"
DRIVER_ADVICE="Install the distribution's NVIDIA driver (on Ubuntu: sudo apt install nvidia-driver-595-open, or on a server without a desktop sudo apt install nvidia-headless-595-open nvidia-utils-595), reboot, and run bash linux/setup.sh again. Do not install the CUDA Toolkit."

dry_run=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) dry_run=1 ;;
        -h | --help)
            sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            say "Unknown option: $arg (try --help)" >&2
            exit 2
            ;;
    esac
done

would_stop=()
# A problem that makes AuraDream unusable: a real run stops here; a dry run
# says so and carries on, so the rest of the plan is still shown.
stop() {
    say "  [stop] $1"
    what_to_do "${2:-}"
    if [ "$dry_run" = 1 ]; then
        say '         (dry run: a real run would stop here; carrying on to show the rest)'
        would_stop+=("$1")
        return 0
    fi
    say ''
    say 'Setup stopped. Fix the problem above and run bash linux/setup.sh again.'
    exit 1
}

say 'AuraDream setup (native Linux)'
say "Project folder: $ROOT"
[ "$dry_run" = 0 ] || say 'Dry run: nothing will be changed; every action is printed instead.'

# 1. This machine ---------------------------------------------------------------
if grep -qi microsoft "${AURADREAM_INSTALL_TEST_PROC_VERSION:-/proc/version}" 2>/dev/null; then
    say '  [stop] This is WSL, which takes the Windows NVIDIA driver and its own setup.'
    what_to_do 'Run bash wsl/setup.sh instead (README.md, WSL 2).'
    exit 1
fi
if is_root; then
    say '  [stop] This is running as root (or under sudo).'
    what_to_do 'Run it as your own user: bash linux/setup.sh. The packages, .venv and the service belong to that user; a command that needs root is printed for you.'
    exit 1
fi
# The service's settings, so the port probed is the one it uses.
load_env_file || true
port=$(server_port)
running=
if service_active; then
    stop "The AuraDream service is running. Installing packages under it could break the ones it is using." 'Stop it with: systemctl --user stop auradream, run setup again, then: systemctl --user start auradream'
elif [ "$(auradream_state "$port")" = auradream ]; then
    stop "AuraDream is running at http://127.0.0.1:$port/. Installing packages under it could break the ones it is using." 'Stop it first (Ctrl-C in its terminal), then run bash linux/setup.sh again.'
elif running=$(venv_processes) && [ -n "$running" ]; then
    stop "A program from this folder's .venv is still running: $(paste -sd ',' <<<"$running" | sed 's/,/, /g'). It may be AuraDream on another port, or a command such as generate." 'Stop it (Ctrl-C in its terminal), then run bash linux/setup.sh again.'
else
    ok 'AuraDream is not running.'
fi

# 2. GPU and driver -------------------------------------------------------------
say ''
say '== NVIDIA GPU and driver'
if ! command -v nvidia-smi >/dev/null 2>&1; then
    stop 'nvidia-smi was not found, so there is no working NVIDIA driver. AuraDream needs an NVIDIA GPU.' "$DRIVER_ADVICE"
else
    smi_code=0
    header=$(nvidia-smi 2>&1) || smi_code=$?
    if [ "$smi_code" != 0 ]; then
        stop "nvidia-smi is installed but failed (exit $smi_code): the driver is not working (after a driver install, reboot first)." "$DRIVER_ADVICE"
    elif [[ "$header" =~ CUDA\ (UMD\ )?Version:\ *([0-9]+)\.([0-9]+) ]]; then
        cuda_major=${BASH_REMATCH[2]}
        cuda_minor=${BASH_REMATCH[3]}
        if [ "$cuda_major" -lt 13 ]; then
            stop "The driver supports CUDA $cuda_major.$cuda_minor; AuraDream's PyTorch needs CUDA 13.0 or higher." "$DRIVER_ADVICE"
        else
            ok "The driver supports CUDA $cuda_major.$cuda_minor (13.0 or higher is needed)."
        fi
    else
        warn "nvidia-smi ran, but its CUDA version could not be read. Carrying on; the CUDA check after the install is the one that counts."
    fi
    query_code=0
    query=$(nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader 2>&1) || query_code=$?
    first=$(grep -m1 ',' <<<"$query" || true)
    if [ "$query_code" != 0 ] || [ -z "$first" ]; then
        warn 'Could not read the card model and generation from nvidia-smi. Carrying on; the CUDA check after the install will tell.'
    else
        IFS=',' read -r gpu_name gpu_cap gpu_mem _ <<<"$first"
        gpu_name=$(trim "$gpu_name")
        gpu_cap=$(trim "$gpu_cap")
        gpu_mem=$(trim "${gpu_mem:-}")
        say "  Card: $gpu_name"
        if [[ "$gpu_cap" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
            cap_major=${BASH_REMATCH[1]}
            cap_minor=${BASH_REMATCH[2]}
            if [ "$cap_major" -lt 7 ] || { [ "$cap_major" = 7 ] && [ "$cap_minor" -lt 5 ]; }; then
                stop "$gpu_name has compute capability $cap_major.$cap_minor. The CUDA 13 PyTorch AuraDream installs supports 7.5 to 12.0 (sm_75 to sm_120): an RTX 20 or GTX 16 series card or newer. This card is not supported." 'AuraDream cannot run on this card. A newer NVIDIA card is needed.'
            else
                ok "Compute capability $cap_major.$cap_minor is supported (7.5 to 12.0)."
                if [ "$cap_major" -lt 10 ]; then
                    note "Nunchaku 4-bit (AURADREAM_QUANT=nunchaku) needs the int4 repos on this card (lite-infer/...-nunchaku-lite-int4...); the nvfp4 ones need an RTX 50 (compute capability 10.0 or higher)."
                fi
            fi
        else
            warn "Could not read the card's compute capability from '$first'. Carrying on; the CUDA check after the install will tell."
        fi
        if [[ "$gpu_mem" =~ ^([0-9]+)( MiB)?$ ]]; then
            vram_gb=$(awk -v m="${BASH_REMATCH[1]}" 'BEGIN { printf "%.1f", m / 1024 }')
            if awk -v g="$vram_gb" 'BEGIN { exit !(g >= 15.5) }'; then
                ok "$vram_gb GB of video memory. The measured card is a 16 GB RTX 5080."
            else
                note "$vram_gb GB of video memory. Only a 16 GB card has been measured; whether a checkpoint fits less is not known. The SDXL anime checkpoints are the lightest, and sequential offload (under Models) uses less video memory, more slowly."
            fi
        fi
        cards=$(grep -c ',' <<<"$query" || true)
        [ "$cards" -le 1 ] || note "$cards NVIDIA cards found; AuraDream uses the first one listed."
    fi
fi

# 3. System libraries -------------------------------------------------------------
say ''
say '== System libraries'
if command -v cc >/dev/null 2>&1 || command -v gcc >/dev/null 2>&1; then
    ok 'A C compiler is installed (torch compiles a small module for its Triton kernels on first use).'
else
    note "No C compiler here. torch's Triton kernels compile a small C module the first time they run, and some checkpoints' first image fails without one ('Failed to find C compiler')."
    say "         Install one with: sudo apt install gcc (or set TORCH_DISABLE_NATIVE_JIT=1 in the service's environment file, as wsl/env.sh does)."
fi
# The prompt critic's pinned llama.cpp build (backend/critic.py) takes glibc
# 2.38 or newer and three libraries from the system rather than its own tree.
# The app checks the same before the critic's first 5 GB download, and
# refuses there without them. (Copied from wsl/setup.sh.)
glibc=$(getconf GNU_LIBC_VERSION 2>/dev/null || true)
glibc=${glibc#glibc }
if [ -n "$glibc" ] && ! printf '%s\n' 2.38 "$glibc" | sort -V -C; then
    note "glibc $glibc is older than the 2.38 the prompt critic's llama.cpp build needs (Ubuntu 24.04 or newer), so the page will not offer the critic."
else
    libraries=$("$(command -v ldconfig || echo /sbin/ldconfig)" -p 2>/dev/null || true)
    critic_packages=""
    for need in libgomp.so.1:libgomp1 libssl.so.3:libssl3 libcrypto.so.3:libssl3; do
        if [ -n "$libraries" ] && ! grep -q "[[:space:]]${need%%:*} (.*x86-64" <<<"$libraries"; then
            case " $critic_packages " in
                *" ${need#*:} "*) ;;
                *) critic_packages="$critic_packages ${need#*:}" ;;
            esac
        fi
    done
    if [ -n "$critic_packages" ]; then
        note "The prompt critic's llama.cpp needs$critic_packages, not installed here; 'sudo apt install$critic_packages' before its first use."
    else
        ok "The prompt critic's system libraries are here (glibc ${glibc:-unknown}, libgomp1, libssl3)."
    fi
fi

# 4. uv and Python 3.13 -----------------------------------------------------------
say ''
say '== uv and Python 3.13'
installed_uv=0
if uv=$(find_uv); then
    ok "uv is installed: $uv"
elif [ "$dry_run" = 1 ]; then
    would "install uv with uv's own installer, version $UV_VERSION: curl -LsSf $UV_INSTALLER | sh"
    uv=uv
elif ! command -v curl >/dev/null 2>&1; then
    stop 'uv is not installed, and curl, which its installer is fetched with, is not either.' 'Install curl (sudo apt install curl), then run bash linux/setup.sh again.'
else
    say "  Installing uv $UV_VERSION with uv's own installer ($UV_INSTALLER)..."
    curl -LsSf "$UV_INSTALLER" | sh
    # For the rest of this script only; the installer's rc-file edit reaches
    # only future shells, which the end of this script says.
    export PATH="$HOME/.local/bin:$PATH"
    if uv=$(find_uv); then
        ok "uv installed: $uv"
        installed_uv=1
    else
        stop "uv's installer ran, but uv was not found afterwards." 'Read its output above, then run bash linux/setup.sh again.'
    fi
fi
if [ "$dry_run" = 1 ]; then
    would "run: uv python install 3.13 (the project needs Python 3.13, not 3.14)"
else
    "$uv" python install 3.13
    ok 'Python 3.13 is installed for uv.'
fi

# 5. Packages and the CUDA check --------------------------------------------------
say ''
say '== Packages (uv sync) and the CUDA check'
cuda_check="import torch; ok = torch.cuda.is_available(); print(ok, torch.cuda.get_device_name(0) if ok else '')"
if [ "$dry_run" = 1 ]; then
    would "run in $ROOT: uv sync (creates .venv and downloads a few GB, PyTorch from the CUDA 13 index; never model weights)"
else
    say '  Installing the packages. The first time downloads a few GB; this can take a while.'
    sync_code=0
    (cd "$ROOT" && "$uv" sync) || sync_code=$?
    if [ "$sync_code" != 0 ]; then
        stop "uv sync failed (exit $sync_code)." 'Check the internet connection (and any proxy or firewall), then run bash linux/setup.sh again: it carries on where it stopped.'
    fi
    ok 'Packages installed.'
fi
if [ -x "$VENV/bin/python" ]; then
    say '  Checking that PyTorch sees the GPU...'
    check_code=0
    out=$("$VENV/bin/python" -c "$cuda_check" 2>&1) || check_code=$?
    if [ "$check_code" != 0 ]; then
        stop "The CUDA check could not run (exit $check_code): $out" 'Run bash linux/setup.sh again; if it repeats, delete the .venv folder and run it again.'
    elif [[ "$out" == *"is not compatible with the current PyTorch"* ]]; then
        stop 'PyTorch sees the card but says it is not compatible: the card is older than the CUDA 13 build supports (RTX 20 / GTX 16 series or newer).' 'A newer NVIDIA card is needed.'
    elif [[ "$out" =~ (^|$'\n')True[[:space:]]+([^$'\n']+) ]]; then
        ok "PyTorch can use the GPU: ${BASH_REMATCH[2]}."
    else
        stop 'PyTorch cannot use the GPU (torch.cuda.is_available() is False). The driver is too old for CUDA 13, or not working.' "$DRIVER_ADVICE"
    fi
elif [ "$dry_run" = 1 ]; then
    would "check that PyTorch sees the GPU: .venv/bin/python -c \"$cuda_check\""
fi

# 6. Doctor -------------------------------------------------------------------------
say ''
say '== Machine check (python -m backend.doctor)'
if [ "$dry_run" = 1 ]; then
    would "run in $ROOT: uv run python -m backend.doctor (read-only)"
else
    doctor_code=0
    (cd "$ROOT" && "$uv" run python -m backend.doctor) || doctor_code=$?
    if [ "$doctor_code" != 0 ]; then
        warn 'The check found a problem (FAIL above). AuraDream may not work until it is fixed; the line says what to do.'
    else
        ok 'No problems found.'
    fi
fi

# 7. Model downloads ----------------------------------------------------------------
say ''
say '== Model downloads'
hf_home=${HF_HOME:-$HOME/.cache/huggingface}
if [ -d "$hf_home" ]; then
    hf_size=$(du -sh "$hf_home" 2>/dev/null | cut -f1 || true)
    say "  Checkpoints download to $hf_home (now ${hf_size:-of unknown size}) the first time each is loaded."
else
    say "  Checkpoints download to $hf_home (not made yet) the first time each is loaded."
fi
hf_parent=$hf_home
while [ ! -d "$hf_parent" ] && [ "$hf_parent" != / ]; do hf_parent=$(dirname "$hf_parent"); done
hf_free=$(trim "$(df -h --output=avail "$hf_parent" 2>/dev/null | tail -n 1 || true)")
[ -z "$hf_free" ] || say "  That disk has $hf_free free. One checkpoint is often tens of GB."
say '  To keep them on another disk, set HF_HOME to a folder there (an absolute path): for the service,'
say "  in its environment file ($ENV_FILE, which install-service.sh writes); for a terminal, export it."

# Done --------------------------------------------------------------------------------
say ''
if [ "$dry_run" = 1 ]; then
    if [ "${#would_stop[@]}" -gt 0 ]; then
        say 'Dry run finished. Nothing was changed. A real run would stop at:'
        for reason in "${would_stop[@]}"; do say "  - $reason"; done
        exit 1
    fi
    say 'Dry run finished. Nothing was changed.'
    exit 0
fi
if [ "$installed_uv" = 1 ]; then
    cat <<MSG
uv was just installed. This shell cannot see it yet; run this first:

    . "\$HOME/.local/bin/env"

MSG
fi
cat <<MSG
Ready. Next:

    uv run python -m backend.main      # a local try: http://127.0.0.1:$port on this machine
    bash linux/install-service.sh      # run it as a service other PCs on the network can open
    uv run hf auth login               # only for the gated checkpoints (FLUX.1-Krea-dev, krea/Krea-2-Turbo)
MSG
