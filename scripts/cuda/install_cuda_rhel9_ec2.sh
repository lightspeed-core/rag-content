#!/usr/bin/env bash
# Install NVIDIA driver (DKMS) + nvidia-container-toolkit on RHEL 9 EC2 (x86_64 or aarch64).
#
# Unattended (survives reboots):
#   sudo ./install_cuda_rhel9_ec2.sh run
#   # SSH will drop twice while the instance reboots; no need to reconnect to continue.
#   # When finished: state file is "done" and the resume service is disabled.
#   ./install_cuda_rhel9_ec2.sh check
#
# Manual phases (optional):
#   ./install_cuda_rhel9_ec2.sh phase-a|phase-b|phase-c
#
# Tear down automation (e.g. to re-run from scratch):
#   sudo ./install_cuda_rhel9_ec2.sh reset
#
# Requires: root for run/daemon/reset; curl; working RHEL subscriptions.
#
# To provision the matching two-node CloudFormation lab and run this on both hosts: provision_cuda_lab.sh (same directory as this file).

set -euo pipefail

readonly THIS_SCRIPT="${BASH_SOURCE[0]:-$0}"
readonly INSTALL_BIN="/usr/local/sbin/install_cuda_rhel9_ec2.sh"
readonly STATE_DIR="/var/lib/nvidia-ec2-install"
readonly STATE_FILE="${STATE_DIR}/state"
readonly INSTALL_LOG="${STATE_DIR}/install.log"
readonly UNIT_PATH="/etc/systemd/system/nvidia-ec2-install-continue.service"

readonly NVIDIA_TESLA_VERSION="580.159.03"
readonly NVIDIA_CONTAINER_TOOLKIT_VERSION="1.19.0-1"

SUDO=""
[[ $EUID -eq 0 ]] || SUDO="sudo"

usage() {
  echo "Usage: $0 {run|daemon-continue|reset|phase-a|phase-b|phase-c|check}" >&2
  exit 1
}

require_root() {
  if [[ $EUID -ne 0 ]]; then
    exec sudo -E bash "$THIS_SCRIPT" "$@"
  fi
}

# True when THIS_SCRIPT and INSTALL_BIN are the same path on disk (skip cp — same source/dest).
this_script_is_install_bin() {
  [[ -e "$THIS_SCRIPT" && -e "$INSTALL_BIN" ]] || return 1
  local s d
  s=$(readlink -f "$THIS_SCRIPT" 2>/dev/null) || true
  d=$(readlink -f "$INSTALL_BIN" 2>/dev/null) || true
  if [[ -n "$s" && -n "$d" && "$s" == "$d" ]]; then
    return 0
  fi
  [[ "$(stat -c '%d:%i' "$THIS_SCRIPT" 2>/dev/null)" == "$(stat -c '%d:%i' "$INSTALL_BIN" 2>/dev/null)" ]]
}

install_self_and_unit() {
  mkdir -p "$(dirname "$INSTALL_BIN")"
  if ! this_script_is_install_bin; then
    cp -f "$THIS_SCRIPT" "$INSTALL_BIN"
  fi
  chmod 755 "$INSTALL_BIN"

  cat >"$UNIT_PATH" <<EOF
[Unit]
Description=Resume NVIDIA driver + container toolkit install after reboot
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=${INSTALL_BIN} daemon-continue
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

  systemctl daemon-reload
  systemctl enable nvidia-ec2-install-continue.service
}

read_state() {
  if [[ -f "$STATE_FILE" ]]; then
    tr -d '[:space:]' <"$STATE_FILE"
  else
    echo "new"
  fi
}

write_state() {
  mkdir -p "$STATE_DIR"
  printf '%s\n' "$1" >"$STATE_FILE"
}

log_install() {
  mkdir -p "$STATE_DIR" 2>/dev/null || true
  printf '%s %s\n' "$(date -Is)" "$*" >>"$INSTALL_LOG" 2>/dev/null || true
}

# On any failed command during advance_install, record failure so polling clients
# do not spin forever on e.g. post_phase_a after dnf/rpm errors.
on_install_err() {
  local ec=$?
  trap - ERR EXIT
  local prev
  prev="$(cat "$STATE_FILE" 2>/dev/null || echo none)"
  log_install "ERR exit=${ec} previous_state=${prev} line=${BASH_LINENO[0]}"
  write_state "failed" 2>/dev/null || true
  systemctl disable nvidia-ec2-install-continue.service 2>/dev/null || true
  systemctl daemon-reload 2>/dev/null || true
  exit "$ec"
}

# Safety-net EXIT trap: if advance_install exits non-zero without the ERR trap
# having fired (can happen with certain bash versions / function-call contexts),
# mark the state as failed so the poller doesn't spin forever.
on_install_exit() {
  local ec=$?
  trap - EXIT ERR
  if [[ $ec -ne 0 ]]; then
    local cur
    cur="$(cat "$STATE_FILE" 2>/dev/null || echo unknown)"
    if [[ "$cur" != "done" && "$cur" != "failed" ]]; then
      log_install "EXIT-trap exit=${ec} state=${cur} (ERR trap missed); marking failed"
      write_state "failed" 2>/dev/null || true
      systemctl disable nvidia-ec2-install-continue.service 2>/dev/null || true
      systemctl daemon-reload 2>/dev/null || true
    fi
  fi
}

phase_a() {
  # Update everything EXCEPT the kernel.  Letting dnf freely upgrade the
  # kernel risks DKMS build failures when a new kernel introduces API
  # changes (e.g. DRM signature changes in RHEL 9.8 5.14.0-687.10.1
  # broke NVIDIA 575.x).
  $SUDO dnf update -y --exclude='kernel*'

  # We need a kernel + kernel-devel pair.  The AMI's running kernel may
  # be an errata build whose -devel was never published (e.g.
  # 5.14.0-427.111.1 ships in the AMI but only 427.42.1 -devel exists).
  # Strategy: try the running kernel first; if its -devel is unavailable,
  # find the newest kernel-devel in the repos, install the matching
  # kernel-core + devel + headers, and set it as the boot default.
  local kver arch
  kver="$(uname -r)"
  arch="$(uname -m)"

  if $SUDO dnf install -y "kernel-devel-${kver}" "kernel-headers-${kver}" 2>/dev/null; then
    log_install "phase_a: kernel-devel for running kernel ${kver} installed"
    return 0
  fi

  log_install "phase_a: kernel-devel-${kver} not available; finding newest available pair"
  local best_kver
  best_kver="$(dnf repoquery --latest-limit=1 --qf '%{VERSION}-%{RELEASE}.%{ARCH}' \
    kernel-devel."${arch}" 2>/dev/null | tail -1)"
  if [[ -z "$best_kver" ]]; then
    log_install "phase_a: dnf repoquery found no kernel-devel for ${arch}"
    echo "No kernel-devel available in repos. Check RHUI mirror state." >&2
    return 1
  fi
  log_install "phase_a: installing kernel + devel for ${best_kver}"
  $SUDO dnf install -y \
    "kernel-core-${best_kver}" \
    "kernel-devel-${best_kver}" \
    "kernel-headers-${best_kver}"

  local target_vmlinuz="/boot/vmlinuz-${best_kver}"
  if [[ -f "$target_vmlinuz" ]]; then
    local current_default
    current_default="$(grubby --default-kernel 2>/dev/null || true)"
    if [[ "$current_default" != "$target_vmlinuz" ]]; then
      log_install "phase_a: setting default kernel to ${target_vmlinuz} (was ${current_default})"
      $SUDO grubby --set-default "$target_vmlinuz"
    fi
  fi
}

phase_b() {
  if ! rpm -q epel-release >/dev/null 2>&1; then
    $SUDO dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm
  else
    log_install "phase_b: epel-release already installed"
  fi

  local kver
  kver="$(uname -r)"

  # kernel-devel/headers are normally pre-installed by phase_a; install here
  # only if missing (e.g. manual phase-b invocation).
  if ! rpm -q "kernel-devel-${kver}" >/dev/null 2>&1; then
    log_install "phase_b: kernel-devel-${kver} not found, installing"
    $SUDO dnf install -y "kernel-devel-${kver}" "kernel-headers-${kver}"
  else
    log_install "phase_b: kernel-devel-${kver} already installed (from phase_a)"
  fi
  $SUDO dnf install -y dkms

  local arch nv_rpm url
  arch="$(uname -m)"
  nv_rpm="nvidia-driver-local-repo-rhel9-${NVIDIA_TESLA_VERSION}-1.0-1.${arch}.rpm"
  url="https://us.download.nvidia.com/tesla/${NVIDIA_TESLA_VERSION}/${nv_rpm}"

  if rpm -qa 'nvidia-driver-local-repo*' | grep -q .; then
    log_install "phase_b: nvidia-driver-local-repo RPM already installed, skipping download"
  else
    curl -fL -o "${nv_rpm}" "${url}"
    $SUDO rpm -U --replacepkgs "${nv_rpm}" || { rm -f "${nv_rpm}"; return 1; }
    rm -f "${nv_rpm}"
  fi
  $SUDO dnf clean all
  $SUDO dnf -y module install nvidia-driver:latest-dkms

  # Verify DKMS actually built AND installed the module for the running
  # kernel.  `dnf module install` can succeed (RPM installed) while the
  # DKMS build itself fails silently.  Require "installed" — not just
  # "built" — so the module is in /lib/modules/${kver} and will load
  # after the reboot into phase_c.
  local dkms_status
  dkms_status="$(dkms status nvidia/${NVIDIA_TESLA_VERSION} -k "${kver}" 2>/dev/null || true)"
  if ! echo "$dkms_status" | grep -q 'installed'; then
    log_install "phase_b: DKMS nvidia/${NVIDIA_TESLA_VERSION} not installed for ${kver} (status: ${dkms_status})"
    echo "DKMS module nvidia/${NVIDIA_TESLA_VERSION} is not installed for kernel ${kver}." >&2
    echo "dkms status: ${dkms_status:-<empty>}" >&2
    echo "Check: /var/lib/dkms/nvidia/${NVIDIA_TESLA_VERSION}/build/make.log" >&2
    return 1
  fi
  log_install "phase_b: DKMS installed: ${dkms_status}"
}

phase_c() {
  if ! command -v dnf >/dev/null 2>&1; then
    echo "dnf not found" >&2
    exit 1
  fi
  $SUDO dnf install -y dnf-plugins-core podman

  local repo_path=/etc/yum.repos.d/nvidia-container-toolkit.repo
  if [[ ! -f "$repo_path" ]]; then
    $SUDO curl -sSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
      -o "$repo_path"
  else
    log_install "phase_c: ${repo_path} already present, skipping download"
  fi
  $SUDO dnf config-manager --enable nvidia-container-toolkit-experimental

  local v="${NVIDIA_CONTAINER_TOOLKIT_VERSION}"
  $SUDO dnf install -y \
    "nvidia-container-toolkit-${v}" \
    "nvidia-container-toolkit-base-${v}" \
    "libnvidia-container-tools-${v}" \
    "libnvidia-container1-${v}"

  # nvidia-ctk cdi generate requires a loaded driver.  After the phase_b
  # reboot the DKMS module should load automatically, but verify before
  # running a command that will fail opaquely with "Driver Not Loaded".
  if ! lsmod | grep -q '^nvidia '; then
    $SUDO modprobe nvidia 2>/dev/null || true
  fi
  if ! lsmod | grep -q '^nvidia '; then
    log_install "phase_c: nvidia kernel module not loaded after modprobe — DKMS may have failed to build"
    echo "nvidia kernel module is not loaded. Check:" >&2
    echo "  dkms status" >&2
    echo "  /var/lib/dkms/nvidia/${NVIDIA_TESLA_VERSION}/build/make.log" >&2
    return 1
  fi

  $SUDO mkdir -p /etc/cdi
  if [[ -s /etc/cdi/nvidia.yaml ]] && command -v nvidia-ctk >/dev/null 2>&1; then
    log_install "phase_c: /etc/cdi/nvidia.yaml already present, skipping nvidia-ctk cdi generate"
  else
    $SUDO nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
  fi
}

check() {
  nvidia-ctk cdi list
  podman run --rm --device nvidia.com/gpu=all --security-opt=label=disable \
    nvcr.io/nvidia/cuda:12.9.1-devel-ubi9 nvidia-smi
}

advance_install() {
  trap on_install_err ERR
  trap on_install_exit EXIT
  local state
  state="$(read_state)"
  log_install "advance_install state=${state}"

  case "$state" in
    done)
      trap - ERR EXIT
      exit 0
      ;;
    failed)
      trap - ERR EXIT
      systemctl disable nvidia-ec2-install-continue.service 2>/dev/null || true
      systemctl daemon-reload 2>/dev/null || true
      log_install "refusing to continue (state=failed); fix host or run: $0 reset"
      echo "Install failed earlier — see ${INSTALL_LOG} and journalctl -u nvidia-ec2-install-continue.service -b" >&2
      exit 1
      ;;
    new)
      phase_a
      write_state post_phase_a
      trap - ERR EXIT
      reboot
      ;;
    post_phase_a)
      phase_b
      write_state post_phase_b
      trap - ERR EXIT
      reboot
      ;;
    post_phase_b)
      phase_c
      write_state "done"
      systemctl disable nvidia-ec2-install-continue.service || true
      systemctl daemon-reload
      trap - ERR EXIT
      log_install "install finished OK"
      echo "Install finished. Run: $INSTALL_BIN check   (as a user with podman access if needed)"
      ;;
    *)
      trap - ERR EXIT
      echo "Unknown state '$state' in $STATE_FILE — fix or run: $0 reset" >&2
      exit 1
      ;;
  esac
}

run_unattended() {
  require_root run

  local state
  state="$(read_state)"
  if [[ "$state" == "done" ]]; then
    echo "Already complete (state=done). Use '$0 check' or '$0 reset' then '$0 run'."
    exit 0
  fi

  install_self_and_unit
  # Ensure subsequent boots (and this process after re-exec) use the installed copy.
  exec bash "$INSTALL_BIN" _run-inner
}

_run_inner() {
  require_root _run-inner
  advance_install
}

daemon_continue() {
  require_root daemon-continue
  advance_install
}

reset_all() {
  require_root reset
  mkdir -p "$STATE_DIR"
  log_install "reset_all: removing phase_b/phase_c NVIDIA artifacts"

  # phase_c: container toolkit, libnvidia-container, CDI, repo file
  mapfile -t _ctk_pkgs < <(rpm -qa | grep -E '^nvidia-container-toolkit|^libnvidia-container' || true)
  if [[ ${#_ctk_pkgs[@]} -gt 0 ]]; then
    dnf remove -y "${_ctk_pkgs[@]}" 2>/dev/null || true
  fi
  rm -f /etc/cdi/nvidia.yaml
  rm -f /etc/yum.repos.d/nvidia-container-toolkit.repo
  dnf config-manager --disable nvidia-container-toolkit-experimental 2>/dev/null || true

  # phase_b: DKMS / driver packages, then NVIDIA local-repo RPM
  dnf -y module reset nvidia-driver 2>/dev/null || true
  mapfile -t _drv_pkgs < <(rpm -qa | grep -E '^kmod-nvidia|^nvidia-driver|^nvidia-fabric|^nvidia-modprobe|^nvidia-persistenced|^nvidia-settings' || true)
  if [[ ${#_drv_pkgs[@]} -gt 0 ]]; then
    dnf remove -y "${_drv_pkgs[@]}" 2>/dev/null || true
  fi
  if command -v dkms >/dev/null 2>&1; then
    while IFS= read -r line; do
      [[ -z "$line" ]] && continue
      [[ "$line" =~ ^nvidia/ ]] || continue
      modver="${line%%,*}"
      dkms remove "$modver" --all --rpm_safe_upgrade 2>/dev/null || true
    done < <(dkms status 2>/dev/null || true)
  fi
  rpm -e --allmatches 'nvidia-driver-local-repo-rhel9-*' 2>/dev/null || true
  rm -f ./nvidia-driver-local-repo-rhel9-*.rpm /root/nvidia-driver-local-repo-rhel9-*.rpm 2>/dev/null || true

  systemctl disable nvidia-ec2-install-continue.service 2>/dev/null || true
  rm -f "$UNIT_PATH"
  rm -rf "$STATE_DIR"
  systemctl daemon-reload
  echo "Reset complete. Run: sudo $THIS_SCRIPT run"
}

cmd="${1:-}"
case "${cmd}" in
  run) run_unattended ;;
  _run-inner) _run_inner ;;
  daemon-continue) daemon_continue ;;
  reset) reset_all ;;
  phase-a) phase_a ;;
  phase-b) phase_b ;;
  phase-c) phase_c ;;
  check) check ;;
  *) usage ;;
esac
