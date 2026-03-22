#!/usr/bin/env bash
# Create the CUDA lab CloudFormation stack (x86_64 G6 + aarch64 G5g), copy
# install_cuda_rhel9_ec2.sh to both instances, run unattended CUDA install, and
# wait until both report state "done".
#
# Prerequisites:
#   - AWS CLI configured (credentials + default region, or pass --region). The region
#     must appear under RegionAMIs in cuda_cloud_formation.yaml (script exits early otherwise).
#   - IAM: ec2:DescribeInstanceTypeOfferings (AZ selection), ec2:DescribeKeyPairs (verify ${username}-keys
#     exists before create-stack), plus CloudFormation/EC2 as needed for the stack.
#   - EC2 key pair named "${username}-keys" already exists in the target region
#   - Private key file at ~/.ssh/${username}-keys.pem (or --key-file)
#   - OpenSSH client (ssh, scp)
#
# Usage:
#   ./provision_cuda_lab.sh <username> [--region REGION] [--key-file PATH] [--stack-name NAME] [--allowed-cidr CIDR]
#   (If --allowed-cidr is omitted on stack create, the script uses this host's public IPv4/32 from checkip.amazonaws.com.)
#   ./provision_cuda_lab.sh <username> --reuse-stack   # stack exists; only copy + install + wait
#   ./provision_cuda_lab.sh <username> --stack-only    # create/wait for stack; skip CUDA install
#
# Environment:
#   POLL_TIMEOUT_SEC           max wall-clock seconds to wait for both hosts (default: 7200)
#   CUDA_LAB_KEY_FILE          default SSH private key path if --key-file is omitted
#   CUDA_LAB_ALLOWED_INGRESS_CIDR  optional; overrides auto-detected public IPv4/32 for SG ingress on stack create
#   KICKOFF_SSH_PROBE_RETRIES  after a non-zero kickoff SSH exit, how many probe attempts (default: 24)
#   KICKOFF_SSH_PROBE_SLEEP    seconds between probe attempts (default: 10)

set -euo pipefail

SCRIPT_DIR_TMP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || {
  echo "error: could not resolve script directory (cd or pwd failed)" >&2
  exit 1
}
[[ -n "$SCRIPT_DIR_TMP" ]] || {
  echo "error: script directory resolved to empty path" >&2
  exit 1
}
readonly SCRIPT_DIR="$SCRIPT_DIR_TMP"
unset SCRIPT_DIR_TMP
readonly CFN_TEMPLATE="${SCRIPT_DIR}/cuda_cloud_formation.yaml"
readonly INSTALL_SCRIPT="${SCRIPT_DIR}/install_cuda_rhel9_ec2.sh"
readonly DEFAULT_SSH_USER="ec2-user"

POLL_TIMEOUT_SEC="${POLL_TIMEOUT_SEC:-7200}"
KICKOFF_SSH_PROBE_RETRIES="${KICKOFF_SSH_PROBE_RETRIES:-24}"
KICKOFF_SSH_PROBE_SLEEP="${KICKOFF_SSH_PROBE_SLEEP:-10}"
SSH_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout=15
  -o StrictHostKeyChecking=accept-new
)

usage() {
  sed -n '1,20p' "$0" | tail -n +2 | head -n 14
  echo "Usage: $0 <username> [--region REGION] [--key-file PATH] [--stack-name NAME] [--allowed-cidr CIDR] [--reuse-stack] [--stack-only]" >&2
  exit 1
}

# Require a non-empty option value that is not another flag (avoids consuming --next-flag as a value).
opt_arg_or_usage() {
  [[ -n "${2:-}" && "${2:0:1}" != "-" ]] || usage
}

die() {
  echo "error: $*" >&2
  exit 1
}

# Public IPv4 of this host as seen on the internet (HTTPS). Used for AllowedIngressCidr /32 when not set explicitly.
detect_public_ingress_cidr() {
  local ip=""
  if command -v curl >/dev/null 2>&1; then
    ip="$(curl -sSf --connect-timeout 8 --max-time 20 "https://checkip.amazonaws.com/" 2>/dev/null | tr -d '[:space:]')"
  elif command -v wget >/dev/null 2>&1; then
    ip="$(wget -qO- --timeout=20 "https://checkip.amazonaws.com/" 2>/dev/null | tr -d '[:space:]')"
  else
    return 1
  fi
  [[ -n "$ip" ]] || return 1
  [[ "$ip" =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]] || return 1
  printf '%s/32\n' "$ip"
}

# Region keys under Mappings.RegionAMIs in cuda_cloud_formation.yaml (source of truth).
list_cuda_lab_supported_regions() {
  sed -n '/^  RegionAMIs:/,/^  CudaInstanceTypes:/p' "$CFN_TEMPLATE" \
    | grep -E '^    [a-z0-9-]+:' \
    | sed 's/^    //;s/:[[:space:]]*$//'
}

require_region_in_cfn_mapping() {
  local r="$1"
  if ! list_cuda_lab_supported_regions | grep -qxF "$r"; then
    echo "error: region ${r} is not listed under RegionAMIs in ${CFN_TEMPLATE}" >&2
    echo "Supported regions: $(list_cuda_lab_supported_regions | paste -sd ', ' -)" >&2
    echo "Add X8664 and Aarch64 AMI IDs for ${r} to that mapping, then retry." >&2
    exit 1
  fi
}

# InstanceType strings from Mappings.CudaInstanceTypes in cuda_cloud_formation.yaml.
cuda_lab_instance_type_x86() {
  sed -n '/^  CudaInstanceTypes:/,/^Resources:/p' "$CFN_TEMPLATE" \
    | sed -n '/^    X8664:/,/^    Aarch64:/p' | grep InstanceType | head -1 \
    | sed 's/.*InstanceType:[[:space:]]*"\([^"]*\)".*/\1/'
}

cuda_lab_instance_type_aarch64() {
  sed -n '/^  CudaInstanceTypes:/,/^Resources:/p' "$CFN_TEMPLATE" \
    | sed -n '/^    Aarch64:/,/^Resources:/p' | grep InstanceType | head -1 \
    | sed 's/.*InstanceType:[[:space:]]*"\([^"]*\)".*/\1/'
}

# AZs (e.g. us-east-1a) where an instance type is offered in this region.
list_azs_for_instance_type() {
  local reg="$1"
  local it="$2"
  aws ec2 describe-instance-type-offerings \
    --region "$reg" \
    --location-type availability-zone \
    --filters Name=instance-type,Values="$it" \
    --query 'InstanceTypeOfferings[].Location' \
    --output text 2>/dev/null | tr '\t' '\n' | sort -u
}

# First AZ (alphabetically) per instance type that offers it; used as stack parameters.
# Same ordering as sort -u on zone names from describe-instance-type-offerings.
resolve_lab_subnet_azs() {
  local reg="$1"
  local x86_type arm_type
  x86_type="$(cuda_lab_instance_type_x86)"
  arm_type="$(cuda_lab_instance_type_aarch64)"
  [[ -n "$x86_type" && -n "$arm_type" ]] \
    || die "could not parse CudaInstanceTypes from ${CFN_TEMPLATE}"

  local azs_x86 azs_arm
  azs_x86="$(list_azs_for_instance_type "$reg" "$x86_type")"
  azs_arm="$(list_azs_for_instance_type "$reg" "$arm_type")"
  [[ -n "$azs_x86" ]] || die "no AZ offers ${x86_type} in ${reg} (required for subnet A / x86_64)."
  [[ -n "$azs_arm" ]] || die "no AZ offers ${arm_type} in ${reg} (required for subnet B / aarch64)."

  local az_a az_b
  az_a="$(echo "$azs_x86" | head -1)"
  az_b="$(echo "$azs_arm" | head -1)"
  printf '%s %s\n' "$az_a" "$az_b"
}

# Print offerings and the AZ pair passed into the template as PublicSubnet*AvailabilityZone.
report_cuda_lab_instance_type_azs() {
  local reg="$1"
  local az_a="$2"
  local az_b="$3"
  local x86_type arm_type
  x86_type="$(cuda_lab_instance_type_x86)"
  arm_type="$(cuda_lab_instance_type_aarch64)"

  local az_x86 az_arm
  az_x86="$(list_azs_for_instance_type "$reg" "$x86_type" | paste -sd ', ' -)"
  az_arm="$(list_azs_for_instance_type "$reg" "$arm_type" | paste -sd ', ' -)"

  echo "EC2 availability zones offering lab instance types in ${reg}:"
  echo "  ${x86_type} (x86_64 / subnet A): ${az_x86}"
  echo "  ${arm_type} (aarch64 / subnet B): ${az_arm}"
  echo "Stack will use: PublicSubnetAAvailabilityZone=${az_a}, PublicSubnetBAvailabilityZone=${az_b}"
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

# cuda_cloud_formation.yaml uses KeyName "${username}-keys" on EC2 instances.
require_ec2_key_pair() {
  local region="$1"
  local user="$2"
  local kn="${user}-keys"
  if ! aws ec2 describe-key-pairs --region "$region" --key-names "$kn" >/dev/null 2>&1; then
    die "EC2 key pair '${kn}' not found in ${region}. Create it in that region (EC2 console or aws ec2 create-key-pair / import-key-pair), or pass a different <username> that matches an existing key pair name."
  fi
}

stack_output() {
  local stack="$1" key="$2"
  aws cloudformation describe-stacks \
    --stack-name "$stack" \
    --query "Stacks[0].Outputs[?OutputKey==\`${key}\`].OutputValue | [0]" \
    --output text
}

wait_ssh() {
  local label="$1" ip="$2" key="$3" user="$4"
  local deadline=$((SECONDS + 900))
  echo "Waiting for SSH on ${label} (${ip}) ..."
  while (( SECONDS < deadline )); do
    if ssh "${SSH_OPTS[@]}" -i "$key" "${user}@${ip}" "echo ok" >/dev/null 2>&1; then
      echo "SSH ready: ${label}"
      return 0
    fi
    sleep 10
  done
  die "SSH not available on ${label} (${ip}) within 15 minutes"
}

# After kickoff SSH fails, verify the installer actually started (matches install_cuda_rhel9_ec2.sh paths).
kickoff_verify_install_progress() {
  local ip="$1" key="$2" user="$3"
  local attempt=0
  while (( attempt < KICKOFF_SSH_PROBE_RETRIES )); do
    if ssh "${SSH_OPTS[@]}" -i "$key" "${user}@${ip}" \
      '[ -f /var/lib/nvidia-ec2-install/install.log ] || [ -f /var/lib/nvidia-ec2-install/state ] || [ -x /usr/local/sbin/install_cuda_rhel9_ec2.sh ]' \
      2>/dev/null; then
      return 0
    fi
    attempt=$((attempt + 1))
    sleep "$KICKOFF_SSH_PROBE_SLEEP"
  done
  return 1
}

kickoff_install() {
  local label="$1" ip="$2" key="$3" user="$4"
  echo "Starting unattended CUDA install on ${label} (${ip}) ..."
  # Connection usually drops on first reboot; install continues via systemd — only assume that if we see evidence.
  set +e
  ssh "${SSH_OPTS[@]}" -i "$key" "${user}@${ip}" \
    "chmod +x install_cuda_rhel9_ec2.sh && sudo ./install_cuda_rhel9_ec2.sh run"
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "SSH session to ${label} (${ip}) ended with exit ${rc}; verifying installer state on the host ..."
    if kickoff_verify_install_progress "$ip" "$key" "$user"; then
      echo "note: follow-up probe found install_cuda_rhel9_ec2.sh state under /var/lib/nvidia-ec2-install/ or the installed script — treating exit ${rc} as expected (e.g. reboot or connection drop)."
    else
      die "kickoff on ${label} (${ip}) failed: SSH exited ${rc} and follow-up probes (${KICKOFF_SSH_PROBE_RETRIES} x ${KICKOFF_SSH_PROBE_SLEEP}s) could not confirm installer progress (no /var/lib/nvidia-ec2-install/install.log or state, and /usr/local/sbin/install_cuda_rhel9_ec2.sh missing). Check sudo, script in ~${user}/, security group, and instance console."
    fi
  fi
}

# Poll install state over SSH. During reboots the host is often unreachable — that is normal.
poll_cuda_state() {
  local ip="$1" key="$2" user="$3"
  ssh "${SSH_OPTS[@]}" -i "$key" "${user}@${ip}" \
    "tr -d '[:space:]' </var/lib/nvidia-ec2-install/state 2>/dev/null || echo missing" \
    2>/dev/null || echo unreachable
}

normalize_state_note() {
  case "$1" in
    unreachable|missing) echo "no_ssh (reboot/network)" ;;
    *) echo "$1" ;;
  esac
}

wait_both_cuda_done() {
  local ip_x86="$1" ip_arm="$2" key="$3" user="$4"
  local deadline=$((SECONDS + POLL_TIMEOUT_SEC))
  echo "Waiting for CUDA install on both hosts (wall-clock timeout ${POLL_TIMEOUT_SEC}s)."
  echo "Status lines: time | x86_64 | aarch64  (unreachable during reboot is expected)"
  local s_x86 s_arm
  while (( SECONDS < deadline )); do
    s_x86="$(poll_cuda_state "$ip_x86" "$key" "$user")"
    s_arm="$(poll_cuda_state "$ip_arm" "$key" "$user")"
    printf '%s  x86_64=%s  aarch64=%s\n' "$(date +%H:%M:%S)" \
      "$(normalize_state_note "$s_x86")" "$(normalize_state_note "$s_arm")"
    if [[ "$s_x86" == "done" && "$s_arm" == "done" ]]; then
      echo "CUDA install complete on both hosts."
      return 0
    fi
    if [[ "$s_x86" == "failed" || "$s_arm" == "failed" ]]; then
      die "install failed (state=failed). On the affected host: sudo cat /var/lib/nvidia-ec2-install/install.log; sudo journalctl -u nvidia-ec2-install-continue.service -b --no-pager; then sudo ~/install_cuda_rhel9_ec2.sh reset (or fix) and re-run this script with --reuse-stack."
    fi
    sleep 45
  done
  die "Timeout — last x86_64=${s_x86} aarch64=${s_arm}"
}

run_check() {
  local label="$1" ip="$2" key="$3" user="$4"
  local remote_script="/home/${user}/install_cuda_rhel9_ec2.sh"
  echo "Running check on ${label} ..."
  ssh "${SSH_OPTS[@]}" -i "$key" "${user}@${ip}" \
    "sudo bash ${remote_script} check" \
    || die "check failed on ${label}"
}

main() {
  local username=""
  local region=""
  local key_file=""
  local stack_name=""
  local allowed_cidr=""
  local reuse_stack=0
  local stack_only=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --region) opt_arg_or_usage "$@"; region="$2"; shift 2 ;;
      --key-file) opt_arg_or_usage "$@"; key_file="$2"; shift 2 ;;
      --stack-name) opt_arg_or_usage "$@"; stack_name="$2"; shift 2 ;;
      --allowed-cidr) opt_arg_or_usage "$@"; allowed_cidr="$2"; shift 2 ;;
      --reuse-stack) reuse_stack=1; shift ;;
      --stack-only) stack_only=1; shift ;;
      -h|--help) usage ;;
      *)
        [[ -z "$username" ]] || usage
        username="$1"
        shift
        ;;
    esac
  done

  [[ -n "$username" ]] || usage
  [[ -f "$CFN_TEMPLATE" ]] || die "template not found: $CFN_TEMPLATE"
  [[ -f "$INSTALL_SCRIPT" ]] || die "install script not found: $INSTALL_SCRIPT"

  need_cmd aws
  need_cmd ssh
  need_cmd scp

  [[ -n "$region" ]] || region="$(aws configure get region 2>/dev/null || true)"
  [[ -n "$region" ]] || die "set AWS region (aws configure or --region)"

  require_region_in_cfn_mapping "$region"

  [[ -n "$stack_name" ]] || stack_name="${username}-cuda-stack"
  if [[ -z "$key_file" ]]; then
    for cand in "${CUDA_LAB_KEY_FILE:-}" "${HOME}/.ssh/${username}-keys.pem"; do
      [[ -n "$cand" && -f "$cand" ]] && key_file="$cand" && break
    done
  fi
  [[ -n "$key_file" && -f "$key_file" ]] || die "SSH private key not found (set --key-file or CUDA_LAB_KEY_FILE, or place key at ~/.ssh/${username}-keys.pem)"

  if [[ -z "$allowed_cidr" ]]; then
    allowed_cidr="${CUDA_LAB_ALLOWED_INGRESS_CIDR:-}"
  fi
  if [[ "$reuse_stack" -eq 0 && -z "$allowed_cidr" ]]; then
    echo "Detecting public IPv4 for AllowedIngressCidr (https://checkip.amazonaws.com/) ..."
    allowed_cidr="$(detect_public_ingress_cidr)" \
      || die "could not detect public IPv4 (need curl or wget, and outbound HTTPS). Set --allowed-cidr or CUDA_LAB_ALLOWED_INGRESS_CIDR."
    echo "Using AllowedIngressCidr=${allowed_cidr} (auto: this host's public IPv4)."
  fi
  if [[ "$reuse_stack" -eq 0 ]]; then
    [[ -n "$allowed_cidr" ]] || die "set --allowed-cidr or CUDA_LAB_ALLOWED_INGRESS_CIDR (IPv4 CIDR for SSH/app ingress; not 0.0.0.0/0)"
    [[ "$allowed_cidr" == "0.0.0.0/0" ]] && die "AllowedIngressCidr must not be 0.0.0.0/0 (use your public IP/32 or corporate CIDR)"
  fi

  export AWS_DEFAULT_REGION="$region"

  local subnet_az_a subnet_az_b
  read -r subnet_az_a subnet_az_b < <(resolve_lab_subnet_azs "$region")
  report_cuda_lab_instance_type_azs "$region" "$subnet_az_a" "$subnet_az_b"

  if [[ "$reuse_stack" -eq 0 ]]; then
    if aws cloudformation describe-stacks --stack-name "$stack_name" >/dev/null 2>&1; then
      die "stack ${stack_name} already exists — use --reuse-stack to only run install, or pick --stack-name"
    fi
    require_ec2_key_pair "$region" "$username"
    echo "Creating stack ${stack_name} in ${region} ..."
    aws cloudformation create-stack \
      --stack-name "$stack_name" \
      --template-body "file://${CFN_TEMPLATE}" \
      --parameters \
        "ParameterKey=username,ParameterValue=${username}" \
        "ParameterKey=PublicSubnetAAvailabilityZone,ParameterValue=${subnet_az_a}" \
        "ParameterKey=PublicSubnetBAvailabilityZone,ParameterValue=${subnet_az_b}" \
        "ParameterKey=AllowedIngressCidr,ParameterValue=${allowed_cidr}"

    echo "Waiting for stack create (this can take several minutes) ..."
    aws cloudformation wait stack-create-complete --stack-name "$stack_name"
    echo "Stack create complete."
  else
    aws cloudformation describe-stacks --stack-name "$stack_name" >/dev/null 2>&1 \
      || die "stack not found: $stack_name"
    echo "Reusing existing stack ${stack_name}."
  fi

  local ip_x86 ip_arm
  ip_x86="$(stack_output "$stack_name" X8664PublicIp)"
  ip_arm="$(stack_output "$stack_name" Aarch64PublicIp)"
  [[ -n "$ip_x86" && "$ip_x86" != "None" ]] || die "could not read X8664PublicIp from stack outputs"
  [[ -n "$ip_arm" && "$ip_arm" != "None" ]] || die "could not read Aarch64PublicIp from stack outputs"

  echo "Public IPs: x86_64=${ip_x86}  aarch64=${ip_arm}"

  if [[ "$stack_only" -eq 1 ]]; then
    echo "Stack-only mode: skipping CUDA install."
    echo "SSH:"
    echo "  ssh -i ${key_file} ${DEFAULT_SSH_USER}@${ip_x86}"
    echo "  ssh -i ${key_file} ${DEFAULT_SSH_USER}@${ip_arm}"
    exit 0
  fi

  wait_ssh "x86_64" "$ip_x86" "$key_file" "$DEFAULT_SSH_USER"
  wait_ssh "aarch64" "$ip_arm" "$key_file" "$DEFAULT_SSH_USER"

  scp "${SSH_OPTS[@]}" -i "$key_file" "$INSTALL_SCRIPT" "${DEFAULT_SSH_USER}@${ip_x86}:~/"
  scp "${SSH_OPTS[@]}" -i "$key_file" "$INSTALL_SCRIPT" "${DEFAULT_SSH_USER}@${ip_arm}:~/"

  kickoff_install "x86_64" "$ip_x86" "$key_file" "$DEFAULT_SSH_USER" &
  local pid_x86=$!
  kickoff_install "aarch64" "$ip_arm" "$key_file" "$DEFAULT_SSH_USER" &
  local pid_arm=$!
  wait "$pid_x86" || true
  wait "$pid_arm" || true

  wait_both_cuda_done "$ip_x86" "$ip_arm" "$key_file" "$DEFAULT_SSH_USER"

  run_check "x86_64" "$ip_x86" "$key_file" "$DEFAULT_SSH_USER"
  run_check "aarch64" "$ip_arm" "$key_file" "$DEFAULT_SSH_USER"

  echo ""
  echo "All done. Both instances have CUDA (driver + container toolkit) installed."
  echo "  x86_64 (G6/L4):  ssh -i ${key_file} ${DEFAULT_SSH_USER}@${ip_x86}"
  echo "  aarch64 (G5g):   ssh -i ${key_file} ${DEFAULT_SSH_USER}@${ip_arm}"
}

main "$@"
