# CUDA lab on AWS (RHEL 9)

This directory contains a small **two-node GPU lab** on EC2:

| Host        | Architecture | Instance type (see template) | Role        |
|------------|----------------|---------------------------------|------------|
| x86_64     | `x86_64`       | `g6.2xlarge` (NVIDIA L4 class)  | Subnet A   |
| aarch64    | `arm64`        | `g5g.4xlarge` (NVIDIA T4G)     | Subnet B   |

Both run RHEL 9.4 (Hourly2-GP3 marketplace-style AMIs defined in `cuda_cloud_formation.yaml`). The stack creates a VPC, two public subnets (often one availability zone so both GPUs are valid and capacity is sane), security groups, and the two instances.

**End-to-end automation** (stack + copy installer + unattended driver/container toolkit install + wait): `provision_cuda_lab.sh`.

---

## ADR: kernel excluded from blanket update, driver version is explicit

**Context.** `phase_a` originally ran `dnf update -y` with no exclusions. This silently upgraded the kernel to whatever was newest in the AWS RHUI mirror. Two problems followed:

1. **DKMS build failure (x86_64).** RHEL 9.8 kernel `5.14.0-687.10.1` changed the DRM framebuffer API (`drm_helper_mode_fill_fb_struct` gained a `drm_format_info *` parameter). The NVIDIA 575.57.08 DKMS build failed with incompatible-pointer-type errors against the new headers.
2. **Missing `kernel-devel` (aarch64).** The RHUI `baseos` repo published `kernel-core-5.14.0-687.10.1` before `appstream` published the matching `kernel-devel`. `phase_b` could not install `kernel-devel-$(uname -r)` because it didn't exist yet.

**Decision.** Exclude kernel from `dnf update`; find the newest kernel+devel pair available; keep the driver version explicit.

- `phase_a` runs `dnf update -y --exclude='kernel*'` so userspace security patches still land but the kernel isn't blindly upgraded.
- `phase_a` then tries to install `kernel-devel` for the running kernel. If that exact `-devel` isn't in the repos (common with errata AMI kernels like `5.14.0-427.111.1`), it falls back to `dnf repoquery` for the newest `kernel-devel` available, installs the matching `kernel-core` + `kernel-devel` + `kernel-headers`, and sets grubby's default boot to that kernel.
- `NVIDIA_TESLA_VERSION` is set at the top of `install_cuda_rhel9_ec2.sh`. When updating the AMI (new `RegionAMIs` mapping), re-validate the driver version builds cleanly against the available kernel before merging.

**Alternatives considered.**

| Approach | Why not |
|---|---|
| Full `dnf update` (no excludes) + find a compatible kernel-devel | The blanket kernel upgrade can pull a kernel whose DRM API is incompatible with the pinned driver, and the matching `-devel` may not be published yet. |
| Pin the kernel to the exact AMI version only | AMI errata kernels (e.g. `427.111.1`) often have no `-devel` in the repos. |
| Pin both kernel *and* driver to exact NVRs in the template | Over-constrained; userspace CVE patches would also be blocked. |
| No `dnf update` at all | Leaves known CVEs in openssl, glibc, etc. unpatched on the instance. |

---

## Prerequisites

- **AWS CLI** configured (`aws configure` or environment variables) with permission to create the stack and use EC2 APIs.
- **Region** must be listed under **`Mappings.RegionAMIs`** in `cuda_cloud_formation.yaml`. The provision script refuses other regions so AMI lookup stays explicit.
- **EC2 key pair** in that region named **`{username}-keys`** (the first script argument is that `username` prefix).
- **SSH private key** on your machine, default paths tried:
  `CUDA_LAB_KEY_FILE`, `~/.ssh/{username}-keys.pem`, or pass **`--key-file`**.
- **OpenSSH** (`ssh`, `scp`).
- **IAM** (typical): CloudFormation, EC2, VPC; **`ec2:DescribeInstanceTypeOfferings`** is required so the provision script can pick AZs where **`g6.2xlarge`** and **`g5g.4xlarge`** are offered and pass them into the template.

**RHEL subscriptions** on the instances are required for `dnf` during the NVIDIA install (same as any RHEL 9 workload).

---

## Quick start (recommended)

From this directory:

```bash
./provision_cuda_lab.sh <username> --region us-east-1
```

**`<username>`** is the same string you used for the EC2 key pair name **`<username>-keys`** (and the default stack name is **`<username>-cuda-stack`**).

- **`--region`** defaults to `aws configure` if omitted.
- **`--allowed-cidr`**: if omitted on **stack create**, the script discovers **this machine’s public IPv4** via `https://checkip.amazonaws.com/` and passes **`{ip}/32`** as **`AllowedIngressCidr`**. Override with **`--allowed-cidr 203.0.113.10/32`** or **`CUDA_LAB_ALLOWED_INGRESS_CIDR`** if you need a different range (office VPN, etc.). **`0.0.0.0/0`** is **not** allowed.

The script creates the CloudFormation stack, waits for it, copies `install_cuda_rhel9_ec2.sh` to both instances, starts the unattended install, and polls until both report **`done`**.

---

## `provision_cuda_lab.sh` usage

```text
./provision_cuda_lab.sh <username> [--region REGION] [--key-file PATH] [--stack-name NAME]
  [--allowed-cidr CIDR] [--reuse-stack] [--stack-only]
```

| Flag / mode | Meaning |
|-------------|---------|
| `<username>` | Prefix for resource names and for the key pair **`{username}-keys`**. |
| `--region` | AWS region (must appear in **`RegionAMIs`**). |
| `--key-file` | Path to SSH private key. |
| `--stack-name` | Default: **`{username}-cuda-stack`**. |
| `--allowed-cidr` | IPv4 CIDR for security group ingress (SSH + HTTP/S ports). Not used with **`--reuse-stack`**. |
| `--reuse-stack` | Stack already exists: skip create; only copy installer, run install, wait. |
| `--stack-only` | Create stack and wait; print SSH hints; **no** CUDA install. |

**Environment variables** (optional):

| Variable | Purpose |
|----------|---------|
| `POLL_TIMEOUT_SEC` | Max seconds waiting for both installs (default `7200`). |
| `CUDA_LAB_KEY_FILE` | Default key path if `--key-file` omitted. |
| `CUDA_LAB_ALLOWED_INGRESS_CIDR` | Overrides auto-detected CIDR for stack create. |
| `KICKOFF_SSH_PROBE_RETRIES` / `KICKOFF_SSH_PROBE_SLEEP` | After a non-zero SSH exit from kickoff, probes for installer artifacts on the host. |

---

## CloudFormation template (`cuda_cloud_formation.yaml`)

You can deploy manually; the template expects parameters:

- **`username`** — name prefix.
- **`PublicSubnetAAvailabilityZone`** / **`PublicSubnetBAvailabilityZone`** — AZs for subnets A/B. `provision_cuda_lab.sh` sets both via `describe-instance-type-offerings`, randomly choosing among AZs where both GPU types are offered when possible (often the same AZ for both subnets).
- **`AllowedIngressCidr`** — must be a valid IPv4 CIDR and cannot be `0.0.0.0/0`.

Comments at the top of the YAML file show an example `aws cloudformation create-stack` invocation.

Validate locally:

```bash
aws cloudformation validate-template --template-body file://cuda_cloud_formation.yaml
```

---

## On-instance install (`install_cuda_rhel9_ec2.sh`)

Normally invoked by **`provision_cuda_lab.sh`**. You can run it yourself on either instance:

```bash
sudo ./install_cuda_rhel9_ec2.sh run      # unattended (survives reboots)
./install_cuda_rhel9_ec2.sh check           # verify GPU + podman (after done)
sudo ./install_cuda_rhel9_ec2.sh reset      # tear down automation state + NVIDIA packages (see script)
```

State and logs: **`/var/lib/nvidia-ec2-install/`** (`state`, `install.log`). A systemd unit resumes after reboot until the install finishes or fails.

---

## Makefile (local checks and AWS helpers)

```bash
make -C scripts/cuda help
```

**Local (no instances):** `check-syntax`, `check-shellcheck`, `validate-cfn`, `test-opt-guards`, **`test-local`** (runs the main checks).

**AWS (costs / side effects):** `stack-only`, `lab-full`, `reuse-install`, `delete-stack` — set **`LAB_USER`**, **`REGION`**, and optionally **`STACK_NAME`**, **`ALLOWED_CIDR`**, **`KEY_FILE`** as documented in the Makefile **`help`** target.

---

## Security notes

- Ingress is restricted to **`AllowedIngressCidr`** (SSH **22**, HTTP **80** / **8000**, HTTPS **443** / **8443**). Ensure the CIDR includes the addresses you use to SSH from (often **`your.public.ip/32`**).
- Outbound from instances remains open for package installs; tighten in the template if your policy requires it.

---

## Cost and cleanup

GPU instances are **not** free. Delete the stack when finished:

```bash
aws cloudformation delete-stack --stack-name <username>-cuda-stack --region <region>
```

Or use **`make -C scripts/cuda delete-stack STACK_NAME=... REGION=...`**.

---

## Troubleshooting

- **Stack create fails:** `aws cloudformation describe-stack-events --stack-name ... --region ...`
- **Provision script fails early:** Region not in **`RegionAMIs`**, missing key pair, or invalid **`AllowedIngressCidr`**.
- **SSH works but install stalls:** On the instance, `sudo cat /var/lib/nvidia-ec2-install/install.log`, `sudo journalctl -u nvidia-ec2-install-continue.service -b --no-pager`, and `sudo ~/install_cuda_rhel9_ec2.sh reset` if you need a clean retry, then re-run **`provision_cuda_lab.sh ... --reuse-stack`** from your laptop.
