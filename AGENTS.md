# Agent notes: Konflux, Hermeto, and hermetic Python lockfiles

This file captures lessons from debugging **prefetch-dependencies / Hermeto** failures (checksum mismatch, “No wheels found”, `pybuild-deps` errors) and aligning **CPU** and **CUDA** requirement generators. Use it when changing `pyproject.toml`, RHOAI indices, or `.tekton` prefetch inputs.

## Hermeto behavior that breaks naive lockfiles

- **Konflux invocation** (shape to mirror locally):  
  `hermeto --log-level debug --mode strict fetch-deps '<json>' --sbom-output-type spdx --source <repo> --output <dir>`  
  The pip slice of `<json>` matches `.tekton` `prefetch-input` (see `scripts/hermeto/*.json` and `scripts/run_hermeto_fetch_deps.sh`).

- **PyPI intersection for wheels**  
  For many binary packages, Hermeto logs lines like:  
  `using intersection of requirements-file and PyPI-reported checksums`  
  It then **drops** any wheel whose digest is not in that intersection.

- **Implication for RHOAI**  
  If a line uses **`package==version`** with **`--index-url`** pointing at RHOAI but the **actual wheel file** is a RHOAI rebuild (e.g. `torch-2.9.0-7-cp312-cp312-linux_x86_64.whl`), PyPI usually **does not** expose the same filename/digest. **Every** candidate can be filtered → `PackageRejected: No wheels found`.

- **What fixes it**  
  - Prefer **`name @ https://…/exact-file.whl`** plus a **single `--hash=sha256:…`** for bytes you control (pulp or `files.pythonhosted.org`), so Hermeto fetches by URL and verifies the hash **without** requiring a PyPI match for that RHOAI rebuild name.  
  - For **CPU multi-arch**, the prefetch lists **both** arch-specific requirement fragments; **`Containerfile`** installs **one** fragment according to `TARGETARCH` (see below).

- **Local Hermeto**  
  **`full-cpu` / `full-cuda`** may fail without **RHSM** client cert paths—Konflux injects `options.ssl` on the **rpm** input; laptops typically do not have `/etc/pki/entitlement/…`. Prefer the **pip slice** for routine validation (see below).

## Local Hermeto validation (pip slice)

**CPU green is not CUDA green.** Konflux runs **separate** prefetch inputs for CPU vs CUDA images. The committed files differ (`requirements.hashes.*` vs `requirements.hashes.*.cuda*`, **`requirements.overrides.txt`** vs **`requirements.overrides.cuda.txt`**), and the generator can put the same package in **wheel** lists for one pipeline and **source** for the other. Validating only **`pip-cpu`** misses failures that appear only in **`pip-cuda`** (and the reverse is possible in principle).

**Hermeto runs `cargo vendor --locked` on extracted Python sdists** that contain Rust (e.g. under `deps/pip/<name>-<ver>/`). That step is **independent of whether you build a CUDA image**—it triggers when the **pip prefetch** delivers an sdist with a `Cargo.toml`/`Cargo.lock` mismatch (example: historical **`jiter` 0.12.x** → `PackageWithCorruptLockfileRejected`). A package resolved as a **manylinux wheel** in CPU can still be an **sdist** on the CUDA requirement split if you do not run **`pip-cuda`**.

**What to run before pushing prefetch or hashed-requirement changes**

| Change touched | Run |
|----------------|-----|
| CPU only (`konflux_requirements.sh`, `requirements.hashes.*` not `.cuda`) | `make hermeto-verify-pip-cpu` |
| CUDA only (`konflux_requirements_cuda.sh`, `*.cuda*`, `requirements.overrides.cuda.txt`) | `make hermeto-verify-pip-cuda` |
| Shared: `pyproject.toml`, `uv.lock`, both overrides files, or both generators | `make hermeto-verify-pip` (CPU **and** CUDA) |

Commands use **strict** Hermeto (same shape as Konflux) and separate output dirs so results are not overwritten:

- `make hermeto-verify-pip-cpu` → **`.hermeto-output-verify-cpu/`**  
- `make hermeto-verify-pip-cuda` → **`.hermeto-output-verify-cuda/`**  

Ad-hoc: `HERMETO_OUT=/path ./scripts/run_hermeto_fetch_deps.sh pip-cuda`. Generic **`make hermeto-fetch-deps`** still defaults to **`HERMETO_MODE=pip-cpu`**—do not treat that alone as sufficient when CUDA inputs changed.

## CPU pipeline (`scripts/konflux_requirements.sh`)

- **Regenerate; do not hand-edit** hashed requirements. Use `make konflux-requirements` (or the script).

- **`--extra-index-url` in lockfiles**  
  Hermeto does not support that line in committed files. The script passes PyPI as an extra index **during** `uv pip compile`, then **`sed`** removes `--extra-index-url` from the generated file.

- **Torch / triton (RHOAI) vs Hermeto**  
  `torch==` / `triton==` under the RHOAI simple index **fail** Hermeto’s PyPI intersection. The generator **strips** those stanzas from `requirements.hashes.wheel.txt` and writes:  
  - `requirements.hashes.wheel.cpu.x86_64.txt` — pulp URLs + hashes for **torch** and **triton**; **torchvision** from **PyPI manylinux** URLs (so PyPI intersection succeeds).  
  - `requirements.hashes.wheel.cpu.aarch64.txt` — same pattern for aarch64.  
  **`Containerfile`** selects one of these via `TARGETARCH` (`amd64` → `x86_64`, `arm64` → `aarch64`).

- **`pylatexenc` on the PyPI wheel file**  
  Same intersection issue: PyPI digest ≠ RHOAI rebuild `*-8-py3-none-any.whl`. After the PyPI-wheel compile, the script rewrites **`pylatexenc==…`** to a **pulp direct URL** (currently the **cuda12.9-ubi9** artifact; the **3.2/cpu-ubi9** pulp path returned 404 for that filename when checked).

- **`pybuild-deps`**  
  It cannot use sdists for **`nvidia-*`**, **`torch` / `torchvision` / `triton`**, **`faiss-cpu`** in this layout. The script feeds a filtered temp file to `pybuild-deps` (see script comments).

- **CPU `requirements.hashes.source.txt` vs transitive CUDA wheels**  
  The resolver can still list **`nvidia-*`** (and related pins) on “PyPI source” lines even when the image installs **CPU** torch from RHOAI. Those packages are **wheel-only / not fetchable as sdists** for Hermeto’s pip input → **`No distributions found`**. **`konflux_requirements.sh`** filters the same set out **before** the `uv pip compile` that writes **`requirements.hashes.source.txt`**, not only before `pybuild-deps`.

- **Tekton / JSON sync**  
  When `prefetch-input` **`requirements_files`** or **`binary.packages`** change, update **`.tekton/rag-tool-*.yaml`**, **`.tekton/lightspeed-core-rag-content-cpu-f176b-*.yaml`**, and **`scripts/hermeto/prefetch-*.json`** together. The konflux script only rewrites the **`"packages"`** string in some Tekton files via `sed`; it does **not** auto-insert new requirement filenames everywhere.

## CUDA pipeline (`scripts/konflux_requirements_cuda.sh`)

- **Regenerate** with `make konflux-requirements-cuda`.

- **Policy**  
  RHOAI **CUDA** `torch` is canonical; do not assume PyPI CUDA `torch`’s `nvidia-*` graph applies. See `README.md` (CUDA / RHOAI / `list_wheel_requires_dist.py`).

- **`hf-xet` (CUDA x86 and hermetic install)**  
  **`hf-xet` 1.3+ / 1.4.x** sdists use **Rust `edition2024`**, which **Cargo ~1.84** in UBI-based build images does not support. If the CUDA wheel requirement file resolves **`hf-xet>=1.2.0`** to **1.4.x** and pip ever uses the **sdist**, metadata/build fails with *“feature `edition2024` is required”*. **Do not try to build it:** pin **`hf-xet==1.2.0`** in **`requirements.overrides.cuda.txt`**, keep **`hf-xet`** in **`PYPI_WHEELS`**, and run the same **force `1.2.0`** step before the PyPI **`--only-binary`** compile as in **`konflux_requirements.sh`** (`requirements.hashes.wheel.pypi.cuda.base.txt` must only carry **1.2.0** wheel hashes). **`huggingface_hub`** remains usable with the older wheel.

- **Never install PyPI `nvidia-*` packages in the CUDA image**  
  Do **not** add **`nvidia-cublas-cu12`**, **`nvidia-cudnn-cu12`**, or any other **`nvidia-*`** wheels from PyPI to hashed requirements or prefetch. RHOAI **`torch`** already ships the CUDA stack it expects; pulling the separate PyPI **`nvidia-*`** graph causes **version skew, duplicate libraries, broken `torch`, and Hermeto/prefetch failures** (`No distributions found` when strict mode disagrees with the lockfile shape). The CUDA generator strips every **`nvidia-*==…`** stanza from **`requirements.hashes.wheel.pypi.cuda.base.txt`** after the PyPI wheel compile (even when `uv` expands PyPI CUDA **`torch`** and injects those lines).

- **CUDA-specific mechanics**  
  - Strip **`torch` / `torchvision` / `triton`** from the **second** PyPI-only compile input so PyPI CUDA torch does not pull **`nvidia-*`** into that file; any **`nvidia-*`** that still appear in the compiled wheel hash file are removed before emitting **`.base.txt` / arch fragments**.  
  - **Sdist-only on PyPI**: pins are moved back to the RHOAI wheel file so `--only-binary :all:` can still run.  
  - **`antlr4-python3-runtime`**: PyPI has no usable wheel for omegaconf’s constraints; inject **pulp URL** + fixed stanza/hash.  
  - **`pylatexenc`**: pulp URL + hash on the appropriate file (same Hermeto intersection issue as CPU).  
  - **`pybuild-deps`**: filtered input excludes wheel-only / problematic packages (see script).  
  - **Wheel layout**: `requirements.hashes.wheel.pypi.cuda.base.txt` plus **`.x86_64.txt` / `.aarch64.txt`** — **not** a single `requirements.hashes.wheel.pypi.cuda.txt` (some older Tekton snippets may still be wrong; **c0ec3** YAMLs are the reference).  
  - **`jiter` / Hermeto `cargo vendor --locked`**: older **`jiter==0.12.x`** sdists shipped a **`Cargo.lock`** out of sync with **`Cargo.toml`**, so prefetch fails with **`PackageWithCorruptLockfileRejected`**. Fix: pin **`jiter==0.13.0`** in **`requirements.overrides.txt`** and **`requirements.overrides.cuda.txt`**, and list **`jiter`** in **`PYPI_WHEELS`** (CPU and CUDA scripts) so resolver emits **manylinux wheels** instead of sdists.

- **rag-tool-cuda Tekton**  
  If prefetch lists the wrong CUDA wheel filenames, fix them to match **`Containerfile-cuda`** and **`lightspeed-core-rag-content-c0ec3-*`**.

## Operational checklist after dependency changes

1. Run **`make konflux-requirements`** and/or **`make konflux-requirements-cuda`**.  
2. Commit generated **`requirements.hashes.*`**, **`requirements-build*.txt`**, new **`requirements.hashes.wheel.cpu.*.txt`** when CPU script emits them, **`Containerfile`** if install paths change, and **`.tekton` / `scripts/hermeto`** if prefetch inputs change.  
3. Run Hermeto pip slice checks per the **Local Hermeto validation** table (**`make hermeto-verify-pip`** when both CPU and CUDA inputs may have moved).  
4. If RHOAI **rebuilds** wheels (new `-*-` segment in filenames), update **pulp URLs / hashes** in the generator constants (and re-run Hermeto).

## Local hermetic container simulation

Konflux mounts prefetched content at **`/cachi2`** and applies Hermeto **`project_files`** (substituting **`file://${output_dir}/…`** paths) before the image build. To approximate that with **`Containerfile`** / **`Containerfile-cuda`**:

1. Run Hermeto (**`make hermeto-verify-pip-cpu`** / **`hermeto-verify-pip-cuda`**, or set **`HERMETO_OUT`**) so outputs land in **`.hermeto-output-verify-*`** or a directory of your choice.  
2. **`./scripts/stage_hermetic_build_context.sh`** — copies **`deps/`** to **`.hermetic-staging/cachi2/output/`**, writes **`cachi2.env`** (**`PIP_FIND_LINKS=/cachi2/output/deps/pip`**), and writes **`.hermetic-staging/patched-requirements/`** from **`.build-config.json`** **`project_files`** with **`/cachi2/output`**.  
3. **`./scripts/simulate_hermetic_build.sh cpu`** (or **`cuda`**) — generates **`.hermetic-staging/Containerfile.sim.*`** (early **`COPY`** of **`cachi2`**, overlay **`COPY`** of patched requirements) and runs **`podman`/`docker` `build`**.  
4. Set **`NETWORK_MODE=none`** only if early **`RUN`** layers are already cached or RPMs/gems are prefetched like on Konflux; **`pip-*`** Hermeto runs do not ship **`deps/generic/model.safetensors`** — use **`full-cpu`**, **`--model`**, **`embeddings_model/`**, or **`ALLOW_PLACEHOLDER_HERMETIC_MODEL=1`** for a build-only stub.  
5. For **CUDA**, point **`HERMETO_OUT`** at a directory produced by **`pip-cuda`** before **`simulate_hermetic_build.sh cuda`**.

## References

- [Hermeto](https://github.com/hermetoproject/hermeto) — prefetch CLI and container image.  
- In-repo: `scripts/konflux_requirements.sh`, `scripts/konflux_requirements_cuda.sh`, `scripts/run_hermeto_fetch_deps.sh`, **`Makefile`** targets **`hermeto-verify-pip-*`**, `scripts/stage_hermetic_build_context.sh`, `scripts/simulate_hermetic_build.sh`, `scripts/hermeto/*.json`, `README.md` (Konflux / CUDA sections).
