"""Build a container image archive from a set of local directories.

Fetches the base image using ``skopeo`` (no daemon, no user-namespace
configuration required) and composes a new Docker-archive tar by stacking a
custom data layer on top of the base image layers entirely in Python.

The image layout is described by the static Containerfile bundled with this
package (``Containerfile.output-image``).  That file is the canonical place
to change what goes into the output image; this module only implements the
privilege-free build mechanism (skopeo + pure Python).

The resulting archive can be imported with::

    podman load < /output/my-rag.tar
    docker load  < /output/my-rag.tar

Requires ``skopeo`` to be installed and accessible on PATH.

Typical usage::

    from lightspeed_rag_content.image_builder import build_image_archive

    build_image_archive(
        vector_db_dir="/output/vector_db",
        output_tar_path="/output/my-rag.tar",
        image_name="my-rag",
        image_tag="v1",
        extra_dirs={"/rag-content/embeddings_model": "/rag/embeddings_model"},
    )
"""

import copy
import gzip
import hashlib
import io
import json
import os
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from typing import Any

DEFAULT_BASE_IMAGE = "registry.access.redhat.com/ubi9/ubi:latest"

# Canonical Containerfile that describes the output image layout.
# Used as a reference specification; the actual build is performed by the
# pure-Python layer composition below (no container build tool required).
_CONTAINERFILE = os.path.join(os.path.dirname(__file__), "Containerfile.output-image")


def _make_layer_tar(
    vector_db_dir: str,
    extra_dirs: dict[str, str] | None,
    dest_file: str,
) -> str:
    """Build the custom image layer tar and write it to *dest_file*.

    Files from *vector_db_dir* are placed under ``/rag/vector_db/`` inside
    the archive.  Any additional source→destination mappings in *extra_dirs*
    are copied using the supplied destination prefix.

    Returns the SHA-256 hex digest of the written tar, computed by streaming
    the file back in chunks so the full layer need not be held in memory.
    """
    with tarfile.open(dest_file, "w") as layer:
        _add_dir_to_tar(
            layer, vector_db_dir, "/rag/vector_db", compress_extensions={".db"}
        )
        if extra_dirs:
            for src, dest in extra_dirs.items():
                if os.path.isdir(src):
                    _add_dir_to_tar(layer, src, dest)
    h = hashlib.sha256()
    with open(dest_file, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _add_dir_to_tar(
    tar: tarfile.TarFile,
    src_dir: str,
    dest_prefix: str,
    compress_extensions: set[str] | None = None,
) -> None:
    """Recursively add *src_dir* into *tar* under *dest_prefix*.

    When *compress_extensions* is provided, files whose suffix is in that set
    are gzip-compressed on the fly and stored with ``.gz`` appended to their
    name; the original uncompressed entry is omitted.
    """
    src_dir = os.path.normpath(src_dir)
    dest_prefix = dest_prefix.lstrip("/")
    for root, dirs, files in os.walk(src_dir):
        dirs.sort()
        rel_root = os.path.relpath(root, src_dir)
        for filename in sorted(files):
            abs_path = os.path.join(root, filename)
            if rel_root == ".":
                arcname = f"{dest_prefix}/{filename}"
            else:
                arcname = f"{dest_prefix}/{rel_root}/{filename}"
            suffix = os.path.splitext(filename)[1]
            if compress_extensions and suffix in compress_extensions:
                data = _gzip_compress(abs_path)
                info = tarfile.TarInfo(name=arcname + ".gz")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            else:
                tar.add(abs_path, arcname=arcname, recursive=False)


def _gzip_compress(src_path: str) -> bytes:
    """Return the gzip-compressed contents of the file at *src_path*."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        with open(src_path, "rb") as f:
            gz.write(f.read())
    return buf.getvalue()


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    """Add raw *data* bytes to *tar* as entry *name*."""
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _fetch_base_archive(base_image: str, dest_tar: str) -> None:
    """Download *base_image* as a docker-archive tar using ``skopeo copy``.

    ``skopeo`` talks directly to the registry API and requires no container
    daemon, user-namespace configuration, or elevated privileges — safe to
    run inside any standard unprivileged container.
    """
    try:
        subprocess.run(
            [
                "/usr/bin/skopeo",
                "copy",
                "--remove-signatures",
                f"docker://{base_image}",
                f"docker-archive:{dest_tar}",
            ],
            check=True,
            timeout=300,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace").strip() if exc.stderr else ""
        raise RuntimeError(f"skopeo failed to fetch {base_image!r}: {stderr}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"skopeo timed out after {exc.timeout}s fetching {base_image!r}"
        ) from exc


def _read_docker_archive(
    tar_path: str,
) -> tuple[list[str], dict[str, Any]]:
    """Parse a docker-archive tar file.

    Returns:
        A tuple ``(layer_names, config)`` where *layer_names* is an ordered
        list of layer entry names from the manifest and *config* is the parsed
        image config dict.  Layer data is *not* loaded into memory.
    """
    with tarfile.open(tar_path) as t:
        manifest_file = t.extractfile("manifest.json")
        if manifest_file is None:
            raise RuntimeError("manifest.json not found in archive")
        manifest = json.loads(manifest_file.read())
        entry = manifest[0]
        config_file = t.extractfile(entry["Config"])
        if config_file is None:
            raise RuntimeError(f"{entry['Config']} not found in archive")
        config = json.loads(config_file.read())
    return entry["Layers"], config


def _build_image_config(
    base_config: dict[str, Any], layer_sha: str
) -> tuple[bytes, str]:
    """Return ``(config_bytes, config_entry)`` for the new image config."""
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    new_config = copy.deepcopy(base_config)
    new_config["created"] = now
    new_config.pop("container", None)
    new_config.setdefault("config", {})["User"] = "65532:65532"
    new_config["rootfs"]["diff_ids"].append(f"sha256:{layer_sha}")
    new_config.setdefault("history", []).append(
        {
            "created": now,
            "created_by": "lightspeed-rag-content image_builder",
        }
    )
    config_bytes = json.dumps(new_config, sort_keys=True).encode()
    config_entry = f"{hashlib.sha256(config_bytes).hexdigest()}.json"
    return config_bytes, config_entry


def _build_manifest(
    base_layer_names: list[str],
    layer_entry: str,
    config_entry: str,
    image_name: str,
    image_tag: str,
) -> bytes:
    """Return serialized ``manifest.json`` bytes for the new image."""
    manifest = [
        {
            "Config": config_entry,
            "RepoTags": [f"{image_name}:{image_tag}"],
            "Layers": [*base_layer_names, layer_entry],
        }
    ]
    return json.dumps(manifest, indent=2).encode()


def build_image_archive(  # pylint: disable=too-many-locals
    vector_db_dir: str,
    output_tar_path: str,
    image_name: str = "rag-content-output",
    image_tag: str = "latest",
    extra_dirs: dict[str, str] | None = None,
    base_image: str = DEFAULT_BASE_IMAGE,
) -> None:
    """Build a container image and save it as a Docker-archive tar file.

    Uses ``skopeo`` to fetch *base_image* as a docker-archive, then composes
    a new archive by stacking a custom data layer (vector DB + optional extras)
    on top of the base image layers entirely in Python.  No container runtime,
    elevated privileges, or user-namespace configuration is required.

    The image layout is defined by ``Containerfile.output-image`` (bundled
    with this package); that file is the canonical reference for what the
    image contains.

    Args:
        vector_db_dir: Directory whose contents are placed at ``/rag/vector_db``
            inside the image.
        output_tar_path: Destination path for the ``.tar`` archive.
        image_name: Repository name for the image (e.g. ``my-rag``).
        image_tag: Tag for the image (e.g. ``v1``).
        extra_dirs: Optional mapping of ``{local_src_path: image_dest_path}``.
            Each existing source directory is added under the given image path.
        base_image: Parent image reference (default: UBI 9 full).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Build the custom layer to a temp file; compute SHA-256 by reading
        #    it back in chunks so it never needs to coexist in RAM with the
        #    base image layers.
        layer_file = os.path.join(tmpdir, "layer.tar")
        layer_sha = _make_layer_tar(vector_db_dir, extra_dirs, layer_file)
        layer_entry = f"{layer_sha}/layer.tar"

        # 2. Fetch the base image as a docker-archive via skopeo.
        base_tar = os.path.join(tmpdir, "base.tar")
        _fetch_base_archive(base_image, base_tar)

        # 3. Parse base archive: layer names and config only (no layer bytes).
        base_layer_names, base_config = _read_docker_archive(base_tar)

        # 4. Build the new image config and manifest.
        config_bytes, config_entry = _build_image_config(base_config, layer_sha)
        manifest_bytes = _build_manifest(
            base_layer_names, layer_entry, config_entry, image_name, image_tag
        )

        # 5. Write the combined docker-archive, streaming layers from disk so
        #    base layers and the custom layer are never all resident in RAM.
        os.makedirs(os.path.dirname(os.path.abspath(output_tar_path)), exist_ok=True)
        with tarfile.open(output_tar_path, "w") as outer:
            with tarfile.open(base_tar) as base_t:
                for name in base_layer_names:
                    member = base_t.getmember(name)
                    fileobj = base_t.extractfile(member)
                    info = tarfile.TarInfo(name=name)
                    info.size = member.size
                    outer.addfile(info, fileobj)
            layer_size = os.path.getsize(layer_file)
            with open(layer_file, "rb") as lf:
                info = tarfile.TarInfo(name=layer_entry)
                info.size = layer_size
                outer.addfile(info, lf)
            _add_bytes(outer, config_entry, config_bytes)
            _add_bytes(outer, "manifest.json", manifest_bytes)
