"""Unit tests for lightspeed_rag_content.image_builder."""

import gzip
import hashlib
import io
import json
import os
import tarfile
from unittest import mock

import pytest

from lightspeed_rag_content.image_builder import (
    DEFAULT_BASE_IMAGE,
    _CONTAINERFILE,
    _fetch_base_archive,
    _make_layer_tar,
    _read_docker_archive,
    build_image_archive,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dir(base: str, structure: dict[str, str]) -> str:
    for rel, content in structure.items():
        path = os.path.join(base, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
    return base


def _make_fake_base_archive(path: str) -> dict:
    """Write a minimal valid docker-archive to *path* and return its config."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as lt:
        info = tarfile.TarInfo(name="base_file.txt")
        data = b"base content"
        info.size = len(data)
        lt.addfile(info, io.BytesIO(data))
    layer_bytes = buf.getvalue()
    sha = hashlib.sha256(layer_bytes).hexdigest()
    layer_name = f"{sha}/layer.tar"

    config = {
        "architecture": "amd64",
        "os": "linux",
        "created": "2024-01-01T00:00:00Z",
        "config": {"User": "0:0"},
        "rootfs": {"type": "layers", "diff_ids": [f"sha256:{sha}"]},
        "history": [{"created": "2024-01-01T00:00:00Z", "created_by": "base"}],
    }
    config_bytes = json.dumps(config, sort_keys=True).encode()
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    config_entry = f"{config_sha}.json"
    manifest = [{"Config": config_entry, "RepoTags": ["base:latest"], "Layers": [layer_name]}]
    manifest_bytes = json.dumps(manifest, indent=2).encode()

    def _add(t, name, data):
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))

    with tarfile.open(path, "w") as t:
        _add(t, layer_name, layer_bytes)
        _add(t, config_entry, config_bytes)
        _add(t, "manifest.json", manifest_bytes)

    return config


def _load_archive(tar_path: str) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    with tarfile.open(tar_path) as outer:
        for member in outer.getmembers():
            if member.isfile():
                f = outer.extractfile(member)
                assert f is not None
                result[member.name] = f.read()
    return result


def _layer_members(layer_bytes: bytes) -> list[str]:
    with tarfile.open(fileobj=io.BytesIO(layer_bytes)) as layer:
        return [m.name for m in layer.getmembers() if m.isfile()]


def _run_build(tmp_path, **kwargs):
    vdb = _make_dir(str(tmp_path / "vdb"), {"index.json": "{}"})
    out = str(tmp_path / "out.tar")

    def fake_fetch(base_image, dest_tar):
        _make_fake_base_archive(dest_tar)

    with mock.patch(
        "lightspeed_rag_content.image_builder._fetch_base_archive",
        side_effect=fake_fetch,
    ):
        build_image_archive(vdb, out, **kwargs)

    return _load_archive(out), out


# ---------------------------------------------------------------------------
# Tests for the static Containerfile
# ---------------------------------------------------------------------------


class TestStaticContainerfile:
    def test_containerfile_exists(self):
        assert os.path.isfile(
            _CONTAINERFILE
        ), f"Containerfile.output-image not found at {_CONTAINERFILE}"

    def test_containerfile_references_ubi9_ubi(self):
        content = open(_CONTAINERFILE).read()
        assert "ubi9/ubi" in content

    def test_containerfile_has_no_run_instruction(self):
        # No RUN instruction — packages come from the base image itself.
        lines = open(_CONTAINERFILE).readlines()
        assert not any(line.startswith("RUN ") for line in lines)

    def test_containerfile_copies_vector_db(self):
        content = open(_CONTAINERFILE).read()
        assert "COPY --from=vector_db" in content
        assert "/rag/vector_db/" in content

    def test_containerfile_copies_embeddings_model(self):
        content = open(_CONTAINERFILE).read()
        assert "COPY --from=embeddings_model" in content
        assert "/rag/embeddings_model/" in content

    def test_containerfile_sets_non_root_user(self):
        content = open(_CONTAINERFILE).read()
        assert "USER 65532" in content

    def test_default_base_image_is_ubi9_full(self):
        assert "ubi9/ubi" in DEFAULT_BASE_IMAGE
        assert "ubi-minimal" not in DEFAULT_BASE_IMAGE


# ---------------------------------------------------------------------------
# Tests for _make_layer_tar
# ---------------------------------------------------------------------------


class TestMakeLayerTar:
    def test_contains_vector_db_files(self, tmp_path):
        vdb = _make_dir(str(tmp_path / "vdb"), {"a.txt": "hello", "sub/b.txt": "world"})
        layer_file = str(tmp_path / "layer.tar")
        _make_layer_tar(vdb, None, layer_file)
        with open(layer_file, "rb") as f:
            names = _layer_members(f.read())
        assert any("rag/vector_db/a.txt" in n for n in names)
        assert any("rag/vector_db/sub/b.txt" in n for n in names)

    def test_extra_dirs_included(self, tmp_path):
        vdb = _make_dir(str(tmp_path / "vdb"), {"idx.json": "{}"})
        model = _make_dir(str(tmp_path / "model"), {"weights.bin": "data"})
        layer_file = str(tmp_path / "layer.tar")
        _make_layer_tar(vdb, {model: "/rag/embeddings_model"}, layer_file)
        with open(layer_file, "rb") as f:
            names = _layer_members(f.read())
        assert any("rag/embeddings_model/weights.bin" in n for n in names)

    def test_nonexistent_extra_dir_skipped(self, tmp_path):
        vdb = _make_dir(str(tmp_path / "vdb"), {"idx.json": "{}"})
        layer_file = str(tmp_path / "layer.tar")
        sha = _make_layer_tar(vdb, {"/nonexistent": "/rag/model"}, layer_file)
        assert os.path.isfile(layer_file)
        assert isinstance(sha, str) and len(sha) == 64

    def test_db_file_added_as_gz(self, tmp_path):
        vdb = _make_dir(str(tmp_path / "vdb"), {"faiss_store.db": "fake-db-content"})
        layer_file = str(tmp_path / "layer.tar")
        _make_layer_tar(vdb, None, layer_file)
        with open(layer_file, "rb") as f:
            names = _layer_members(f.read())
        assert any("rag/vector_db/faiss_store.db.gz" in n for n in names)

    def test_db_file_not_added_uncompressed(self, tmp_path):
        vdb = _make_dir(str(tmp_path / "vdb"), {"faiss_store.db": "fake-db-content"})
        layer_file = str(tmp_path / "layer.tar")
        _make_layer_tar(vdb, None, layer_file)
        with open(layer_file, "rb") as f:
            names = _layer_members(f.read())
        assert not any(n.endswith("faiss_store.db") for n in names)

    def test_db_gz_content_decompresses_correctly(self, tmp_path):
        original = b"fake-db-content"
        vdb = _make_dir(str(tmp_path / "vdb"), {"faiss_store.db": original.decode()})
        layer_file = str(tmp_path / "layer.tar")
        _make_layer_tar(vdb, None, layer_file)
        with tarfile.open(layer_file) as layer:
            gz_member = next(m for m in layer.getmembers() if m.name.endswith("faiss_store.db.gz"))
            gz_data = layer.extractfile(gz_member).read()
        assert gzip.decompress(gz_data) == original

    def test_returns_sha256_hex(self, tmp_path):
        vdb = _make_dir(str(tmp_path / "vdb"), {"idx.json": "{}"})
        layer_file = str(tmp_path / "layer.tar")
        sha = _make_layer_tar(vdb, None, layer_file)
        assert len(sha) == 64
        with open(layer_file, "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == sha


# ---------------------------------------------------------------------------
# Tests for _fetch_base_archive
# ---------------------------------------------------------------------------


class TestFetchBaseArchive:
    def test_calls_skopeo_copy(self, tmp_path):
        out = str(tmp_path / "base.tar")
        with mock.patch("lightspeed_rag_content.image_builder.subprocess.run") as mock_run:
            _fetch_base_archive("registry.example.com/img:tag", out)
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/usr/bin/skopeo" and cmd[1] == "copy"

    def test_uses_docker_transport_prefix(self, tmp_path):
        out = str(tmp_path / "base.tar")
        with mock.patch("lightspeed_rag_content.image_builder.subprocess.run") as mock_run:
            _fetch_base_archive("myregistry/myimage:v1", out)
        cmd = mock_run.call_args[0][0]
        assert "docker://myregistry/myimage:v1" in cmd

    def test_uses_docker_archive_destination(self, tmp_path):
        out = str(tmp_path / "base.tar")
        with mock.patch("lightspeed_rag_content.image_builder.subprocess.run") as mock_run:
            _fetch_base_archive("myregistry/myimage:v1", out)
        cmd = mock_run.call_args[0][0]
        assert any(arg.startswith("docker-archive:") for arg in cmd)


# ---------------------------------------------------------------------------
# Tests for _read_docker_archive
# ---------------------------------------------------------------------------


class TestReadDockerArchive:
    def test_returns_layer_names_and_config(self, tmp_path):
        archive = str(tmp_path / "base.tar")
        _make_fake_base_archive(archive)
        layer_names, config = _read_docker_archive(archive)
        assert len(layer_names) == 1
        assert config["architecture"] == "amd64"

    def test_layer_name_ends_with_layer_tar(self, tmp_path):
        archive = str(tmp_path / "base.tar")
        _make_fake_base_archive(archive)
        layer_names, _ = _read_docker_archive(archive)
        assert layer_names[0].endswith("/layer.tar")

    def test_layer_data_not_loaded(self, tmp_path):
        # _read_docker_archive returns names only; verify no bytes are returned.
        archive = str(tmp_path / "base.tar")
        _make_fake_base_archive(archive)
        layer_names, _ = _read_docker_archive(archive)
        assert all(isinstance(name, str) for name in layer_names)


# ---------------------------------------------------------------------------
# Tests for build_image_archive
# ---------------------------------------------------------------------------


class TestBuildImageArchive:
    def test_creates_tar_file(self, tmp_path):
        _, out = _run_build(tmp_path)
        assert os.path.isfile(out) and os.path.getsize(out) > 0

    def test_manifest_has_correct_repo_tag(self, tmp_path):
        members, _ = _run_build(tmp_path, image_name="myimg", image_tag="v2")
        manifest = json.loads(members["manifest.json"])
        assert manifest[0]["RepoTags"] == ["myimg:v2"]

    def test_manifest_references_existing_entries(self, tmp_path):
        members, _ = _run_build(tmp_path)
        entry = json.loads(members["manifest.json"])[0]
        assert entry["Config"] in members
        for layer in entry["Layers"]:
            assert layer in members

    def test_base_layers_preserved_and_custom_layer_appended(self, tmp_path):
        members, _ = _run_build(tmp_path)
        manifest = json.loads(members["manifest.json"])
        assert len(manifest[0]["Layers"]) == 2  # 1 base + 1 custom

    def test_config_extends_base_diff_ids(self, tmp_path):
        members, _ = _run_build(tmp_path)
        manifest = json.loads(members["manifest.json"])
        config = json.loads(members[manifest[0]["Config"]])
        assert len(config["rootfs"]["diff_ids"]) == 2

    def test_config_sets_user(self, tmp_path):
        members, _ = _run_build(tmp_path)
        manifest = json.loads(members["manifest.json"])
        config = json.loads(members[manifest[0]["Config"]])
        assert config["config"]["User"] == "65532:65532"

    def test_custom_layer_contains_vector_db_files(self, tmp_path):
        vdb = _make_dir(str(tmp_path / "vdb"), {"store.json": "{}", "sub/idx.faiss": "data"})
        out = str(tmp_path / "out.tar")

        def fake_fetch(base_image, dest_tar):
            _make_fake_base_archive(dest_tar)

        with mock.patch(
            "lightspeed_rag_content.image_builder._fetch_base_archive",
            side_effect=fake_fetch,
        ):
            build_image_archive(vdb, out)

        members = _load_archive(out)
        custom_layer = json.loads(members["manifest.json"])[0]["Layers"][-1]
        layer_files = _layer_members(members[custom_layer])
        assert any("rag/vector_db/store.json" in f for f in layer_files)
        assert any("rag/vector_db/sub/idx.faiss" in f for f in layer_files)

    def test_extra_dirs_included_in_custom_layer(self, tmp_path):
        vdb = _make_dir(str(tmp_path / "vdb"), {"index.json": "{}"})
        model = _make_dir(str(tmp_path / "model"), {"model.safetensors": "weights"})
        out = str(tmp_path / "out.tar")

        def fake_fetch(base_image, dest_tar):
            _make_fake_base_archive(dest_tar)

        with mock.patch(
            "lightspeed_rag_content.image_builder._fetch_base_archive",
            side_effect=fake_fetch,
        ):
            build_image_archive(vdb, out, extra_dirs={model: "/rag/embeddings_model"})

        members = _load_archive(out)
        custom_layer = json.loads(members["manifest.json"])[0]["Layers"][-1]
        assert any(
            "rag/embeddings_model/model.safetensors" in f
            for f in _layer_members(members[custom_layer])
        )

    def test_custom_diff_id_matches_layer_sha(self, tmp_path):
        vdb = _make_dir(str(tmp_path / "vdb"), {"index.json": "{}"})
        out = str(tmp_path / "out.tar")

        def fake_fetch(base_image, dest_tar):
            _make_fake_base_archive(dest_tar)

        with mock.patch(
            "lightspeed_rag_content.image_builder._fetch_base_archive",
            side_effect=fake_fetch,
        ):
            build_image_archive(vdb, out)

        members = _load_archive(out)
        manifest = json.loads(members["manifest.json"])[0]
        config = json.loads(members[manifest["Config"]])
        custom_layer_name = manifest["Layers"][-1]
        custom_sha = custom_layer_name.split("/")[0]
        assert f"sha256:{custom_sha}" in config["rootfs"]["diff_ids"]

    def test_output_dir_created_if_missing(self, tmp_path):
        vdb = _make_dir(str(tmp_path / "vdb"), {"index.json": "{}"})
        out = str(tmp_path / "nested" / "deep" / "out.tar")

        def fake_fetch(base_image, dest_tar):
            _make_fake_base_archive(dest_tar)

        with mock.patch(
            "lightspeed_rag_content.image_builder._fetch_base_archive",
            side_effect=fake_fetch,
        ):
            build_image_archive(vdb, out)

        assert os.path.isfile(out)

    def test_default_repo_tag(self, tmp_path):
        members, _ = _run_build(tmp_path)
        manifest = json.loads(members["manifest.json"])
        assert manifest[0]["RepoTags"] == ["rag-content-output:latest"]
