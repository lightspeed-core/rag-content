# Copyright 2025 Red Hat, Inc.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

"""Tests for the sqlite-faiss KV writer."""

import inspect
import json
import sqlite3

import numpy as np
import pytest

from lightspeed_rag_content.sqlite_faiss import (
    SqliteFaissChunk,
    list_sqlite_faiss_vector_store_ids,
    search_sqlite_faiss_store,
    write_sqlite_faiss_store,
)


def _unit_embedding(index: int, dim: int = 8) -> list[float]:
    vec = np.zeros(dim, dtype=np.float32)
    vec[index % dim] = 1.0
    return vec.tolist()


def _sample_chunks() -> list[SqliteFaissChunk]:
    return [
        SqliteFaissChunk(
            content="alpha chunk about penguins",
            chunk_id="chunk-a",
            metadata={
                "title": "Penguins",
                "docs_url": "https://example.com/penguins",
                "document_id": "doc-a",
                "filename": "penguins.md",
            },
            chunk_metadata={
                "document_id": "doc-a",
                "chunk_id": "chunk-a",
                "source": "https://example.com/penguins",
            },
            embedding=_unit_embedding(0),
        ),
        SqliteFaissChunk(
            content="beta chunk about zebras",
            chunk_id="chunk-b",
            metadata={
                "title": "Zebras",
                "docs_url": "https://example.com/zebras",
                "document_id": "doc-b",
                "filename": "zebras.md",
            },
            chunk_metadata={
                "document_id": "doc-b",
                "chunk_id": "chunk-b",
                "source": "https://example.com/zebras",
            },
            embedding=_unit_embedding(1),
        ),
    ]


def test_sqlite_faiss_module_does_not_import_ogx() -> None:
    """The writer must not depend on ogx or llama_stack packages."""
    from lightspeed_rag_content import sqlite_faiss

    source = inspect.getsource(sqlite_faiss)
    assert "import ogx" not in source
    assert "from ogx" not in source
    assert "llama_stack" not in source
    assert "ogx_client" not in source
    assert "ogx_api" not in source


def test_write_sqlite_faiss_store_creates_namespaced_v3_keys(tmp_path) -> None:
    """Written SQLite uses OGX 1.0 vector_io::faiss v3 key names."""
    db_path = tmp_path / "faiss_store.db"
    vs_id = "vs_11111111-2222-3333-4444-555555555555"
    write_sqlite_faiss_store(
        db_path,
        vector_store_id=vs_id,
        provider_id="ocp-docs",
        store_name="ocp-docs",
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        embedding_dimension=8,
        chunks=_sample_chunks(),
    )

    con = sqlite3.connect(db_path)
    keys = {row[0] for row in con.execute("SELECT key FROM kvstore")}
    assert f"vector_io::faiss:faiss_index:v3::{vs_id}" in keys
    assert f"vector_io::faiss:vector_stores:v3::{vs_id}" in keys
    assert f"vector_io::faiss:openai_vector_stores:v3::{vs_id}" in keys
    assert any(
        k.startswith(f"vector_io::faiss:openai_vector_stores_files:v3::{vs_id}:") for k in keys
    )
    assert any(
        k.startswith(f"vector_io::faiss:openai_vector_stores_files_contents:v3::{vs_id}:")
        for k in keys
    )
    assert not any(k.startswith("faiss_index:v3::") for k in keys)


def test_write_sqlite_faiss_store_roundtrip_search(tmp_path) -> None:
    """Serialized FAISS index plus chunk_by_index round-trip."""
    db_path = tmp_path / "faiss_store.db"
    vs_id = "vs_aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    chunks = _sample_chunks()
    write_sqlite_faiss_store(
        db_path,
        vector_store_id=vs_id,
        provider_id="ocp-docs",
        store_name="ocp-docs",
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        embedding_dimension=8,
        chunks=chunks,
    )

    hits = search_sqlite_faiss_store(db_path, vs_id, chunks[0].embedding, k=1)
    assert len(hits) == 1
    hit = hits[0]
    assert hit["chunk_id"] == "chunk-a"
    assert "penguins" in hit["content"]
    assert hit["score"] == pytest.approx(0.0, abs=1e-5)


def test_faiss_index_value_is_double_encoded_chunk_map(tmp_path) -> None:
    """chunk_by_index values are JSON strings, matching llamastack-faiss."""
    db_path = tmp_path / "faiss_store.db"
    vs_id = "vs_00000000-0000-0000-0000-000000000001"
    write_sqlite_faiss_store(
        db_path,
        vector_store_id=vs_id,
        provider_id="idx",
        store_name="idx",
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        embedding_dimension=8,
        chunks=_sample_chunks(),
    )
    raw = (
        sqlite3.connect(db_path)
        .execute(
            "SELECT value FROM kvstore WHERE key=?",
            (f"vector_io::faiss:faiss_index:v3::{vs_id}",),
        )
        .fetchone()[0]
    )
    payload = json.loads(raw)
    assert set(payload.keys()) == {"chunk_by_index", "faiss_index"}
    assert isinstance(payload["faiss_index"], str)
    chunk0 = json.loads(payload["chunk_by_index"]["0"])
    assert chunk0["content"] == "alpha chunk about penguins"
    assert chunk0["embedding_dimension"] == 8
    assert len(chunk0["embedding"]) == 8


def test_list_sqlite_faiss_vector_store_ids(tmp_path) -> None:
    """Helper lists vector_store_ids from faiss_index keys."""
    db_path = tmp_path / "faiss_store.db"
    vs_id = "vs_listed-store-id"
    write_sqlite_faiss_store(
        db_path,
        vector_store_id=vs_id,
        provider_id="idx",
        store_name="idx",
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        embedding_dimension=8,
        chunks=_sample_chunks(),
    )
    assert list_sqlite_faiss_vector_store_ids(db_path) == [vs_id]
