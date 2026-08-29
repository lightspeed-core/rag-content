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

"""Tests for DocumentProcessor sqlite-faiss wiring."""

import hashlib
import json
import os
import sqlite3

import numpy as np
import pytest
from llama_index.core.schema import Document, TextNode

from lightspeed_rag_content import document_processor
from lightspeed_rag_content.sqlite_faiss import (
    list_sqlite_faiss_vector_store_ids,
    search_sqlite_faiss_store,
)
from tests.conftest import RagMockEmbedding


def test_document_processor_routes_sqlite_faiss(mocker) -> None:
    """DocumentProcessor uses _SqliteFaissDB for vector_store_type sqlite-faiss."""
    mocker.patch.object(document_processor, "HuggingFaceEmbedding", new=RagMockEmbedding)
    sqlite_db = mocker.patch.object(document_processor, "_SqliteFaissDB")
    mocker.patch.object(document_processor, "_LlamaStackDB")
    mocker.patch.object(document_processor, "_LlamaIndexDB")

    processor = document_processor.DocumentProcessor(
        chunk_size=380,
        chunk_overlap=0,
        model_name="sentence-transformers/all-mpnet-base-v2",
        embeddings_model_dir="embeddings_model",
        vector_store_type="sqlite-faiss",
    )
    sqlite_db.assert_called_once_with(processor.config)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)


def test_sqlite_faiss_db_save_writes_store(mocker, tmp_path) -> None:
    """_SqliteFaissDB.save writes faiss_store.db and lightspeed-stack.yaml."""
    mocker.patch.object(document_processor, "HuggingFaceEmbedding", new=RagMockEmbedding)
    st = mocker.patch("lightspeed_rag_content.sqlite_faiss.SentenceTransformer")
    st.return_value.get_sentence_embedding_dimension.return_value = 8
    st.return_value.encode.return_value = np.eye(2, 8, dtype=np.float32)

    config = document_processor._Config(
        chunk_size=380,
        chunk_overlap=0,
        model_name="sentence-transformers/all-mpnet-base-v2",
        embeddings_model_dir="",
        vector_store_type="sqlite-faiss",
        embedding_dimension=None,
        manual_chunking=True,
        doc_type="text",
        show_progress=False,
    )
    db = document_processor._SqliteFaissDB(config)
    mocker.patch.object(
        db,
        "_split_and_filter",
        return_value=[
            TextNode(
                text="alpha chunk about penguins",
                id_="chunk-a",
                ref_doc_id="doc-a",
                metadata={
                    "title": "Penguins",
                    "docs_url": "https://example.com/penguins",
                    "filename": "penguins.md",
                },
            ),
            TextNode(
                text="beta chunk about zebras",
                id_="chunk-b",
                ref_doc_id="doc-b",
                metadata={
                    "title": "Zebras",
                    "docs_url": "https://example.com/zebras",
                    "filename": "zebras.md",
                },
            ),
        ],
    )
    db.add_docs([Document(text="unused", metadata={})])
    vector_store_id = db.save("ocp-docs", str(tmp_path), embedded_files=1, exec_time=1)

    db_file = tmp_path / "faiss_store.db"
    assert db_file.is_file()
    keys = {row[0] for row in sqlite3.connect(db_file).execute("SELECT key FROM kvstore")}
    assert f"vector_io::faiss:faiss_index:v3::{vector_store_id}" in keys

    hits = search_sqlite_faiss_store(
        db_file, vector_store_id, np.eye(2, 8, dtype=np.float32)[0].tolist(), k=1
    )
    assert hits[0]["chunk_id"] == "chunk-a"

    lcs_yaml = (tmp_path / "lightspeed-stack.yaml").read_text(encoding="utf-8")
    assert f"vector_db_id: {vector_store_id}" in lcs_yaml
    assert "backend: faiss" in lcs_yaml
    assert not (tmp_path / "llama-stack.yaml").exists()


def _fake_sentence_transformer(dim: int = 8):
    """Deterministic embedder shared by OGX and sqlite-faiss."""

    class FakeSentenceTransformer:
        """Hash-based unit-ish embeddings so both writers see the same vectors."""

        def __init__(self, *args, **kwargs):
            pass

        def get_sentence_embedding_dimension(self) -> int:
            return dim

        def encode(self, sentences, **kwargs):
            if isinstance(sentences, str):
                sentences = [sentences]
            rows = []
            for text in sentences:
                digest = hashlib.sha256(text.encode("utf-8")).digest()
                vec = np.zeros(dim, dtype=np.float32)
                for i in range(dim):
                    vec[i] = digest[i] / 255.0
                rows.append(vec)
            return np.stack(rows)

    return FakeSentenceTransformer


def _sample_nodes() -> list[TextNode]:
    return [
        TextNode(
            text="alpha chunk about penguins",
            id_="chunk-a",
            ref_doc_id="doc-a",
            metadata={
                "title": "Penguins",
                "docs_url": "https://example.com/penguins",
                "filename": "penguins.md",
            },
        ),
        TextNode(
            text="beta chunk about zebras",
            id_="chunk-b",
            ref_doc_id="doc-b",
            metadata={
                "title": "Zebras",
                "docs_url": "https://example.com/zebras",
                "filename": "zebras.md",
            },
        ),
    ]


def _kv_key_kinds(db_path) -> set[str]:
    keys = {row[0] for row in sqlite3.connect(db_path).execute("SELECT key FROM kvstore")}
    kinds: set[str] = set()
    for key in keys:
        if ":v3::" not in key:
            kinds.add(key)
            continue
        prefix, _rest = key.split(":v3::", 1)
        kinds.add(prefix.rsplit(":", 1)[-1] if ":" in prefix else prefix)
    return kinds


def _chunks_by_id(db_path) -> dict[str, dict]:
    vs_id = list_sqlite_faiss_vector_store_ids(db_path)[0]
    raw = (
        sqlite3.connect(db_path)
        .execute(
            "SELECT value FROM kvstore WHERE key=?",
            (f"vector_io::faiss:faiss_index:v3::{vs_id}",),
        )
        .fetchone()[0]
    )
    payload = json.loads(raw)
    chunks: dict[str, dict] = {}
    for encoded in payload["chunk_by_index"].values():
        chunk = json.loads(encoded) if isinstance(encoded, str) else encoded
        chunks[chunk["chunk_id"]] = chunk
    return chunks


def _drop_nones(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if value is not None}


def _canonical_chunk(chunk: dict) -> dict:
    metadata = _drop_nones(dict(chunk.get("metadata") or {}))
    chunk_metadata = _drop_nones(dict(chunk.get("chunk_metadata") or {}))
    metadata["document_id"] = "<file-id>"
    chunk_metadata["document_id"] = "<file-id>"
    return {
        "content": chunk["content"],
        "chunk_id": chunk["chunk_id"],
        "metadata": metadata,
        "chunk_metadata": chunk_metadata,
        "embedding_model": chunk.get("embedding_model"),
        "embedding_dimension": chunk.get("embedding_dimension"),
        "embedding": chunk["embedding"],
    }


def test_llamastack_faiss_and_sqlite_faiss_equivalent_stores(mocker, tmp_path) -> None:
    """llamastack-faiss (OGX) and sqlite-faiss persist equivalent KV stores from the same chunks."""
    pytest.importorskip("ogx")
    fake_cls = _fake_sentence_transformer()
    mocker.patch.object(document_processor, "HuggingFaceEmbedding", new=RagMockEmbedding)
    mocker.patch.object(document_processor, "SentenceTransformer", fake_cls)
    mocker.patch("lightspeed_rag_content.sqlite_faiss.SentenceTransformer", fake_cls)
    mocker.patch("sentence_transformers.SentenceTransformer", fake_cls)

    docs = [Document(text="unused", metadata={})]
    index_name = "ocp-docs"

    llama_config = document_processor._Config(
        chunk_size=380,
        chunk_overlap=0,
        model_name="sentence-transformers/all-mpnet-base-v2",
        embeddings_model_dir="",
        vector_store_type="llamastack-faiss",
        embedding_dimension=None,
        manual_chunking=True,
        doc_type="text",
        show_progress=False,
    )
    sqlite_config = document_processor._Config(
        chunk_size=380,
        chunk_overlap=0,
        model_name="sentence-transformers/all-mpnet-base-v2",
        embeddings_model_dir="",
        vector_store_type="sqlite-faiss",
        embedding_dimension=None,
        manual_chunking=True,
        doc_type="text",
        show_progress=False,
    )

    llama_db = document_processor._LlamaStackDB(llama_config)
    mocker.patch.object(llama_db, "_split_and_filter", return_value=_sample_nodes())
    llama_db.add_docs(docs)
    llama_out = tmp_path / "llamastack"
    llama_db.save(index_name, str(llama_out), embedded_files=1, exec_time=1)

    sqlite_db = document_processor._SqliteFaissDB(sqlite_config)
    mocker.patch.object(sqlite_db, "_split_and_filter", return_value=_sample_nodes())
    sqlite_db.add_docs(docs)
    sqlite_out = tmp_path / "sqlite_faiss"
    sqlite_db.save(index_name, str(sqlite_out), embedded_files=1, exec_time=1)

    llama_store = llama_out / "faiss_store.db"
    sqlite_store = sqlite_out / "faiss_store.db"
    assert llama_store.is_file()
    assert sqlite_store.is_file()

    llama_kinds = _kv_key_kinds(llama_store)
    ogx_kinds = _kv_key_kinds(sqlite_store)
    # OGX stores placeholder file rows on the files provider (files_metadata.db),
    # not in faiss_store.db. sqlite-faiss inlines those keys in the same SQLite file.
    assert {"faiss_index", "vector_stores", "openai_vector_stores"} <= llama_kinds
    assert llama_kinds <= ogx_kinds

    llama_chunks = _chunks_by_id(llama_store)
    ogx_chunks = _chunks_by_id(sqlite_store)
    assert set(llama_chunks) == set(ogx_chunks) == {"chunk-a", "chunk-b"}

    for chunk_id in llama_chunks:
        left = _canonical_chunk(llama_chunks[chunk_id])
        right = _canonical_chunk(ogx_chunks[chunk_id])
        assert left["content"] == right["content"]
        assert left["metadata"] == right["metadata"]
        assert left["chunk_metadata"] == right["chunk_metadata"]
        assert left["embedding_model"] == right["embedding_model"]
        assert left["embedding_dimension"] == right["embedding_dimension"]
        np.testing.assert_allclose(left["embedding"], right["embedding"], rtol=1e-5, atol=1e-5)

    llama_vs = list_sqlite_faiss_vector_store_ids(llama_store)[0]
    sqlite_vs = list_sqlite_faiss_vector_store_ids(sqlite_store)[0]
    for chunk in llama_chunks.values():
        llama_hits = search_sqlite_faiss_store(llama_store, llama_vs, chunk["embedding"], k=2)
        sqlite_hits = search_sqlite_faiss_store(sqlite_store, sqlite_vs, chunk["embedding"], k=2)
        assert [hit["chunk_id"] for hit in llama_hits] == [hit["chunk_id"] for hit in sqlite_hits]
