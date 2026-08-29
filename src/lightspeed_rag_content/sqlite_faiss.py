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
"""Write and read FAISS indexes stored in a SQLite kvstore table.

The on-disk layout matches rag-content's historical ``llamastack-faiss``
output: a SQLite ``kvstore`` table whose keys are prefixed with
``vector_io::faiss:`` and versioned ``v3``. Existing Lightspeed Core Stack
BYOK files keep working; this module does not import ``ogx``.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from lightspeed_rag_content import config_templates

if TYPE_CHECKING:
    from collections.abc import Sequence

    from llama_index.core.schema import Document

LOG = logging.getLogger(__name__)

KV_NAMESPACE: Final[str] = "vector_io::faiss"
KV_VERSION: Final[str] = "v3"
KVSTORE_DDL: Final[str] = """
CREATE TABLE IF NOT EXISTS kvstore (
    key TEXT PRIMARY KEY,
    value TEXT,
    expiration TIMESTAMP
)
"""


@dataclass(frozen=True)
class SqliteFaissChunk:
    """One chunk plus its embedding, ready to persist.

    Attributes:
        content: Chunk text stored for retrieval.
        chunk_id: Stable chunk identifier.
        metadata: Citation metadata (title, docs_url, document_id, ...).
        chunk_metadata: OGX-shaped chunk_metadata dict.
        embedding: Embedding vector; length must match embedding_dimension.
    """

    content: str
    chunk_id: str
    metadata: dict[str, Any]
    chunk_metadata: dict[str, Any]
    embedding: list[float]


def _kv_key(kind: str, vector_store_id: str, *rest: str) -> str:
    """Build an OGX 1.0 namespaced KV key.

    Store-level keys are ``{namespace}:{kind}:{version}::{vector_store_id}``.
    Nested keys join extra parts with a single colon
    (``...::{vector_store_id}:{file_id}:{idx}``).
    """
    tail = ":".join((vector_store_id, *rest)) if rest else vector_store_id
    return f"{KV_NAMESPACE}:{kind}:{KV_VERSION}::{tail}"


def _serialize_faiss_index(index: faiss.Index) -> str:
    """Encode a FAISS index the way OGX stores it (base64 of an npy uint8 blob)."""
    serialized = faiss.serialize_index(index)
    buf = io.BytesIO()
    np.save(buf, serialized, allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _deserialize_faiss_index(payload: str) -> faiss.Index:
    """Decode an OGX-stored FAISS index blob."""
    raw = base64.b64decode(payload)
    arr = np.load(io.BytesIO(raw), allow_pickle=False)
    return faiss.deserialize_index(arr)


def _put(connection: sqlite3.Connection, key: str, value: dict[str, Any] | str) -> None:
    """Insert or replace a KV row."""
    encoded = value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))
    connection.execute(
        "INSERT OR REPLACE INTO kvstore (key, value, expiration) VALUES (?, ?, NULL)",
        (key, encoded),
    )


def write_sqlite_faiss_store(  # pylint: disable=too-many-arguments,too-many-locals
    db_path: str | Path,
    *,
    vector_store_id: str,
    provider_id: str,
    store_name: str,
    embedding_model: str,
    embedding_dimension: int,
    chunks: Sequence[SqliteFaissChunk],
    chunk_size: int = 380,
    chunk_overlap: int = 0,
    created_at: int | None = None,
) -> None:
    """Write a llamastack-faiss-compatible SQLite file.

    Parameters:
        db_path: Destination SQLite path (created or replaced).
        vector_store_id: Store id used in KV keys (``vs_<uuid>``).
        provider_id: Value stored as ``provider_id`` on the vector_store row.
        store_name: Human-readable store name (OpenAI vector_store.name).
        embedding_model: Embedding model id recorded on each chunk.
        embedding_dimension: Dimensionality of ``chunks[].embedding``.
        chunks: Chunks in FAISS row order.
        chunk_size: Recorded static chunking max tokens.
        chunk_overlap: Recorded static chunking overlap tokens.
        created_at: Unix timestamp; defaults to now.

    Raises:
        ValueError: If ``chunks`` is empty or an embedding has the wrong length.
    """
    if not chunks:
        raise ValueError("chunks must not be empty")
    for chunk in chunks:
        if len(chunk.embedding) != embedding_dimension:
            raise ValueError(
                f"chunk {chunk.chunk_id!r} embedding length "
                f"{len(chunk.embedding)} != {embedding_dimension}"
            )

    timestamp = int(time.time()) if created_at is None else created_at
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = np.asarray([chunk.embedding for chunk in chunks], dtype=np.float32)
    index = faiss.IndexFlatL2(embedding_dimension)
    index.add(matrix)  # pylint: disable=no-value-for-parameter

    chunk_by_index: dict[str, str] = {}
    file_ids_by_doc: dict[str, str] = {}
    contents_by_file: dict[str, list[tuple[int, dict[str, Any]]]] = {}

    for row, chunk in enumerate(chunks):
        doc_id = str(
            chunk.metadata.get("document_id")
            or chunk.chunk_metadata.get("document_id")
            or chunk.chunk_id
        )
        file_id = file_ids_by_doc.setdefault(doc_id, f"file-{uuid.uuid4().hex}")
        metadata = {**chunk.metadata, "document_id": file_id}
        chunk_metadata = {**chunk.chunk_metadata, "document_id": file_id}
        record = {
            "content": chunk.content,
            "chunk_id": chunk.chunk_id,
            "metadata": metadata,
            "chunk_metadata": chunk_metadata,
            "embedding": chunk.embedding,
            "embedding_model": embedding_model,
            "embedding_dimension": embedding_dimension,
        }
        chunk_by_index[str(row)] = json.dumps(record, separators=(",", ":"))
        contents_by_file.setdefault(file_id, []).append((row, record))

    file_ids = list(file_ids_by_doc.values())
    faiss_payload = {
        "chunk_by_index": chunk_by_index,
        "faiss_index": _serialize_faiss_index(index),
    }

    connection = sqlite3.connect(db_path)
    try:
        connection.execute(KVSTORE_DDL)
        _put(connection, _kv_key("faiss_index", vector_store_id), faiss_payload)
        _put(
            connection,
            _kv_key("vector_stores", vector_store_id),
            {
                "identifier": vector_store_id,
                "provider_resource_id": None,
                "provider_id": provider_id,
                "type": "vector_store",
                "owner": None,
                "source": "via_register_api",
                "embedding_model": embedding_model,
                "embedding_dimension": embedding_dimension,
                "vector_store_name": None,
            },
        )
        _put(
            connection,
            _kv_key("openai_vector_stores", vector_store_id),
            {
                "id": vector_store_id,
                "object": "vector_store",
                "created_at": timestamp,
                "name": store_name,
                "usage_bytes": 0,
                "file_counts": {
                    "completed": len(file_ids),
                    "cancelled": 0,
                    "failed": 0,
                    "in_progress": 0,
                    "total": len(file_ids),
                },
                "status": "completed",
                "expires_after": None,
                "expires_at": None,
                "last_active_at": timestamp,
                "file_ids": file_ids,
                "chunking_strategy": {
                    "type": "static",
                    "static": {
                        "chunk_overlap_tokens": chunk_overlap,
                        "max_chunk_size_tokens": chunk_size,
                    },
                },
                "metadata": {
                    "provider_id": provider_id,
                    "provider_vector_store_id": vector_store_id,
                    "embedding_model": embedding_model,
                    "embedding_dimension": str(embedding_dimension),
                },
            },
        )
        for doc_id, file_id in file_ids_by_doc.items():
            first_record = contents_by_file[file_id][0][1]
            filename = str(first_record["metadata"].get("filename") or f"{doc_id}.txt")
            _put(
                connection,
                _kv_key("openai_vector_stores_files", vector_store_id, file_id),
                {
                    "id": file_id,
                    "object": "vector_store.file",
                    "attributes": {},
                    "chunking_strategy": {
                        "type": "static",
                        "static": {
                            "chunk_overlap_tokens": chunk_overlap,
                            "max_chunk_size_tokens": chunk_size,
                        },
                    },
                    "created_at": timestamp,
                    "status": "completed",
                    "usage_bytes": 0,
                    "vector_store_id": vector_store_id,
                    "filename": filename,
                },
            )
            for local_idx, (_row, record) in enumerate(contents_by_file[file_id]):
                _put(
                    connection,
                    _kv_key(
                        "openai_vector_stores_files_contents",
                        vector_store_id,
                        file_id,
                        str(local_idx),
                    ),
                    record,
                )
        connection.commit()
    finally:
        connection.close()


def search_sqlite_faiss_store(
    db_path: str | Path,
    vector_store_id: str,
    query_embedding: Sequence[float],
    k: int = 1,
) -> list[dict[str, Any]]:
    """Search an sqlite-faiss / llamastack-faiss SQLite file.

    Parameters:
        db_path: Path to the KV SQLite file.
        vector_store_id: Store id used when the file was written.
        query_embedding: Query vector.
        k: Number of nearest neighbors.

    Returns:
        Hits with ``content``, ``chunk_id``, ``metadata``, ``score`` (L2 distance).

    Raises:
        KeyError: If the FAISS index key is missing.
    """
    connection = sqlite3.connect(db_path)
    try:
        row = connection.execute(
            "SELECT value FROM kvstore WHERE key=?",
            (_kv_key("faiss_index", vector_store_id),),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise KeyError(_kv_key("faiss_index", vector_store_id))

    payload = json.loads(row[0])
    index = _deserialize_faiss_index(payload["faiss_index"])
    query = np.asarray([list(query_embedding)], dtype=np.float32)
    distances, labels = index.search(query, k)

    hits: list[dict[str, Any]] = []
    for distance, label in zip(distances[0], labels[0], strict=True):
        if int(label) < 0:
            continue
        chunk = json.loads(payload["chunk_by_index"][str(int(label))])
        hits.append(
            {
                "content": chunk["content"],
                "chunk_id": chunk["chunk_id"],
                "metadata": chunk.get("metadata", {}),
                "score": float(distance),
            }
        )
    return hits


def list_sqlite_faiss_vector_store_ids(db_path: str | Path) -> list[str]:
    """Return vector_store_ids present in an sqlite-faiss SQLite file."""
    prefix = f"{KV_NAMESPACE}:faiss_index:{KV_VERSION}::"
    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            "SELECT key FROM kvstore WHERE key LIKE ?",
            (f"{prefix}%",),
        ).fetchall()
    finally:
        connection.close()
    return [str(row[0][len(prefix) :]) for row in rows]


def resolve_model_name_or_dir(model_name: str, embeddings_model_dir: str | os.PathLike[str]) -> str:
    """Return a local model directory when it exists, otherwise the model name."""
    if os.path.exists(embeddings_model_dir):
        return os.path.realpath(embeddings_model_dir)
    return model_name


def manual_chunk_dicts(nodes: Sequence[Any]) -> list[dict[str, Any]]:
    """Build OGX-shaped chunk dicts from LlamaIndex text nodes."""
    records: list[dict[str, Any]] = []
    for node in nodes:
        node.metadata["document_id"] = node.ref_doc_id
        records.append(
            {
                "content": node.text,
                "metadata": node.metadata,
                "chunk_metadata": {
                    "document_id": node.ref_doc_id,
                    "chunk_id": node.id_,
                    "source": node.metadata.get("docs_url", node.metadata["title"]),
                },
                "chunk_id": node.id_,
            }
        )
    return records


class SqliteFaissDB:
    """Write a FAISS index into a SQLite kvstore file.

    Combined with ``_BaseDB`` in :mod:`document_processor` so LlamaIndex
    chunking is shared with the other backends. ``super().__init__`` runs
    ``_BaseDB.__init__``.
    """

    config: Any
    _split_and_filter: Any
    write_lcs_config: Any

    LCS_CFG_FILENAME = config_templates.LCS_CFG_FILENAME

    def __init__(self, config: Any) -> None:
        """Initialize the sqlite-faiss writer.

        Chunking uses LlamaIndex (same as ``llamastack-faiss`` manual chunking).
        Embeddings come from SentenceTransformer.
        """
        if config.vector_store_type != "sqlite-faiss":
            raise RuntimeError(
                f"Unexpected vector store type for SqliteFaissDB: {config.vector_store_type}"
            )

        super().__init__(config)  # type: ignore[call-arg]

        self.model_name_or_dir = resolve_model_name_or_dir(
            config.model_name, config.embeddings_model_dir
        )

        self._embedder = SentenceTransformer(self.model_name_or_dir)
        self.config.embedding_dimension = self._embedder.get_sentence_embedding_dimension()
        self.db_filename = "faiss_store.db"
        self.documents: list[dict[str, Any]] = []

    def add_docs(self, docs: list[Document]) -> None:
        """Chunk documents with LlamaIndex and queue them for save."""
        if not self.config.manual_chunking:
            LOG.warning(
                "Ignoring auto-chunking for sqlite-faiss; OGX Files API is unavailable. "
                "Using LlamaIndex chunking."
            )
        self.documents.extend(manual_chunk_dicts(self._split_and_filter(docs)))

    def save(  # pylint: disable=unused-argument
        self,
        index: str,
        output_dir: str,
        embedded_files: Optional[int] = None,
        exec_time: Optional[int] = None,
    ) -> str:
        """Embed queued chunks and write faiss_store.db plus lightspeed-stack.yaml."""
        if not self.documents:
            raise ValueError("No documents to save")

        os.makedirs(output_dir, exist_ok=True)
        db_file = os.path.realpath(os.path.join(output_dir, self.db_filename))
        texts = [document["content"] for document in self.documents]
        matrix = np.asarray(
            self._embedder.encode(texts, show_progress_bar=self.config.show_progress),
            dtype=np.float32,
        )
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)

        chunks: list[SqliteFaissChunk] = []
        for document, embedding in zip(self.documents, matrix, strict=True):
            metadata = {**document["metadata"], "source": index}
            chunks.append(
                SqliteFaissChunk(
                    content=document["content"],
                    chunk_id=document["chunk_id"],
                    metadata=metadata,
                    chunk_metadata=document["chunk_metadata"],
                    embedding=[float(value) for value in embedding],
                )
            )

        vector_store_id = f"vs_{uuid.uuid4()}"
        write_sqlite_faiss_store(
            db_file,
            vector_store_id=vector_store_id,
            provider_id=index,
            store_name=index,
            embedding_model=self.config.model_name,
            embedding_dimension=int(self.config.embedding_dimension),
            chunks=chunks,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        lcs_file = os.path.join(output_dir, self.LCS_CFG_FILENAME)
        self.write_lcs_config(index, lcs_file, vector_store_id, db_file)
        return vector_store_id
