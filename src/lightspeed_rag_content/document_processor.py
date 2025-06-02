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
"""Document processing for vector database."""

import json
import logging
import os
import time
from collections import namedtuple
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import faiss
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.llms.utils import resolve_llm
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import TextNode
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.vector_stores.postgres import PGVectorStore

if TYPE_CHECKING:
    from llama_index.core.vector_stores.types import BasePydanticVectorStore

from lightspeed_rag_content.metadata_processor import MetadataProcessor

LOG = logging.getLogger(__name__)

DocumentSettings = namedtuple(
    "DocumentSettings", ["settings", "embedding_dimension", "storage_context"]
)


class DocumentProcessor:
    """Processes documents into vector database entries."""

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        model_name: str,
        embeddings_model_dir: Path,
        num_workers: Optional[int] = 0,
        vector_store_type: str = "faiss",
        table_name: str = "table_name",
    ):
        """Initialize instance."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self.embeddings_model_dir = embeddings_model_dir
        self.num_workers = num_workers
        self.vector_store_type = vector_store_type
        self.table_name = table_name

        # List of good nodes
        self._good_nodes: list[TextNode] = []
        # Total number of embedded files
        self._num_embedded_files = 0
        # Start of time, used to calculate the execution time
        self._start_time = time.time()

        os.environ["HF_HOME"] = str(self.embeddings_model_dir)
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        self._settings = self._get_settings()

    def _get_settings(self) -> DocumentSettings:
        """Return DocumentSettings tuple.

        DocumenSettings consists of llama-index Settings, embedding dimension and StorageContext.
        """
        Settings.chunk_size = self.chunk_size
        Settings.chunk_overlap = self.chunk_overlap
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=str(self.embeddings_model_dir)
        )
        Settings.llm = resolve_llm(None)

        embedding_dimension = len(
            Settings.embed_model.get_text_embedding("random text")
        )

        vector_store: None | BasePydanticVectorStore = None
        if self.vector_store_type == "faiss":
            faiss_index = faiss.IndexFlatIP(embedding_dimension)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
        elif self.vector_store_type == "postgres":
            user = os.getenv("POSTGRES_USER")
            password = os.getenv("POSTGRES_PASSWORD")
            host = os.getenv("POSTGRES_HOST")
            port = os.getenv("POSTGRES_PORT")
            database = os.getenv("POSTGRES_DATABASE")
            table_name = self.table_name
            vector_store = PGVectorStore.from_params(
                database=database,
                host=host,
                password=password,
                port=port,
                user=user,
                table_name=table_name,
                embed_dim=embedding_dimension,  # openai embedding dimension
            )
        else:
            raise RuntimeError(f"Unknown vector store type: {self.vector_store_type}")

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        return DocumentSettings(Settings, embedding_dimension, storage_context)

    def _got_whitespace(self, text: str) -> bool:
        """Indicate if the parameter string contains whitespace."""
        for c in text:
            if c.isspace():
                return True
        return False

    def _filter_out_invalid_nodes(self, nodes: list[TextNode]) -> list[TextNode]:
        """Filter out invalid nodes."""
        good_nodes = []
        for node in nodes:
            if isinstance(node, TextNode) and self._got_whitespace(node.text):
                # Exclude given metadata during embedding
                good_nodes.append(node)
            else:
                LOG.debug("Skipping node without whitespace: %s", repr(node))
        return good_nodes

    def _save_index(self, index: str, persist_folder: str) -> None:
        """Create and save the Vector Store Index."""
        idx = VectorStoreIndex(
            self._good_nodes,
            storage_context=self._settings.storage_context,
        )
        idx.set_index_id(index)
        idx.storage_context.persist(persist_dir=persist_folder)

    def _save_metadata(self, index: str, persist_folder: str) -> None:
        """Create and save the metadata."""
        metadata: dict[str, Union[str, int, float]] = {}
        metadata["execution-time"] = time.time() - self._start_time
        metadata["llm"] = "None"
        metadata["embedding-model"] = self.model_name
        metadata["index-id"] = index
        if self.vector_store_type == "faiss":
            metadata["vector-db"] = "faiss.IndexFlatIP"
        elif self.vector_store_type == "postgres":
            metadata["vector-db"] = "PGVectorStore"
        metadata["embedding-dimension"] = self._settings.embedding_dimension
        metadata["chunk"] = self.chunk_size
        metadata["overlap"] = self.chunk_overlap
        metadata["total-embedded-files"] = self._num_embedded_files
        with open(os.path.join(persist_folder, "metadata.json"), "w") as file:
            file.write(json.dumps(metadata))

    def process(
        self,
        docs_dir: Path,
        metadata: MetadataProcessor,
        required_exts: Optional[list[str]] = None,
        file_extractor: Optional[dict[str, BaseReader]] = None,
    ) -> None:
        """Read documents from a path and split them into nodes for a vector database."""
        reader = SimpleDirectoryReader(
            str(docs_dir),
            recursive=True,
            file_metadata=metadata.populate,
            required_exts=required_exts,
            file_extractor=file_extractor,
        )

        # Create chunks/nodes
        docs = reader.load_data(num_workers=self.num_workers)
        nodes = self._settings.settings.text_splitter.get_nodes_from_documents(docs)
        self._good_nodes.extend(self._filter_out_invalid_nodes(nodes))

        # Count embedded files and unreachable nodes
        self._num_embedded_files += len(docs)

    def save(self, index: str, output_dir: str) -> None:
        """Save vector store index and metadata."""
        self._save_index(index, output_dir)
        self._save_metadata(index, output_dir)
