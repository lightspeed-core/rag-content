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

import os
from unittest import mock
from unittest.mock import AsyncMock

import pytest
from llama_index.core.schema import TextNode

from lightspeed_rag_content import document_processor
from tests.conftest import RagMockEmbedding

FAISS_EXPECTED = """version: 2
image_name: starter

apis:
- files
- tool_runtime
- vector_io
- inference

server:
  port: 8321
conversations_store:
  db_path: /tmp/conversations.db
  type: sqlite
metadata_store:
  db_path: /tmp/registry.db
  type: sqlite

providers:
  inference:
  - config: {{}}
    provider_id: sentence-transformers
    provider_type: inline::sentence-transformers
  files:
  - config:
      metadata_store:
        table_name: files_metadata
        backend: sql_default
      storage_dir: /tmp/files
    provider_id: meta-reference-files
    provider_type: inline::localfs
  tool_runtime:
  - config: {{}}
    provider_id: rag-runtime
    provider_type: inline::rag-runtime
  vector_io:
  - config:
      persistence:
        namespace: vector_io::{provider_type}
        backend: kv_rag
    provider_id: {index_id}
    provider_type: inline::{provider_type}
storage:
  backends:
    kv_rag:
      type: kv_sqlite
      db_path: {kv_db_path}
    kv_default:
      type: kv_sqlite
      db_path: /tmp/kv_store.db
    sql_default:
      type: sql_sqlite
      db_path: /tmp/sql_store.db
  stores:
    metadata:
      namespace: registry
      backend: kv_default
    inference:
      table_name: inference_store
      backend: sql_default
    conversations:
      table_name: openai_conversations
      backend: sql_default
registered_resources:
  models:
  - metadata:
      embedding_dimension: {dimension}
    model_id: {model_name}
    provider_id: sentence-transformers
    provider_model_id: {model_name_or_dir}
    model_type: embedding
  vector_stores: []
  shields: []
  datasets: []
  scoring_fns: []
  benchmarks: []
  tool_groups:
  - toolgroup_id: builtin::rag
    provider_id: rag-runtime
"""


@pytest.fixture
def llama_stack_processor(mocker):
    """Fixture for _LlamaStackDB tests."""
    mocker.patch.object(
        document_processor, "HuggingFaceEmbedding", new=RagMockEmbedding
    )
    st = mocker.patch.object(document_processor, "SentenceTransformer")
    st.return_value.get_sentence_embedding_dimension.return_value = 768
    # Mock os.path.exists to return False only for the embeddings_model_dir check
    original_exists = os.path.exists

    def mock_exists(path):
        # Return False for embeddings_model_dir, True for everything else
        if "embeddings_model" in str(path):
            return False
        return original_exists(path)

    mocker.patch("os.path.exists", side_effect=mock_exists)

    mocker.patch.object(
        document_processor.Settings.text_splitter.__class__,
        "get_nodes_from_documents",
    )

    model_name = "sentence-transformers/all-mpnet-base-v2"
    config = document_processor._Config(
        chunk_size=380,
        chunk_overlap=0,
        model_name=model_name,
        embeddings_model_dir="",
        vector_store_type="llamastack-faiss",
        embedding_dimension=None,
        manual_chunking=True,
        doc_type="text",
    )
    return {"config": config, "model_name": model_name}


class TestDocumentProcessorLlamaStack:
    """Test cases for the _LlamaStackDB document processor class."""

    def test_init(self, mocker, llama_stack_processor):
        """Test basic initialization of _LlamaStackDB with default settings."""
        temp_dir = mocker.patch.object(
            document_processor.tempfile, "TemporaryDirectory"
        )
        temp_dir.return_value.name = "temp_dir"
        doc = document_processor._LlamaStackDB(llama_stack_processor["config"])

        assert doc.config == llama_stack_processor["config"]
        assert doc.model_name_or_dir == llama_stack_processor["model_name"]
        assert doc.config.embedding_dimension == 768
        assert doc.db_filename == "faiss_store.db"
        assert doc.document_class.__name__ == "RAGDocument"
        assert doc.client_class.__name__ == "AsyncLlamaStackAsLibraryClient"
        assert doc.documents == []
        temp_dir.assert_called_once_with(prefix="ls-rag-")
        assert doc.tmp_dir is temp_dir.return_value
        assert os.environ["LLAMA_STACK_CONFIG_DIR"] == temp_dir.return_value.name

    def test_init_model_path(self, mocker, llama_stack_processor):
        """Test initialization when embeddings_model_dir exists as a local path."""
        temp_dir = mocker.patch.object(
            document_processor.tempfile, "TemporaryDirectory"
        )
        temp_dir.return_value.name = "temp_dir"
        exists_mock = mocker.patch("os.path.exists", return_value=True)
        realpath_mock = mocker.patch("os.path.realpath")

        config = llama_stack_processor["config"]
        config.embeddings_model_dir = "embeddings_model"
        doc = document_processor._LlamaStackDB(config)

        assert doc.config == config
        exists_mock.assert_called_once_with(config.embeddings_model_dir)
        realpath_mock.assert_called_once_with(config.embeddings_model_dir)
        assert doc.model_name_or_dir == realpath_mock.return_value
        assert doc.config.embedding_dimension == 768
        assert doc.db_filename == "faiss_store.db"
        assert doc.document_class.__name__ == "RAGDocument"
        assert doc.client_class.__name__ == "AsyncLlamaStackAsLibraryClient"
        assert doc.documents == []
        temp_dir.assert_called_once_with(prefix="ls-rag-")
        assert doc.tmp_dir is temp_dir.return_value
        assert os.environ["LLAMA_STACK_CONFIG_DIR"] == temp_dir.return_value.name

    def test_write_yaml_config_faiss(self, mocker, llama_stack_processor):
        """Test YAML configuration generation for FAISS vector store backend."""
        mock_open = mocker.patch("builtins.open", new_callable=mocker.mock_open)
        doc = document_processor._LlamaStackDB(llama_stack_processor["config"])

        index_id = "my_index_id"
        yaml_file = "yaml_file"
        db_file = "db_file"
        files_metadata_db_file = "files_metadata_db_file"
        doc.write_yaml_config(index_id, yaml_file, db_file, files_metadata_db_file)

        mock_open.assert_called_once_with(yaml_file, "w", encoding="utf-8")
        data = mock_open.return_value.write.mock_calls[0].args[0]
        assert data == FAISS_EXPECTED.format(
            provider_type="faiss",
            index_id=index_id,
            kv_db_path=db_file,
            dimension=768,
            model_name=llama_stack_processor["model_name"],
            model_name_or_dir=llama_stack_processor["model_name"],
        )

    def test_write_yaml_config_sqlitevec(self, mocker, llama_stack_processor):
        """Test YAML configuration generation for SQLiteVec vector store backend."""
        mock_open = mocker.patch("builtins.open", new_callable=mocker.mock_open)
        config = llama_stack_processor["config"]
        config.vector_store_type = "llamastack-sqlite-vec"
        doc = document_processor._LlamaStackDB(config)

        provider_id = "my_provider_id"
        yaml_file = "yaml_file"
        db_file = "db_file"
        files_metadata_db_file = "files_metadata_db_file"

        doc.write_yaml_config(provider_id, yaml_file, db_file, files_metadata_db_file)

        mock_open.assert_called_once_with(yaml_file, "w", encoding="utf-8")
        expected = f"""version: 2
image_name: starter

apis:
- files
- tool_runtime
- vector_io
- inference

server:
  port: 8321
conversations_store:
  db_path: /tmp/conversations.db
  type: sqlite
metadata_store:
  db_path: /tmp/registry.db
  type: sqlite

providers:
  inference:
  - config: {{}}
    provider_id: sentence-transformers
    provider_type: inline::sentence-transformers
  files:
  - config:
      metadata_store:
        table_name: files_metadata
        backend: sql_default
      storage_dir: /tmp/files
    provider_id: meta-reference-files
    provider_type: inline::localfs
  tool_runtime:
  - config: {{}}
    provider_id: rag-runtime
    provider_type: inline::rag-runtime
  vector_io:
  - config:
      persistence:
        namespace: vector_io::sqlite-vec
        backend: kv_rag
    provider_id: {provider_id}
    provider_type: inline::sqlite-vec
storage:
  backends:
    kv_rag:
      type: kv_sqlite
      db_path: {db_file}
    kv_default:
      type: kv_sqlite
      db_path: /tmp/kv_store.db
    sql_default:
      type: sql_sqlite
      db_path: /tmp/sql_store.db
  stores:
    metadata:
      namespace: registry
      backend: kv_default
    inference:
      table_name: inference_store
      backend: sql_default
    conversations:
      table_name: openai_conversations
      backend: sql_default
registered_resources:
  models:
  - metadata:
      embedding_dimension: 768
    model_id: sentence-transformers/all-mpnet-base-v2
    provider_id: sentence-transformers
    provider_model_id: sentence-transformers/all-mpnet-base-v2
    model_type: embedding
  vector_stores: []
  shields: []
  datasets: []
  scoring_fns: []
  benchmarks: []
  tool_groups:
  - toolgroup_id: builtin::rag
    provider_id: rag-runtime
"""
        data = mock_open.return_value.write.mock_calls[0].args[0]
        assert data == expected

    def test_run_llama_stack(self, mocker, llama_stack_processor):
        """Test running with llama-stack client lifecycle management."""
        import asyncio

        temp_dir = mocker.patch.object(
            document_processor.tempfile, "TemporaryDirectory"
        )
        temp_dir.return_value.name = "tempdir"
        doc = document_processor._LlamaStackDB(llama_stack_processor["config"])
        yaml_file = "yaml_file"

        # Mock the client class to support async context manager
        client_instance = mocker.Mock()
        client_mock = mocker.patch.object(doc, "client_class")
        client_mock.return_value.__aenter__ = AsyncMock(return_value=client_instance)
        client_mock.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock the processing method
        insert_mock = mocker.patch.object(
            doc, "_insert_prechunked_documents", new=AsyncMock(return_value="vs_123")
        )

        res = asyncio.run(doc._run_llama_stack(yaml_file, "test-index"))
        assert res == "vs_123"
        client_mock.assert_called_once_with(yaml_file)
        client_mock.return_value.__aenter__.assert_awaited_once()
        client_mock.return_value.__aexit__.assert_awaited_once()
        insert_mock.assert_awaited_once()

        temp_dir.assert_called_once_with(prefix="ls-rag-")
        assert os.environ["LLAMA_STACK_CONFIG_DIR"] == temp_dir.return_value.name

    def test_add_docs_manual_chunking(self, mocker, llama_stack_processor):
        """Test adding documents with manual chunking enabled."""
        doc = document_processor._LlamaStackDB(llama_stack_processor["config"])
        nodes = [
            mocker.Mock(
                spec=TextNode,
                ref_doc_id=i,
                id_=i * 3,
                text=str(i),
                metadata={"title": f"title{i}", "docs_url": f"https://redhat.com/{i}"},
            )
            for i in range(1, 3)
        ]
        mock_filter = mocker.patch.object(doc, "_split_and_filter", return_value=nodes)

        docs = list(range(5))
        doc.add_docs(docs)

        mock_filter.assert_called_once_with(docs)
        expect = [
            {
                "content": "1",
                "metadata": {
                    "document_id": 1,
                    "title": "title1",
                    "docs_url": "https://redhat.com/1",
                },
                "chunk_metadata": {
                    "document_id": 1,
                    "chunk_id": 3,
                    "source": "https://redhat.com/1",
                },
                "chunk_id": 3,
            },
            {
                "content": "2",
                "metadata": {
                    "document_id": 2,
                    "title": "title2",
                    "docs_url": "https://redhat.com/2",
                },
                "chunk_metadata": {
                    "document_id": 2,
                    "chunk_id": 6,
                    "source": "https://redhat.com/2",
                },
                "chunk_id": 6,
            },
        ]
        assert doc.documents == expect

    def test_add_docs_auto_chunking(self, mocker, llama_stack_processor):
        """Test adding documents with automatic chunking enabled."""
        config = llama_stack_processor["config"]
        config.manual_chunking = False
        doc = document_processor._LlamaStackDB(config)

        fake_out_docs = [mocker.Mock(), mocker.Mock()]
        doc_class = mocker.patch.object(
            doc, "document_class", side_effect=fake_out_docs
        )
        mock_filter = mocker.patch.object(doc, "_split_and_filter")

        in_docs = [
            mocker.Mock(doc_id=str(i), text=str(i), metadata={"title": f"title{i}"})
            for i in range(1, 3)
        ]

        doc.add_docs(in_docs)

        mock_filter.assert_not_called()
        assert doc_class.call_count == len(in_docs)
        doc_class.assert_has_calls(
            [
                mocker.call(
                    document_id=d.doc_id,
                    content=d.text,
                    mime_type="text/plain",
                    metadata=d.metadata,
                )
                for d in in_docs
            ]
        )
        assert doc.documents == fake_out_docs

    def _test_save(self, mocker, config):
        """Helper function to set up and verify save functionality."""
        doc = document_processor._LlamaStackDB(config)
        # Create sample documents with proper structure
        if config.manual_chunking:
            # Manual chunking uses dicts
            doc.documents = [
                {
                    "content": "test content 1",
                    "metadata": {
                        "document_id": "doc1",
                        "title": "Test Doc 1",
                        "docs_url": "https://example.com/doc1",
                    },
                    "chunk_metadata": {
                        "document_id": "doc1",
                        "chunk_id": 1,
                        "source": "https://example.com/doc1",
                    },
                    "chunk_id": 1,
                },
                {
                    "content": "test content 2",
                    "metadata": {
                        "document_id": "doc2",
                        "title": "Test Doc 2",
                        "docs_url": "https://example.com/doc2",
                    },
                    "chunk_metadata": {
                        "document_id": "doc2",
                        "chunk_id": 2,
                        "source": "https://example.com/doc2",
                    },
                    "chunk_id": 2,
                },
            ]
        else:
            doc.documents = [
                mocker.Mock(
                    document_id="doc1",
                    content="test content 1",
                    metadata={
                        "title": "Test Doc 1",
                        "docs_url": "https://example.com/doc1",
                    },
                ),
                mocker.Mock(
                    document_id="doc2",
                    content="test content 2",
                    metadata={
                        "title": "Test Doc 2",
                        "docs_url": "https://example.com/doc2",
                    },
                ),
            ]

        write_cfg = mocker.patch.object(doc, "write_yaml_config")
        update_yaml = mocker.patch.object(doc, "_update_yaml_config")

        # Mock client_class to support async context manager
        client_instance = mocker.Mock()
        client_class_mock = mocker.patch.object(doc, "client_class")
        client_class_mock.return_value.__aenter__ = AsyncMock(
            return_value=client_instance
        )
        client_class_mock.return_value.__aexit__ = AsyncMock(return_value=None)

        makedirs = mocker.patch("os.makedirs")
        realpath = mocker.patch(
            "os.path.realpath",
            side_effect=[
                "/cwd/out_dir/faiss_store.db",
                "/cwd/out_dir/files_metadata.db",
            ],
        )

        # Mock async client methods
        vector_store_mock = mocker.Mock()
        vector_store_mock.id = "vs_123"
        client_instance.vector_stores.create = AsyncMock(return_value=vector_store_mock)
        client_instance.files.create = AsyncMock(
            return_value=mocker.Mock(id="file_123")
        )
        client_instance.vector_io.insert = AsyncMock()

        # Mock embeddings.create for manual chunking
        embedding_response_mock = mocker.Mock()
        embedding_response_mock.data = [mocker.Mock(embedding=[0.1] * 768)]
        client_instance.embeddings.create = AsyncMock(
            return_value=embedding_response_mock
        )

        batch_mock = mocker.Mock()
        batch_mock.status = "completed"
        client_instance.vector_stores.file_batches.create = AsyncMock(
            return_value=batch_mock
        )

        # Mock vector_stores.files methods for auto chunking
        vs_file_mock = mocker.Mock()
        vs_file_mock.status = "completed"
        client_instance.vector_stores.files.create = AsyncMock(
            return_value=vs_file_mock
        )
        client_instance.vector_stores.files.retrieve = AsyncMock(
            return_value=vs_file_mock
        )

        doc.save(mock.sentinel.index, "out_dir")

        makedirs.assert_called_once_with("out_dir", exist_ok=True)
        assert realpath.call_count == 2
        realpath.assert_any_call("out_dir/faiss_store.db")
        realpath.assert_any_call("out_dir/files_metadata.db")
        write_cfg.assert_called_once_with(
            mock.sentinel.index,
            "out_dir/llama-stack.yaml",
            "/cwd/out_dir/faiss_store.db",
            "/cwd/out_dir/files_metadata.db",
        )
        update_yaml.assert_called_once_with(
            "out_dir/llama-stack.yaml",
            mock.sentinel.index,
            "vs_123",
        )
        # Verify client lifecycle (async context manager)
        client_class_mock.return_value.__aenter__.assert_awaited_once()
        client_class_mock.return_value.__aexit__.assert_awaited_once()

        return client_instance

    def test_save_manual_chunking(self, mocker, llama_stack_processor):
        """Test saving documents with manual chunking workflow."""
        client = self._test_save(mocker, llama_stack_processor["config"])
        # Verify vector_io.insert was called once with the right vector_store_id
        # Documents are modified during processing, so we just check it was called
        assert client.vector_io.insert.await_count == 1
        call_kwargs = client.vector_io.insert.await_args.kwargs
        assert call_kwargs["vector_store_id"] == "vs_123"
        assert "chunks" in call_kwargs
        assert len(call_kwargs["chunks"]) == 2

    def test_save_auto_chunking(self, mocker, llama_stack_processor):
        """Test saving documents with automatic chunking workflow."""
        config = llama_stack_processor["config"]
        config.manual_chunking = False
        client = self._test_save(mocker, config)
        # Verify files.create was called for each document (single file upload)
        assert client.files.create.await_count == 2
        assert client.vector_stores.files.create.await_count == 2
