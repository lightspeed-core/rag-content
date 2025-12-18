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

import pytest

from lightspeed_rag_content import document_processor
from tests.conftest import RagMockEmbedding


class TestConfig:
    """Test cases for the _Config class in document_processor module."""

    def test_config(self):
        """Test that _Config class properly initializes and stores configuration values."""
        config = document_processor._Config(
            chunk_size=380,
            chunk_overlap=0,
            model_name="sentence-transformers/all-mpnet-base-v2",
            embeddings_model_dir="./embeddings_model",
        )
        assert config.chunk_size == 380
        assert config.chunk_overlap == 0
        assert config.model_name == "sentence-transformers/all-mpnet-base-v2"
        assert config.embeddings_model_dir == "./embeddings_model"


@pytest.fixture
def mock_processor(mocker):
    """Fixture to mock dependencies for DocumentProcessor tests."""
    mocker.patch.object(
        document_processor, "HuggingFaceEmbedding", new=RagMockEmbedding
    )
    log = mocker.patch.object(document_processor, "LOG")
    indexdb = mocker.patch.object(document_processor, "_LlamaIndexDB")
    llamadb = mocker.patch.object(document_processor, "_LlamaStackDB")
    yield {
        "log": log,
        "indexdb": indexdb,
        "llamadb": llamadb,
        "params": {
            "chunk_size": 380,
            "chunk_overlap": 0,
            "model_name": "sentence-transformers/all-mpnet-base-v2",
            "embeddings_model_dir": "embeddings_model",
            "num_workers": 10,
            "doc_type": "text",
        },
    }


class TestDocumentProcessor:
    """Test cases for the DocumentProcessor class in document_processor module."""

    def test_init_default(self, mock_processor):
        """Test DocumentProcessor initialization with default vector store type (faiss)."""
        doc_processor = document_processor.DocumentProcessor(**mock_processor["params"])

        mock_processor["log"].warning.assert_not_called()
        mock_processor["indexdb"].assert_called_once_with(doc_processor.config)

        assert doc_processor is not None

        expected_params = mock_processor["params"].copy()
        expected_params.update(  # Add default values
            embedding_dimension=None,  # Not calculated because class is mocked
            manual_chunking=True,
            table_name=None,
            vector_store_type="faiss",
            exclude_embed_metadata=None,
            exclude_llm_metadata=None,
        )
        assert expected_params == doc_processor.config._Config__attributes
        assert doc_processor._num_embedded_files == 0

        assert expected_params["embeddings_model_dir"] == os.environ["HF_HOME"]
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

    @pytest.mark.parametrize("vector_store_type", ["faiss", "postgres"])
    def test_init_llama_index(self, vector_store_type, mock_processor):
        """Test DocumentProcessor initialization with LlamaIndex-compatible vector store types."""
        params = mock_processor["params"].copy()
        params["vector_store_type"] = vector_store_type

        doc_processor = document_processor.DocumentProcessor(**params)
        mock_processor["log"].warning.assert_not_called()
        mock_processor["indexdb"].assert_called_once_with(doc_processor.config)

        assert doc_processor is not None

        params.update(  # Add default values
            embedding_dimension=None,  # Not calculated because class is mocked
            manual_chunking=True,
            table_name=None,
            vector_store_type=vector_store_type,
            exclude_embed_metadata=None,
            exclude_llm_metadata=None,
        )
        if vector_store_type == "postgres":
            params["table_name"] = "table_name"
        assert params == doc_processor.config._Config__attributes
        assert doc_processor._num_embedded_files == 0

        assert params["embeddings_model_dir"] == os.environ["HF_HOME"]
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        mock_processor["indexdb"].reset_mock()

    @pytest.mark.parametrize(
        "vector_store_type", ["llamastack-faiss", "llamastack-sqlite-vec"]
    )
    def test_init_llama_stack(self, vector_store_type, mock_processor):
        """Test DocumentProcessor initialization with LlamaStack-compatible vector store types."""
        params = mock_processor["params"].copy()
        params["vector_store_type"] = vector_store_type

        doc_processor = document_processor.DocumentProcessor(**params)
        mock_processor["log"].warning.assert_not_called()
        mock_processor["llamadb"].assert_called_once_with(doc_processor.config)

        assert doc_processor is not None

        params.update(  # Add default values
            embedding_dimension=None,  # Not calculated because class is mocked
            manual_chunking=True,
            table_name=None,
            exclude_embed_metadata=None,
            exclude_llm_metadata=None,
        )
        assert params == doc_processor.config._Config__attributes
        assert doc_processor._num_embedded_files == 0

        assert params["embeddings_model_dir"] == os.environ["HF_HOME"]
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

    def test__check_config_faiss_auto_chunking(self, mock_processor):
        """Test that _check_config logs warning when using faiss with auto chunking disabled."""
        config = document_processor._Config(
            vector_store_type="faiss",
            manual_chunking=False,
            table_name=None,
        )
        document_processor.DocumentProcessor._check_config(config)
        mock_processor["log"].warning.assert_called_once_with(mock.ANY)

    def test__check_config_faiss_table_name(self, mock_processor):
        """Test that _check_config logs warning when using faiss with a table name specified."""
        config = document_processor._Config(
            vector_store_type="faiss",
            manual_chunking=True,
            table_name="table_name",
        )
        document_processor.DocumentProcessor._check_config(config)
        mock_processor["log"].warning.assert_called_once_with(mock.ANY)

    def test_process(self, mock_processor, mocker):
        """Test the document processing that reads and adds them to the database."""
        doc_processor = document_processor.DocumentProcessor(**mock_processor["params"])

        metadata = mocker.Mock()
        docs = list(range(5))

        reader_mock = mocker.patch.object(document_processor, "SimpleDirectoryReader")
        reader_mock.return_value.load_data.return_value = docs

        doc_processor.process(
            mock.sentinel.docs_dir,
            metadata,
            mock.sentinel.required_exts,
            mock.sentinel.file_extractor,
        )

        reader_mock.assert_called_once_with(
            str(mock.sentinel.docs_dir),
            recursive=True,
            file_metadata=metadata.populate,
            required_exts=mock.sentinel.required_exts,
            file_extractor=mock.sentinel.file_extractor,
        )

        doc_processor.db.add_docs.assert_called_once_with(docs)
        assert len(docs) == doc_processor._num_embedded_files

    def test_save(self, mock_processor):
        """Test saving the document processor's database to disk."""
        doc_processor = document_processor.DocumentProcessor(**mock_processor["params"])

        doc_processor.save(mock.sentinel.index, mock.sentinel.output_dir)


class TestBaseDB:
    """Test cases for the _BaseDB class in document_processor module."""

    def test__remove_metadata(self):
        """Test that _remove_metadata removes specified keys from metadata dictionary."""
        metadata = {
            "file_path": "/path/to/file",
            "url": "https://example.com",
            "title": "Test Document",
            "file_name": "test.txt",
        }
        keys_to_remove = ["file_path", "file_name"]

        result = document_processor._BaseDB._remove_metadata(metadata, keys_to_remove)

        assert "file_path" not in result
        assert "file_name" not in result
        assert "url" in result
        assert "title" in result
        assert result["url"] == "https://example.com"
        assert result["title"] == "Test Document"

    def test__remove_metadata_empty_list(self):
        """Test that _remove_metadata returns original metadata when no keys to remove."""
        metadata = {"key1": "value1", "key2": "value2"}

        result = document_processor._BaseDB._remove_metadata(metadata, [])

        assert result == metadata

    def test__remove_metadata_nonexistent_keys(self):
        """Test that _remove_metadata handles nonexistent keys gracefully."""
        metadata = {"key1": "value1", "key2": "value2"}

        result = document_processor._BaseDB._remove_metadata(
            metadata, ["nonexistent_key"]
        )

        assert result == metadata
