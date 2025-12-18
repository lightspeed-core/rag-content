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
from llama_index.core.schema import TextNode

from lightspeed_rag_content import document_processor
from tests.conftest import RagMockEmbedding

FAISS_EXPECTED = """version: 2
image_name: ollama
apis:
  - inference
  - vector_io
  - tool_runtime
  - files
providers:
  inference:
    - provider_id: sentence-transformers
      provider_type: inline::sentence-transformers
      config: {{}}
  vector_io:
    - provider_id: {provider_id}
      provider_type: inline::faiss
      config:
        kvstore:
          type: sqlite
          namespace: null
          db_path: {db_file}
        
  files:
    - provider_id: localfs
      provider_type: inline::localfs
      config:
        storage_dir: /tmp/llama-stack-files
        metadata_store:
          type: sqlite
          db_path: files_metadata.db
  tool_runtime:
  - provider_id: rag-runtime
    provider_type: inline::rag-runtime
    config: {{}}
models:
  - metadata:
      embedding_dimension: 768
    model_id: sentence-transformers/all-mpnet-base-v2
    provider_id: sentence-transformers
    provider_model_id: sentence-transformers/all-mpnet-base-v2
    model_type: embedding
tool_groups:
  - toolgroup_id: builtin::rag
    provider_id: rag-runtime
vector_dbs:
  - vector_db_id: {provider_id}
    embedding_model: sentence-transformers/all-mpnet-base-v2
    embedding_dimension: 768
    provider_id: {provider_id}
    {vector_store_id}
"""


@pytest.fixture
def llama_stack_processor(mocker):
    """Fixture for _LlamaStackDB tests."""
    mocker.patch.object(
        document_processor, "HuggingFaceEmbedding", new=RagMockEmbedding
    )
    st = mocker.patch.object(document_processor, "SentenceTransformer")
    st.return_value.get_sentence_embedding_dimension.return_value = 768
    mocker.patch("os.path.exists", return_value=False)
    # Mock tiktoken to prevent network calls during initialization
    mock_encoding = mocker.Mock()
    mocker.patch("tiktoken.get_encoding", return_value=mock_encoding)
    mocker.patch("tiktoken.encoding_for_model", return_value=mock_encoding)

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
        exclude_embed_metadata=[],
        exclude_llm_metadata=[],
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
        assert doc.client_class.__name__ == "LlamaStackAsLibraryClient"
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

        # Mock exists to return True for embeddings_model_dir, False for tiktoken cache
        def exists_side_effect(path):
            if "embeddings_model" in str(path):
                return True
            return False

        exists_mock = mocker.patch("os.path.exists", side_effect=exists_side_effect)
        realpath_mock = mocker.patch("os.path.realpath")

        config = llama_stack_processor["config"]
        config.embeddings_model_dir = "embeddings_model"
        doc = document_processor._LlamaStackDB(config)

        assert doc.config == config
        # Check that exists was called with embeddings_model_dir
        assert any(
            "embeddings_model" in str(call) for call in exists_mock.call_args_list
        )
        realpath_mock.assert_called_once_with(config.embeddings_model_dir)
        assert doc.model_name_or_dir == realpath_mock.return_value
        assert doc.config.embedding_dimension == 768
        assert doc.db_filename == "faiss_store.db"
        assert doc.document_class.__name__ == "RAGDocument"
        assert doc.client_class.__name__ == "LlamaStackAsLibraryClient"
        assert doc.documents == []
        temp_dir.assert_called_once_with(prefix="ls-rag-")
        assert doc.tmp_dir is temp_dir.return_value
        assert os.environ["LLAMA_STACK_CONFIG_DIR"] == temp_dir.return_value.name

    def test_write_yaml_config_faiss(self, mocker, llama_stack_processor):
        """Test YAML configuration generation for FAISS vector store backend."""
        mock_open = mocker.patch("builtins.open", new_callable=mocker.mock_open)
        doc = document_processor._LlamaStackDB(llama_stack_processor["config"])

        provider_id = "my_provider_id"
        yaml_file = "yaml_file"
        db_file = "db_file"
        vector_store_id = ""
        doc.write_yaml_config(provider_id, yaml_file, db_file)

        mock_open.assert_called_once_with(yaml_file, "w", encoding="utf-8")
        data = mock_open.return_value.write.mock_calls[0].args[0]
        assert data == FAISS_EXPECTED.format(
            provider_id=provider_id, db_file=db_file, vector_store_id=vector_store_id
        )

    def test_write_yaml_config_faiss_with_provider_vector_db_id(
        self, mocker, llama_stack_processor
    ):
        """Test YAML configuration generation for FAISS vector store backend."""
        mock_open = mocker.patch("builtins.open", new_callable=mocker.mock_open)
        doc = document_processor._LlamaStackDB(llama_stack_processor["config"])

        provider_id = "my_provider_id"
        yaml_file = "yaml_file"
        db_file = "db_file"
        vector_store_id = "provider_vector_db_id: my_provider_vector_db_id"
        doc.write_yaml_config(
            provider_id, yaml_file, db_file, provider_vector_db_id=vector_store_id
        )

        mock_open.assert_called_once_with(yaml_file, "w", encoding="utf-8")
        data = mock_open.return_value.write.mock_calls[0].args[0]
        assert data == FAISS_EXPECTED.format(
            provider_id=provider_id, db_file=db_file, vector_store_id=vector_store_id
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
        vector_store_id = ""

        doc.write_yaml_config(provider_id, yaml_file, db_file)

        mock_open.assert_called_once_with(yaml_file, "w", encoding="utf-8")
        expected = f"""version: 2
image_name: ollama
apis:
  - inference
  - vector_io
  - tool_runtime
  - files
providers:
  inference:
    - provider_id: sentence-transformers
      provider_type: inline::sentence-transformers
      config: {{}}
  vector_io:
    - provider_id: {provider_id}
      provider_type: inline::sqlite-vec
      config:
        kvstore:
          type: sqlite
          namespace: null
          db_path: {db_file}
        db_path: {db_file}
  files:
    - provider_id: localfs
      provider_type: inline::localfs
      config:
        storage_dir: /tmp/llama-stack-files
        metadata_store:
          type: sqlite
          db_path: files_metadata.db
  tool_runtime:
  - provider_id: rag-runtime
    provider_type: inline::rag-runtime
    config: {{}}
models:
  - metadata:
      embedding_dimension: 768
    model_id: sentence-transformers/all-mpnet-base-v2
    provider_id: sentence-transformers
    provider_model_id: sentence-transformers/all-mpnet-base-v2
    model_type: embedding
tool_groups:
  - toolgroup_id: builtin::rag
    provider_id: rag-runtime
vector_dbs:
  - vector_db_id: {provider_id}
    embedding_model: sentence-transformers/all-mpnet-base-v2
    embedding_dimension: 768
    provider_id: {provider_id}
    {vector_store_id}
"""
        data = mock_open.return_value.write.mock_calls[0].args[0]
        assert data == expected

    def test_start_llama_stack(self, mocker, llama_stack_processor):
        """Test starting the Llama Stack client."""
        temp_dir = mocker.patch.object(
            document_processor.tempfile, "TemporaryDirectory"
        )
        temp_dir.return_value.name = "tempdir"
        doc = document_processor._LlamaStackDB(llama_stack_processor["config"])
        yaml_file = "yaml_file"

        client_mock = mocker.patch.object(doc, "client_class")
        res = doc._start_llama_stack(yaml_file)
        assert res == client_mock.return_value
        client_mock.assert_called_once_with(yaml_file)

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
                "mime_type": "text/plain",
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
                "embed_metadata": {
                    "document_id": 1,
                    "title": "title1",
                    "docs_url": "https://redhat.com/1",
                },
            },
            {
                "content": "2",
                "mime_type": "text/plain",
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
                "embed_metadata": {
                    "document_id": 2,
                    "title": "title2",
                    "docs_url": "https://redhat.com/2",
                },
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
        doc.documents = [
            {
                "content": "test",
                "mime_type": "text/plain",
                "embed_metadata": {"title": "test"},
                "metadata": {"title": "test"},
                "chunk_metadata": {"document_id": 1, "chunk_id": 1},
            }
        ]

        write_cfg = mocker.patch.object(doc, "write_yaml_config")
        client = mocker.patch.object(doc, "_start_llama_stack")
        mock_embeddings_response = mocker.Mock()
        mock_embeddings_response.embeddings = [[0.1] * 768]
        client.return_value.inference.embeddings.return_value = mock_embeddings_response
        realpath = mocker.patch(
            "os.path.realpath", return_value="/cwd/out_dir/vector_store.db"
        )

        doc.save(mock.sentinel.index, "out_dir")

        realpath.assert_called_once_with("out_dir/faiss_store.db")
        write_cfg.assert_called_once_with(
            mock.sentinel.index,
            "out_dir/llama-stack.yaml",
            "/cwd/out_dir/vector_store.db",
            f"provider_vector_db_id: {mock.sentinel.index}",
        )

        client.return_value.vector_dbs.register.assert_not_called()
        return client.return_value

    def test_save_manual_chunking(self, mocker, llama_stack_processor):
        """Test saving documents with manual chunking workflow."""
        client = self._test_save(mocker, llama_stack_processor["config"])
        client.vector_io.insert.assert_called_once()
        call_args = client.vector_io.insert.call_args
        assert call_args.kwargs["vector_db_id"] == mock.sentinel.index
        assert "chunks" in call_args.kwargs
        assert len(call_args.kwargs["chunks"]) == 1

    def test_save_auto_chunking(self, mocker, llama_stack_processor):
        """Test saving documents with automatic chunking workflow."""
        config = llama_stack_processor["config"]
        config.manual_chunking = False
        client = self._test_save(mocker, config)
        client.tool_runtime.rag_tool.insert.assert_called_once()
        call_args = client.tool_runtime.rag_tool.insert.call_args
        assert call_args.kwargs["vector_db_id"] == mock.sentinel.index
        assert "documents" in call_args.kwargs
        assert len(call_args.kwargs["documents"]) == 1
        assert call_args.kwargs["chunk_size_in_tokens"] == 380

    def test_calculate_embeddings(self, mocker, llama_stack_processor):
        """Test _calculate_embeddings method formats metadata and calculates embeddings."""
        doc = document_processor._LlamaStackDB(llama_stack_processor["config"])
        client = mocker.Mock()
        mock_embedding_response = mocker.Mock()
        mock_embedding_response.embeddings = [[0.5] * 768]
        client.inference.embeddings.return_value = mock_embedding_response

        documents = [
            {
                "content": "test content",
                "embed_metadata": {
                    "title": "Test Title",
                    "docs_url": "https://example.com",
                },
            }
        ]

        doc._calculate_embeddings(client, documents)

        # Verify embed_metadata was removed
        assert "embed_metadata" not in documents[0]
        # Verify embedding was added
        assert "embedding" in documents[0]
        assert documents[0]["embedding"] == [0.5] * 768
        # Verify client.inference.embeddings was called with correct data
        client.inference.embeddings.assert_called_once()
        call_args = client.inference.embeddings.call_args
        assert call_args.kwargs["model_id"] == llama_stack_processor["model_name"]
        assert len(call_args.kwargs["contents"]) == 1
        # Verify the formatted data includes metadata and content
        formatted_data = call_args.kwargs["contents"][0]
        assert "title: Test Title" in formatted_data
        assert "docs_url: https://example.com" in formatted_data
        assert "test content" in formatted_data
        assert formatted_data.startswith("title: Test Title")

    def test_calculate_embeddings_empty_metadata(self, mocker, llama_stack_processor):
        """Test _calculate_embeddings with empty metadata."""
        doc = document_processor._LlamaStackDB(llama_stack_processor["config"])
        client = mocker.Mock()
        mock_embedding_response = mocker.Mock()
        mock_embedding_response.embeddings = [[0.3] * 768]
        client.inference.embeddings.return_value = mock_embedding_response

        documents = [
            {
                "content": "test content",
                "embed_metadata": {},
            }
        ]

        doc._calculate_embeddings(client, documents)

        # Verify embedding was added
        assert "embedding" in documents[0]
        # Verify the formatted data only contains content (empty metadata_str)
        call_args = client.inference.embeddings.call_args
        formatted_data = call_args.kwargs["contents"][0]
        assert formatted_data == "\n\ntest content"

    def test_add_docs_exclude_embed_metadata(self, mocker, llama_stack_processor):
        """Test that exclude_embed_metadata removes keys from embed_metadata."""
        config = llama_stack_processor["config"]
        config.exclude_embed_metadata = ["docs_url", "title"]
        doc = document_processor._LlamaStackDB(config)
        nodes = [
            mocker.Mock(
                spec=TextNode,
                ref_doc_id=1,
                id_=3,
                text="test",
                metadata={
                    "title": "Test Title",
                    "docs_url": "https://example.com",
                    "author": "Test Author",
                },
            )
        ]
        mocker.patch.object(doc, "_split_and_filter", return_value=nodes)

        doc.add_docs([mocker.Mock()])

        assert len(doc.documents) == 1
        # embed_metadata should not contain excluded keys
        assert "docs_url" not in doc.documents[0]["embed_metadata"]
        assert "title" not in doc.documents[0]["embed_metadata"]
        assert "author" in doc.documents[0]["embed_metadata"]
        # llm_metadata should still contain all keys (except those in exclude_llm_metadata)
        assert "title" in doc.documents[0]["metadata"]
        assert "docs_url" in doc.documents[0]["metadata"]
        assert "author" in doc.documents[0]["metadata"]

    def test_add_docs_exclude_llm_metadata(self, mocker, llama_stack_processor):
        """Test that exclude_llm_metadata removes keys from llm_metadata."""
        config = llama_stack_processor["config"]
        config.exclude_llm_metadata = ["docs_url", "author"]
        doc = document_processor._LlamaStackDB(config)
        nodes = [
            mocker.Mock(
                spec=TextNode,
                ref_doc_id=1,
                id_=3,
                text="test",
                metadata={
                    "title": "Test Title",
                    "docs_url": "https://example.com",
                    "author": "Test Author",
                },
            )
        ]
        mocker.patch.object(doc, "_split_and_filter", return_value=nodes)

        doc.add_docs([mocker.Mock()])

        assert len(doc.documents) == 1
        # llm_metadata should not contain excluded keys
        assert "docs_url" not in doc.documents[0]["metadata"]
        assert "author" not in doc.documents[0]["metadata"]
        assert "title" in doc.documents[0]["metadata"]
        # embed_metadata should still contain all keys (except those in exclude_embed_metadata)
        assert "title" in doc.documents[0]["embed_metadata"]
        assert "docs_url" in doc.documents[0]["embed_metadata"]
        assert "author" in doc.documents[0]["embed_metadata"]

    def test_add_docs_exclude_both_metadata(self, mocker, llama_stack_processor):
        """Test that both exclude_embed_metadata and exclude_llm_metadata work together."""
        config = llama_stack_processor["config"]
        config.exclude_embed_metadata = ["docs_url"]
        config.exclude_llm_metadata = ["author"]
        doc = document_processor._LlamaStackDB(config)
        nodes = [
            mocker.Mock(
                spec=TextNode,
                ref_doc_id=1,
                id_=3,
                text="test",
                metadata={
                    "title": "Test Title",
                    "docs_url": "https://example.com",
                    "author": "Test Author",
                },
            )
        ]
        mocker.patch.object(doc, "_split_and_filter", return_value=nodes)

        doc.add_docs([mocker.Mock()])

        assert len(doc.documents) == 1
        # embed_metadata should exclude docs_url
        assert "docs_url" not in doc.documents[0]["embed_metadata"]
        assert "title" in doc.documents[0]["embed_metadata"]
        assert "author" in doc.documents[0]["embed_metadata"]
        # llm_metadata should exclude author
        assert "author" not in doc.documents[0]["metadata"]
        assert "title" in doc.documents[0]["metadata"]
        assert "docs_url" in doc.documents[0]["metadata"]

    def test_calculate_embeddings_multiple_documents(
        self, mocker, llama_stack_processor
    ):
        """Test _calculate_embeddings with multiple documents."""
        doc = document_processor._LlamaStackDB(llama_stack_processor["config"])
        client = mocker.Mock()
        mock_embedding_response = mocker.Mock()
        # Return different embeddings for each call
        mock_embedding_response.embeddings = [[0.1] * 768]
        client.inference.embeddings.side_effect = [
            mocker.Mock(embeddings=[[0.1] * 768]),
            mocker.Mock(embeddings=[[0.2] * 768]),
        ]

        documents = [
            {
                "content": "content 1",
                "embed_metadata": {"title": "Title 1"},
            },
            {
                "content": "content 2",
                "embed_metadata": {"title": "Title 2"},
            },
        ]

        doc._calculate_embeddings(client, documents)

        # Verify both documents have embeddings
        assert documents[0]["embedding"] == [0.1] * 768
        assert documents[1]["embedding"] == [0.2] * 768
        # Verify client was called twice
        assert client.inference.embeddings.call_count == 2
        # Verify embed_metadata was removed from both
        assert "embed_metadata" not in documents[0]
        assert "embed_metadata" not in documents[1]
