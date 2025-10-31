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
        doc.documents = mock.sentinel.documents

        write_cfg = mocker.patch.object(doc, "write_yaml_config")
        client = mocker.patch.object(doc, "_start_llama_stack")
        client.inspect.version.return_value = "0.2.15"
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
        client.vector_io.insert.assert_called_once_with(
            vector_db_id=mock.sentinel.index, chunks=mock.sentinel.documents
        )

    def test_save_auto_chunking(self, mocker, llama_stack_processor):
        """Test saving documents with automatic chunking workflow."""
        config = llama_stack_processor["config"]
        config.manual_chunking = False
        client = self._test_save(mocker, config)
        client.tool_runtime.rag_tool.insert.assert_called_once_with(
            documents=mock.sentinel.documents,
            vector_db_id=mock.sentinel.index,
            chunk_size_in_tokens=380,
        )
