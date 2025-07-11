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

from llama_index.core.schema import TextNode

from lightspeed_rag_content import document_processor
from tests import utils


class TestDocumentProcessorLlamaStack(utils.TestCase):
    """Test cases for the _LlamaStackDB document processor class.

    This test class verifies the functionality of the Llama Stack database
    integration for document processing, including initialization, configuration
    generation, document addition, and persistence operations.
    """

    def setUp(self):
        """Set up test fixtures and mock objects for each test."""
        self.patch_object(
            document_processor, "HuggingFaceEmbedding", new=utils.RagMockEmbedding
        )
        st = self.patch_object(document_processor, "SentenceTransformer")
        st.return_value.get_sentence_embedding_dimension.return_value = 768
        # Default to the embedding being a model reference and not a directory
        self.exists = self.patch("os.path.exists", return_value=False)

        self.get_nodes = self.patch_object(
            document_processor.Settings.text_splitter.__class__,
            "get_nodes_from_documents",
        )

        self.model_name = "sentence-transformers/all-mpnet-base-v2"
        self.config = document_processor._Config(
            chunk_size=380,
            chunk_overlap=0,
            model_name=self.model_name,
            embeddings_model_dir="",
            vector_store_type="llamastack-faiss",
            embedding_dimension=None,
            manual_chunking=True,
        )

    @mock.patch.object(document_processor.tempfile, "TemporaryDirectory")
    def test_init(self, temp_dir):
        """Test basic initialization of _LlamaStackDB with default settings."""
        temp_dir.return_value.name = "temp_dir"
        doc = document_processor._LlamaStackDB(self.config)
        self.assertEqual(self.config, doc.config)
        self.assertEqual(self.model_name, doc.model_name_or_dir)
        self.assertEqual(768, self.config.embedding_dimension)
        self.assertEqual("faiss_store.db", doc.db_filename)
        self.assertEqual(doc.document_class.__name__, "RAGDocument")
        self.assertEqual(doc.client_class.__name__, "LlamaStackAsLibraryClient")
        self.assertListEqual([], doc.documents)
        temp_dir.assert_called_once_with(prefix="ls-rag-")
        self.assertIs(temp_dir.return_value, doc.tmp_dir)
        self.assertEqual(
            temp_dir.return_value.name, os.environ["LLAMA_STACK_CONFIG_DIR"]
        )

    @mock.patch.object(document_processor.tempfile, "TemporaryDirectory")
    def test_init_model_path(self, temp_dir):
        """Test initialization when embeddings_model_dir exists as a local path."""
        temp_dir.return_value.name = "temp_dir"
        self.exists.return_value = True
        self.config.embeddings_model_dir = "embeddings_model"
        realpath = self.patch("os.path.realpath")
        self.exists.reset_mock()
        doc = document_processor._LlamaStackDB(self.config)

        self.assertEqual(self.config, doc.config)
        self.exists.assert_called_once_with(self.config.embeddings_model_dir)
        realpath.assert_called_once_with(self.config.embeddings_model_dir)
        self.assertEqual(realpath.return_value, doc.model_name_or_dir)
        self.assertEqual(768, self.config.embedding_dimension)
        self.assertEqual("faiss_store.db", doc.db_filename)
        self.assertEqual(doc.document_class.__name__, "RAGDocument")
        self.assertEqual(doc.client_class.__name__, "LlamaStackAsLibraryClient")
        self.assertListEqual([], doc.documents)
        temp_dir.assert_called_once_with(prefix="ls-rag-")
        self.assertIs(temp_dir.return_value, doc.tmp_dir)
        self.assertEqual(
            temp_dir.return_value.name, os.environ["LLAMA_STACK_CONFIG_DIR"]
        )

    @mock.patch("builtins.open", new_callable=mock.mock_open)
    def test_write_yaml_config_faiss(self, mock_open):
        """Test YAML configuration generation for FAISS vector store backend."""
        doc = document_processor._LlamaStackDB(self.config)

        provider_id = "my_provider_id"
        yaml_file = "yaml_file"
        db_file = "db_file"

        doc.write_yaml_config(provider_id, yaml_file, db_file)

        mock_open.assert_called_once_with(yaml_file, "w", encoding="utf-8")
        expected = f"""version: '2'
image_name: ollama
apis:
  - inference
  - vector_io
  - tool_runtime
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
        {''}
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
"""
        data = mock_open.return_value.write.mock_calls[0].args[0]
        self.assertEqual(expected, data)

    def test_write_yaml_config_sqlitevec(self):
        """Test YAML configuration generation for SQLiteVec vector store backend."""
        self.config.vector_store_type = "llamastack-sqlite-vec"
        doc = document_processor._LlamaStackDB(self.config)

        provider_id = "my_provider_id"
        yaml_file = "yaml_file"
        db_file = "db_file"

        with mock.patch("builtins.open", new_callable=mock.mock_open) as mock_open:
            doc.write_yaml_config(provider_id, yaml_file, db_file)

        mock_open.assert_called_once_with(yaml_file, "w", encoding="utf-8")
        self.maxDiff = None

        expected = f"""version: '2'
image_name: ollama
apis:
  - inference
  - vector_io
  - tool_runtime
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
"""
        data = mock_open.return_value.write.mock_calls[0].args[0]
        self.assertEqual(expected, data)

    @mock.patch.object(document_processor.tempfile, "TemporaryDirectory")
    def test_start_llama_stack(self, temp_dir):
        """Test starting the Llama Stack client."""
        temp_dir.return_value.name = "tempdir"
        doc = document_processor._LlamaStackDB(self.config)
        yaml_file = "yaml_file"

        with mock.patch.object(doc, "client_class") as client:
            res = doc._start_llama_stack(yaml_file)
            self.assertEqual(client.return_value, res)
            client.assert_called_once_with(yaml_file)

        temp_dir.assert_called_once_with(prefix="ls-rag-")
        self.assertEqual(
            temp_dir.return_value.name, os.environ["LLAMA_STACK_CONFIG_DIR"]
        )

    def test_add_docs_manual_chunking(self):
        """Test adding documents with manual chunking enabled.

        Verifies that documents are properly split and filtered before being
        converted to the expected format for manual chunking workflow.
        """
        doc = document_processor._LlamaStackDB(self.config)

        nodes = [
            mock.Mock(
                spec=TextNode,
                ref_doc_id=i,
                id_=i * 3,
                text=str(i),
                metadata={"title": f"title{i}", "docs_url": f"https://redhat.com/{i}"},
            )
            for i in range(1, 3)
        ]
        mock_filter = self.patch_object(doc, "_split_and_filter", return_value=nodes)

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

        self.assertListEqual(expect, doc.documents)

    def test_add_docs_auto_chunking(self):
        """Test adding documents with automatic chunking enabled.

        Verifies that documents are directly converted to document objects
        without manual splitting when auto chunking is configured.
        """
        self.config.manual_chunking = False
        doc = document_processor._LlamaStackDB(self.config)

        fake_out_docs = [mock.Mock(), mock.Mock()]
        doc_class = self.patch_object(doc, "document_class", side_effect=fake_out_docs)
        mock_filter = self.patch_object(doc, "_split_and_filter")

        in_docs = [
            mock.Mock(doc_id=str(i), text=str(i), metadata={"title": f"title{i}"})
            for i in range(1, 3)
        ]

        doc.add_docs(in_docs)

        mock_filter.assert_not_called()
        self.assertEqual(len(in_docs), doc_class.call_count)
        doc_class.assert_has_calls(
            [
                mock.call(
                    document_id=doc.doc_id,
                    content=doc.text,
                    mime_type="text/plain",
                    metadata=doc.metadata,
                )
                for doc in in_docs
            ]
        )
        self.assertListEqual(fake_out_docs, doc.documents)

    def _test_save(self):
        """Set up and verify save functionality for testing.

        Sets up common test fixtures and verifies core save operations
        including YAML config generation and vector DB registration.

        Returns:
            Mock client object for additional assertions by calling tests.
        """
        doc = document_processor._LlamaStackDB(self.config)
        doc.documents = mock.sentinel.documents

        write_cfg = self.patch_object(doc, "write_yaml_config")
        client = self.patch_object(doc, "_start_llama_stack")
        client.inspect.version.return_value = "0.2.15"
        realpath = self.patch(
            "os.path.realpath", return_value="/cwd/out_dir/vector_store.db"
        )

        doc.save(mock.sentinel.index, "out_dir")

        realpath.assert_called_once_with("out_dir/faiss_store.db")
        write_cfg.assert_called_once_with(
            mock.sentinel.index,
            "out_dir/llama-stack.yaml",
            "/cwd/out_dir/vector_store.db",
        )
        # We save the DB information on the yaml config, no need to register it
        client.return_value.vector_dbs.register.assert_not_called()

        return client.return_value

    def test_save_manual_chunking(self):
        """Test saving documents with manual chunking workflow.

        Verifies that chunks are inserted directly via vector_io.insert
        when manual chunking is enabled.
        """
        client = self._test_save()
        client.vector_io.insert.assert_called_once_with(
            vector_db_id=mock.sentinel.index, chunks=mock.sentinel.documents
        )

    def test_save_auto_chunking(self):
        """Test saving documents with automatic chunking workflow.

        Verifies that documents are processed via rag_tool.insert
        when automatic chunking is enabled.
        """
        self.config.manual_chunking = False
        client = self._test_save()
        client.tool_runtime.rag_tool.insert.assert_called_once_with(
            documents=mock.sentinel.documents,
            vector_db_id=mock.sentinel.index,
            chunk_size_in_tokens=380,
        )
