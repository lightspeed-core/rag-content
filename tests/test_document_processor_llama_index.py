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
from pathlib import Path
from unittest import mock

from llama_index.core import Document
from llama_index.core.schema import TextNode

from lightspeed_rag_content import document_processor
from tests import utils


@mock.patch.object(
    document_processor, "HuggingFaceEmbedding", new=utils.RagMockEmbedding
)
class TestDocumentProcessorLlamaIndex(utils.TestCase):
    """Test the Document Processor using the Llama Index."""

    def setUp(self):
        """Set common test prerequisites."""
        self.patch_object(
            document_processor, "HuggingFaceEmbedding", new=utils.RagMockEmbedding
        )
        self.patch_object(document_processor, "SentenceTransformer")
        self.patch("os.path.exists", return_value=True)
        self.chunk_size = 380
        self.chunk_overlap = 0
        self.model_name = "sentence-transformers/all-mpnet-base-v2"
        self.embeddings_model_dir = "./embeddings_model"
        self.num_workers = 10

        self.doc_processor = document_processor.DocumentProcessor(
            self.chunk_size,
            self.chunk_overlap,
            self.model_name,
            Path(self.embeddings_model_dir),
            self.num_workers,
        )

    def test__got_whitespace_false(self):
        """Test that _got_whitespace returns False for text without whitespace."""
        text = "NoWhitespace"

        result = self.doc_processor.db._got_whitespace(text)

        self.assertFalse(result)

    def test__got_whitespace_true(self):
        """Test that _got_whitespace returns True for text containing whitespace."""
        text = "Got whitespace"

        result = self.doc_processor.db._got_whitespace(text)

        self.assertTrue(result)

    def test__filter_out_invalid_nodes(self):
        """Test that _filter_out_invalid_nodes only returns nodes with whitespace."""
        fake_node_0 = mock.Mock(spec=TextNode)
        fake_node_1 = mock.Mock(spec=TextNode)
        fake_node_0.text = "Got whitespace"
        fake_node_1.text = "NoWhitespace"

        result = self.doc_processor.db._filter_out_invalid_nodes(
            [fake_node_0, fake_node_1]
        )

        # Only nodes with whitespaces should be returned
        self.assertEqual([fake_node_0], result)

    @mock.patch.object(document_processor, "VectorStoreIndex")
    def test__save_index(self, mock_vector_index):
        """Test that _save_index sets index ID and persists the storage context."""
        fake_index = mock_vector_index.return_value

        self.doc_processor.db._save_index("fake-index", "/fake/path")

        fake_index.set_index_id.assert_called_once_with("fake-index")
        fake_index.storage_context.persist.assert_called_once_with(
            persist_dir="/fake/path"
        )

    @mock.patch.object(document_processor.json, "dumps")
    @mock.patch("builtins.open", new_callable=mock.mock_open)
    def test__save_metadata(self, mock_file, mock_dumps):
        """Test that _save_metadata writes correct metadata to JSON file."""
        self.doc_processor.db._save_metadata(
            "fake-index",
            "/fake/path",
            mock.sentinel.embedded_files,
            mock.sentinel.exec_time,
        )

        mock_file.assert_called_once_with(
            "/fake/path/metadata.json", "w", encoding="utf-8"
        )
        expected_dict = {
            "execution-time": mock.sentinel.exec_time,
            "llm": "None",
            "embedding-model": self.model_name,
            "index-id": "fake-index",
            "vector-db": "faiss.IndexFlatIP",
            "embedding-dimension": mock.ANY,
            "chunk": self.chunk_size,
            "overlap": self.chunk_overlap,
            "total-embedded-files": mock.sentinel.embedded_files,
        }
        mock_dumps.assert_called_once_with(expected_dict)

    @mock.patch.object(document_processor, "SimpleDirectoryReader")
    def test_process(self, mock_dir_reader):
        """Test that process method loads documents and filters nodes correctly."""
        reader = mock_dir_reader.return_value
        reader.load_data.return_value = ["doc0", "doc1", "doc3"]
        fake_metadata = mock.MagicMock()
        fake_good_nodes = [mock.Mock(), mock.Mock()]

        mock_filter = self.patch_object(
            self.doc_processor.db.__class__,
            "_filter_out_invalid_nodes",
            return_value=fake_good_nodes,
        )

        mock_get_nodes = self.patch_object(
            document_processor.Settings.text_splitter.__class__,
            "get_nodes_from_documents",
        )

        self.doc_processor.process(Path("/fake/path/docs"), fake_metadata)

        mock_filter.assert_called_once_with(mock_get_nodes.return_value)

        reader.load_data.assert_called_once_with(num_workers=self.num_workers)
        self.assertEqual(fake_good_nodes, self.doc_processor.db._good_nodes)
        self.assertEqual(3, self.doc_processor._num_embedded_files)

    @mock.patch.object(document_processor, "SimpleDirectoryReader")
    def test_process_drop_unreachable(self, mock_dir_reader):
        """Test that process method drops unreachable documents when action is drop."""
        reader = mock_dir_reader.return_value
        reader.load_data.return_value = [
            Document(
                text="doc0",
                metadata={"url_reachable": False},  # pyright: ignore[reportCallIssue]
            ),
            Document(
                text="doc1",
                metadata={"url_reachable": True},  # pyright: ignore[reportCallIssue]
            ),
            Document(
                text="doc2",
                metadata={"url_reachable": False},  # pyright: ignore[reportCallIssue]
            ),
        ]
        fake_metadata = mock.MagicMock()
        fake_good_nodes = [mock.Mock(), mock.Mock()]

        self.patch_object(
            self.doc_processor.db.__class__,
            "_filter_out_invalid_nodes",
            return_value=fake_good_nodes,
        )

        self.doc_processor.process(
            Path("/fake/path/docs"), fake_metadata, unreachable_action="drop"
        )

        self.assertEqual(1, self.doc_processor._num_embedded_files)

    @mock.patch.object(document_processor, "SimpleDirectoryReader")
    def test_process_fail_unreachable(self, mock_dir_reader):
        """Test that process raises RuntimeError for unreachable documents when action is fail."""
        reader = mock_dir_reader.return_value
        reader.load_data.return_value = [
            Document(
                text="doc0",
                metadata={"url_reachable": False},  # pyright: ignore[reportCallIssue]
            )
        ]
        fake_metadata = mock.MagicMock()
        fake_good_nodes = [mock.Mock(), mock.Mock()]

        self.patch_object(
            self.doc_processor.db.__class__,
            "_filter_out_invalid_nodes",
            return_value=fake_good_nodes,
        )
        with self.assertRaises(RuntimeError):
            self.doc_processor.process(
                Path("/fake/path/docs"), fake_metadata, unreachable_action="fail"
            )

    def test_save(self):
        """Test that save method calls both _save_index and _save_metadata."""
        mock_index = self.patch_object(self.doc_processor.db, "_save_index")
        mock_md = self.patch_object(self.doc_processor.db, "_save_metadata")

        self.doc_processor.save("fake-index", "/fake/output_dir")

        mock_index.assert_called_once_with("fake-index", "/fake/output_dir")
        mock_md.assert_called_once_with("fake-index", "/fake/output_dir", 0, mock.ANY)

    @mock.patch.dict(
        os.environ,
        {
            "POSTGRES_USER": "postgres",
            "POSTGRES_PASSWORD": "somesecret",
            "POSTGRES_HOST": "localhost",
            "POSTGRES_PORT": "15432",
            "POSTGRES_DATABASE": "postgres",
        },
    )
    def test_pgvector(self):
        """Test that DocumentProcessor initializes successfully with postgres vector store."""
        self.doc_processor = document_processor.DocumentProcessor(
            self.chunk_size,
            self.chunk_overlap,
            self.model_name,
            Path(self.embeddings_model_dir),
            self.num_workers,
            "postgres",
        )
        self.assertIsNotNone(self.doc_processor)

    def test_invalid_vector_store_type(self):
        """Test that DocumentProcessor raises RuntimeError for invalid vector store type."""
        self.assertRaises(
            RuntimeError,
            document_processor.DocumentProcessor,
            self.chunk_size,
            self.chunk_overlap,
            self.model_name,
            Path(self.embeddings_model_dir),
            self.num_workers,
            "nonexisting",
        )
