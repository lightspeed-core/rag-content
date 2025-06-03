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
import unittest
from unittest import mock

from llama_index.core.schema import TextNode
from lightspeed_rag_content import document_processor


# Mock class for HuggingFaceEmbedding
class MockEmbedding:
    def __init__(self, model_name="ABC"):
        pass

    def get_text_embedding(self, text):
        return "ABC"


class TestMetadataProcessor(unittest.TestCase):

    def setUp(self):
        self.chunk_size = 380
        self.chunk_overlap = 0
        self.model_name = "sentence-transformers/all-mpnet-base-v2"
        self.embeddings_model_dir = "./embeddings_model"
        self.num_workers = 10

        # Mock the _get_settings() method
        self.settings_obj = mock.MagicMock()
        self.patcher = mock.patch.object(
            document_processor.DocumentProcessor, "_get_settings"
        )
        self._settings = self.patcher.start()
        self._settings.return_value = self.settings_obj
        self.addCleanup(self.patcher.stop)

        self.doc_processor = document_processor.DocumentProcessor(
            self.chunk_size,
            self.chunk_overlap,
            self.model_name,
            self.embeddings_model_dir,
            self.num_workers,
        )

    def test__got_whitespace_false(self):
        text = "NoWhitespace"

        result = self.doc_processor._got_whitespace(text)

        self.assertFalse(result)

    def test__got_whitespace_true(self):
        text = "Got whitespace"

        result = self.doc_processor._got_whitespace(text)

        self.assertTrue(result)

    def test__filter_out_invalid_nodes(self):
        fake_node_0 = mock.Mock(spec=TextNode)
        fake_node_1 = mock.Mock(spec=TextNode)
        fake_node_0.text = "Got whitespace"
        fake_node_1.text = "NoWhitespace"

        result = self.doc_processor._filter_out_invalid_nodes(
            [fake_node_0, fake_node_1]
        )

        # Only nodes with whitespaces should be returned
        self.assertEqual([fake_node_0], result)

    @mock.patch.object(document_processor, "VectorStoreIndex")
    def test__save_index(self, mock_vector_index):
        fake_index = mock_vector_index.return_value

        self.doc_processor._save_index("fake-index", "/fake/path")

        fake_index.set_index_id.assert_called_once_with("fake-index")
        fake_index.storage_context.persist.assert_called_once_with(
            persist_dir="/fake/path"
        )

    @mock.patch.object(document_processor.json, "dumps")
    @mock.patch("builtins.open", new_callable=mock.mock_open)
    def test__save_metadata(self, mock_file, mock_dumps):
        self.doc_processor._save_metadata("fake-index", "/fake/path")

        mock_file.assert_called_once_with("/fake/path/metadata.json", "w")
        expected_dict = {
            "execution-time": mock.ANY,
            "llm": "None",
            "embedding-model": self.model_name,
            "index-id": "fake-index",
            "vector-db": "faiss.IndexFlatIP",
            "embedding-dimension": mock.ANY,
            "chunk": self.chunk_size,
            "overlap": self.chunk_overlap,
            "total-embedded-files": 0,
        }
        mock_dumps.assert_called_once_with(expected_dict)

    @mock.patch.object(document_processor, "SimpleDirectoryReader")
    def test_process(self, mock_dir_reader):
        reader = mock_dir_reader.return_value
        reader.load_data.return_value = ["doc0", "doc1", "doc3"]
        fake_metadata = mock.MagicMock()
        fake_good_nodes = [mock.Mock(), mock.Mock()]

        with mock.patch.object(
            self.doc_processor, "_filter_out_invalid_nodes"
        ) as mock_filter:
            mock_filter.return_value = fake_good_nodes
            self.doc_processor.process("/fake/path/docs", fake_metadata)

        reader.load_data.assert_called_once_with(num_workers=self.num_workers)
        self.assertEqual(fake_good_nodes, self.doc_processor._good_nodes)
        self.assertEqual(3, self.doc_processor._num_embedded_files)

    def test_save(self):
        with (
            mock.patch.object(self.doc_processor, "_save_index") as mock_index,
            mock.patch.object(self.doc_processor, "_save_metadata") as mock_md,
        ):
            self.doc_processor.save("fake-index", "/fake/output_dir")

        mock_index.assert_called_once_with("fake-index", "/fake/output_dir")
        mock_md.assert_called_once_with("fake-index", "/fake/output_dir")

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
    @mock.patch(
        "lightspeed_rag_content.document_processor.HuggingFaceEmbedding",
        new=MockEmbedding,
    )
    def test_pgvector(self):
        self.patcher.stop()  # Remove the mock on the _get_settings() method
        self.doc_processor = document_processor.DocumentProcessor(
            self.chunk_size,
            self.chunk_overlap,
            self.model_name,
            self.embeddings_model_dir,
            self.num_workers,
            "postgres",
        )
        self.assertIsNotNone(self.doc_processor)

    @mock.patch(
        "lightspeed_rag_content.document_processor.HuggingFaceEmbedding",
        new=MockEmbedding,
    )
    def test_invalid_vector_store_type(self):
        self.patcher.stop()  # Remove the mock on the _get_settings() method
        self.assertRaises(
            RuntimeError,
            document_processor.DocumentProcessor,
            self.chunk_size,
            self.chunk_overlap,
            self.model_name,
            self.embeddings_model_dir,
            self.num_workers,
            "nonexisting",
        )
