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

import pytest
from llama_index.core import Document
from llama_index.core.schema import TextNode

from lightspeed_rag_content import document_processor
from tests.conftest import RagMockEmbedding


@pytest.fixture
def doc_processor(mocker):
    """Fixture for DocumentProcessor tests using Llama Index."""
    mocker.patch.object(
        document_processor, "HuggingFaceEmbedding", new=RagMockEmbedding
    )
    mocker.patch.object(document_processor, "SentenceTransformer")
    mocker.patch("os.path.exists", return_value=True)

    chunk_size = 380
    chunk_overlap = 0
    model_name = "sentence-transformers/all-mpnet-base-v2"
    embeddings_model_dir = "./embeddings_model"
    num_workers = 10

    processor = document_processor.DocumentProcessor(
        chunk_size,
        chunk_overlap,
        model_name,
        Path(embeddings_model_dir),
        num_workers,
    )
    return {
        "processor": processor,
        "model_name": model_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "num_workers": num_workers,
        "embeddings_model_dir": embeddings_model_dir,
    }


class TestDocumentProcessorLlamaIndex:
    """Test the Document Processor using the Llama Index."""

    def test__got_whitespace_false(self, doc_processor):
        """Test that _got_whitespace returns False for text without whitespace."""
        text = "NoWhitespace"
        result = doc_processor["processor"].db._got_whitespace(text)
        assert not result

    def test__got_whitespace_true(self, doc_processor):
        """Test that _got_whitespace returns True for text containing whitespace."""
        text = "Got whitespace"
        result = doc_processor["processor"].db._got_whitespace(text)
        assert result

    def test__filter_out_invalid_nodes(self, doc_processor):
        """Test that _filter_out_invalid_nodes only returns nodes with whitespace."""
        fake_node_0 = mock.Mock(spec=TextNode)
        fake_node_1 = mock.Mock(spec=TextNode)
        fake_node_0.text = "Got whitespace"
        fake_node_1.text = "NoWhitespace"

        result = doc_processor["processor"].db._filter_out_invalid_nodes(
            [fake_node_0, fake_node_1]
        )
        assert result == [fake_node_0]

    def test__save_index(self, mocker, doc_processor):
        """Test that _save_index sets index ID and persists the storage context."""
        mock_vector_index = mocker.patch.object(document_processor, "VectorStoreIndex")
        fake_index = mock_vector_index.return_value

        doc_processor["processor"].db._save_index("fake-index", "/fake/path")

        fake_index.set_index_id.assert_called_once_with("fake-index")
        fake_index.storage_context.persist.assert_called_once_with(
            persist_dir="/fake/path"
        )

    def test__save_metadata(self, mocker, doc_processor):
        """Test that _save_metadata writes correct metadata to JSON file."""
        mock_dumps = mocker.patch.object(document_processor.json, "dumps")
        mock_file = mocker.patch("builtins.open", new_callable=mocker.mock_open)

        doc_processor["processor"].db._save_metadata(
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
            "embedding-model": doc_processor["model_name"],
            "index-id": "fake-index",
            "vector-db": "faiss.IndexFlatIP",
            "embedding-dimension": mock.ANY,
            "chunk": doc_processor["chunk_size"],
            "overlap": doc_processor["chunk_overlap"],
            "total-embedded-files": mock.sentinel.embedded_files,
        }
        mock_dumps.assert_called_once_with(expected_dict)

    def test_process(self, mocker, doc_processor):
        """Test that process method loads documents and filters nodes correctly."""
        mock_dir_reader = mocker.patch.object(
            document_processor, "SimpleDirectoryReader"
        )
        reader = mock_dir_reader.return_value
        reader.load_data.return_value = ["doc0", "doc1", "doc3"]
        fake_metadata = mocker.MagicMock()
        fake_good_nodes = [mocker.Mock(), mocker.Mock()]

        mock_filter = mocker.patch.object(
            doc_processor["processor"].db.__class__,
            "_filter_out_invalid_nodes",
            return_value=fake_good_nodes,
        )
        mock_get_nodes = mocker.patch.object(
            document_processor.Settings.text_splitter.__class__,
            "get_nodes_from_documents",
        )

        doc_processor["processor"].process(Path("/fake/path/docs"), fake_metadata)

        mock_filter.assert_called_once_with(mock_get_nodes.return_value)
        reader.load_data.assert_called_once_with(
            num_workers=doc_processor["num_workers"]
        )
        assert doc_processor["processor"].db._good_nodes == fake_good_nodes
        assert doc_processor["processor"]._num_embedded_files == 3

    def test_process_drop_unreachable(self, mocker, doc_processor):
        """Test that process method drops unreachable documents when action is drop."""
        mock_dir_reader = mocker.patch.object(
            document_processor, "SimpleDirectoryReader"
        )
        reader = mock_dir_reader.return_value
        reader.load_data.return_value = [
            Document(text="doc0", metadata={"url_reachable": False}),
            Document(text="doc1", metadata={"url_reachable": True}),
            Document(text="doc2", metadata={"url_reachable": False}),
        ]
        fake_metadata = mocker.MagicMock()
        fake_good_nodes = [mocker.Mock(), mocker.Mock()]

        mocker.patch.object(
            doc_processor["processor"].db.__class__,
            "_filter_out_invalid_nodes",
            return_value=fake_good_nodes,
        )

        doc_processor["processor"].process(
            Path("/fake/path/docs"), fake_metadata, unreachable_action="drop"
        )
        assert doc_processor["processor"]._num_embedded_files == 1

    def test_process_fail_unreachable(self, mocker, doc_processor):
        """Test that process raises RuntimeError for unreachable documents when action is fail."""
        mock_dir_reader = mocker.patch.object(
            document_processor, "SimpleDirectoryReader"
        )
        reader = mock_dir_reader.return_value
        reader.load_data.return_value = [
            Document(text="doc0", metadata={"url_reachable": False})
        ]
        fake_metadata = mocker.MagicMock()
        fake_good_nodes = [mocker.Mock(), mocker.Mock()]

        mocker.patch.object(
            doc_processor["processor"].db.__class__,
            "_filter_out_invalid_nodes",
            return_value=fake_good_nodes,
        )
        with pytest.raises(RuntimeError):
            doc_processor["processor"].process(
                Path("/fake/path/docs"), fake_metadata, unreachable_action="fail"
            )

    def test_save(self, mocker, doc_processor):
        """Test that save method calls both _save_index and _save_metadata."""
        mock_index = mocker.patch.object(doc_processor["processor"].db, "_save_index")
        mock_md = mocker.patch.object(doc_processor["processor"].db, "_save_metadata")

        doc_processor["processor"].save("fake-index", "/fake/output_dir")

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
    def test_pgvector(self, doc_processor):
        """Test that DocumentProcessor initializes successfully with postgres vector store."""
        proc = document_processor.DocumentProcessor(
            doc_processor["chunk_size"],
            doc_processor["chunk_overlap"],
            doc_processor["model_name"],
            Path(doc_processor["embeddings_model_dir"]),
            doc_processor["num_workers"],
            "postgres",
        )
        assert proc is not None

    def test_invalid_vector_store_type(self, doc_processor):
        """Test that DocumentProcessor raises RuntimeError for invalid vector store type."""
        with pytest.raises(RuntimeError):
            document_processor.DocumentProcessor(
                doc_processor["chunk_size"],
                doc_processor["chunk_overlap"],
                doc_processor["model_name"],
                Path(doc_processor["embeddings_model_dir"]),
                doc_processor["num_workers"],
                "nonexisting",
            )

    def test_exclude_metadata_sets_keys(self, mocker):
        """Test that exclude_metadata sets excluded_embed_metadata_keys and excluded_llm_metadata_keys."""
        mocker.patch.object(
            document_processor, "HuggingFaceEmbedding", new=RagMockEmbedding
        )
        mocker.patch("os.path.exists", return_value=True)

        exclude_embed = ["file_path", "url"]
        exclude_llm = ["file_path", "url_reachable"]

        processor = document_processor.DocumentProcessor(
            380,
            0,
            "sentence-transformers/all-mpnet-base-v2",
            Path("./embeddings_model"),
            10,
            exclude_embed_metadata=exclude_embed,
            exclude_llm_metadata=exclude_llm,
        )

        # Create mock nodes
        node1 = mock.Mock(spec=TextNode)
        node1.excluded_embed_metadata_keys = None
        node1.excluded_llm_metadata_keys = None
        node1.metadata = {"file_path": "/path", "url": "https://example.com"}

        node2 = mock.Mock(spec=TextNode)
        node2.excluded_embed_metadata_keys = None
        node2.excluded_llm_metadata_keys = None
        node2.metadata = {"file_path": "/path2", "title": "Test"}

        nodes = [node1, node2]

        processor.db.exclude_metadata(nodes)

        assert node1.excluded_embed_metadata_keys == exclude_embed
        assert node1.excluded_llm_metadata_keys == exclude_llm
        assert node2.excluded_embed_metadata_keys == exclude_embed
        assert node2.excluded_llm_metadata_keys == exclude_llm

    def test_exclude_metadata_overrides_model_copy(self, mocker):
        """Test that exclude_metadata overrides model_copy method on nodes."""
        mocker.patch.object(
            document_processor, "HuggingFaceEmbedding", new=RagMockEmbedding
        )
        mocker.patch("os.path.exists", return_value=True)

        exclude_embed = ["file_path"]
        exclude_llm = ["file_path", "url_reachable"]

        processor = document_processor.DocumentProcessor(
            380,
            0,
            "sentence-transformers/all-mpnet-base-v2",
            Path("./embeddings_model"),
            10,
            exclude_embed_metadata=exclude_embed,
            exclude_llm_metadata=exclude_llm,
        )

        # Create a real TextNode to test model_copy override
        node = TextNode(
            text="Test content",
            metadata={
                "file_path": "/path/to/file",
                "url_reachable": True,
                "title": "Test Document",
            },
        )

        original_model_copy = node.model_copy

        processor.db.exclude_metadata([node])

        # Verify model_copy was overridden
        assert node.model_copy != original_model_copy
        # Verify it's a bound method
        assert hasattr(node.model_copy, "__self__")

    def test__model_copy_excluding_llm_metadata(self, mocker):
        """Test that _model_copy_excluding_llm_metadata removes excluded metadata."""
        mocker.patch.object(
            document_processor, "HuggingFaceEmbedding", new=RagMockEmbedding
        )
        mocker.patch("os.path.exists", return_value=True)

        exclude_llm = ["file_path", "url_reachable"]

        processor = document_processor.DocumentProcessor(
            380,
            0,
            "sentence-transformers/all-mpnet-base-v2",
            Path("./embeddings_model"),
            10,
            exclude_llm_metadata=exclude_llm,
        )

        node = TextNode(
            text="Test content",
            metadata={
                "file_path": "/path/to/file",
                "url_reachable": True,
                "title": "Test Document",
                "url": "https://example.com",
            },
        )
        node.excluded_llm_metadata_keys = exclude_llm

        # Call the method directly
        result = processor.db._model_copy_excluding_llm_metadata(node)

        # Verify excluded keys are removed
        assert "file_path" not in result.metadata
        assert "url_reachable" not in result.metadata
        # Verify non-excluded keys remain
        assert "title" in result.metadata
        assert "url" in result.metadata
        assert result.metadata["title"] == "Test Document"
        assert result.metadata["url"] == "https://example.com"
        # Verify text is preserved
        assert result.text == "Test content"

    def test_add_docs_calls_exclude_metadata(self, mocker, doc_processor):
        """Test that add_docs calls exclude_metadata on nodes."""
        mock_exclude = mocker.patch.object(
            doc_processor["processor"].db, "exclude_metadata"
        )
        mock_split = mocker.patch.object(
            doc_processor["processor"].db,
            "_split_and_filter",
            return_value=[mock.Mock(spec=TextNode), mock.Mock(spec=TextNode)],
        )

        docs = [Document(text="doc1"), Document(text="doc2")]
        doc_processor["processor"].db.add_docs(docs)

        mock_split.assert_called_once_with(docs)
        # exclude_metadata should be called with the good nodes
        mock_exclude.assert_called_once()
        assert len(mock_exclude.call_args[0][0]) == 2

    def test_exclude_metadata_defaults_to_none(self, mocker):
        """Test that exclude_metadata parameters default to None when not provided."""
        mocker.patch.object(
            document_processor, "HuggingFaceEmbedding", new=RagMockEmbedding
        )
        mocker.patch("os.path.exists", return_value=True)

        processor = document_processor.DocumentProcessor(
            380,
            0,
            "sentence-transformers/all-mpnet-base-v2",
            Path("./embeddings_model"),
            10,
        )

        assert processor.config.exclude_embed_metadata is None
        assert processor.config.exclude_llm_metadata is None

    def test_exclude_metadata_with_custom_values(self, mocker):
        """Test that exclude_metadata parameters are stored correctly when provided."""
        mocker.patch.object(
            document_processor, "HuggingFaceEmbedding", new=RagMockEmbedding
        )
        mocker.patch("os.path.exists", return_value=True)

        exclude_embed = ["custom_key1", "custom_key2"]
        exclude_llm = ["custom_key3"]

        processor = document_processor.DocumentProcessor(
            380,
            0,
            "sentence-transformers/all-mpnet-base-v2",
            Path("./embeddings_model"),
            10,
            exclude_embed_metadata=exclude_embed,
            exclude_llm_metadata=exclude_llm,
        )

        assert processor.config.exclude_embed_metadata == exclude_embed
        assert processor.config.exclude_llm_metadata == exclude_llm
