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
from llama_index.core.schema import Node, TextNode

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

    def test__valid_text_node(self, doc_processor):
        """Test that valid text node checks for got whitespace on non markdown chunker."""
        db = doc_processor["processor"].db
        db.config.doc_type = "plain"

        with (
            mock.patch.object(db, "_got_whitespace") as mock_got_ws,
            mock.patch.object(db, "_got_non_headers") as mock_got_nh,
        ):
            text = "NoWhitespace"
            res = db._valid_text_node(text)
            assert res is mock_got_ws.return_value
            mock_got_ws.assert_called_once_with(text)
            mock_got_nh.assert_not_called()

    def test__valid_text_node_markdown_non_headers_true(self, doc_processor):
        """Test that text node is valid when markdown has non header content."""
        db = doc_processor["processor"].db
        db.config.doc_type = "markdown"

        with (
            mock.patch.object(db, "_got_non_headers", return_value=True) as mock_got_nh,
            mock.patch.object(db, "_got_whitespace") as mock_got_ws,
        ):
            text = "# Header\nActual content here"
            res = db._valid_text_node(text)
            assert res is mock_got_ws.return_value
            mock_got_nh.assert_called_once_with(text)
            mock_got_ws.assert_called_once_with(text)

    def test__valid_text_node_markdown_non_headers_false(self, doc_processor):
        """Test that text node is invalid when markdown only has headers."""
        db = doc_processor["processor"].db
        db.config.doc_type = "markdown"

        with (
            mock.patch.object(
                db, "_got_non_headers", return_value=False
            ) as mock_got_nh,
            mock.patch.object(db, "_got_whitespace") as mock_got_ws,
        ):
            text = "# Header1\n# Header2\n\n"
            res = db._valid_text_node(text)
            assert res is False
            mock_got_nh.assert_called_once_with(text)
            mock_got_ws.assert_not_called()

    @pytest.mark.parametrize(
        "text",
        [
            "# Header\nSome content",  # Header followed by content
            "# Header1\n# Header2\nAlso here",  # Multiple headers, then content
            "No headers, just content",  # No headers, just content
            "# H\n# H2\n\tThis is non-header",  # Tabs and spaces before content
        ],
    )
    def test__got_non_headers_with_content(self, doc_processor, text):
        """Test we detect when markdown has something beside headers."""
        db = doc_processor["processor"].db
        assert db._got_non_headers(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "# Only header",
            "# Another header\n## Subheader",
            "# Header with space \n",
            "## \n#",
            "#Header1\n#Header2\n#Header3",
            "#    ",  # header with whitespace
            "    # Header with leading space",
            "   \n\t  \n",  # only whitespaces
        ],
    )
    def test__got_non_headers_only_headers(self, doc_processor, text):
        """Test we detect when markdown only has headers."""
        db = doc_processor["processor"].db
        assert db._got_non_headers(text) is False

    def test__filter_out_invalid_nodes(self, doc_processor):
        """Test that _filter_out_invalid_nodes checks for validity of text nodes."""
        fake_text_node_0 = mock.Mock(spec=TextNode, text="fake_text_node_0")
        fake_text_node_1 = mock.Mock(spec=TextNode, text="fake_text_node_1")
        fake_node_2 = mock.Mock(spec=Node)

        db = doc_processor["processor"].db
        with mock.patch.object(
            db, "_valid_text_node", side_effect=(True, False)
        ) as valid_tn_mock:
            result = db._filter_out_invalid_nodes(
                [fake_text_node_0, fake_text_node_1, fake_node_2]
            )

            expected_calls = [
                mock.call(fake_text_node_0.text),
                mock.call(fake_text_node_1.text),
            ]
            valid_tn_mock.assert_has_calls(expected_calls)
            assert len(expected_calls) == valid_tn_mock.call_count

        assert result == [fake_text_node_0]

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
