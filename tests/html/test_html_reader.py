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
"""Tests for HTMLReader class."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lightspeed_rag_content.html.html_reader import (
    HTMLReader,
    convert_html_file_to_markdown,
    convert_html_string_to_markdown,
)


@pytest.fixture
def mock_docling(mocker):
    """Mock docling DocumentConverter and related classes."""
    mock_document = MagicMock()
    mock_document.export_to_markdown.return_value = "# Test Title\n\nTest content."

    mock_result = MagicMock()
    mock_result.document = mock_document

    mock_converter = MagicMock()
    mock_converter.convert.return_value = mock_result
    mock_converter.convert_string.return_value = mock_result

    mock_converter_class = mocker.patch(
        "lightspeed_rag_content.html.html_reader.DocumentConverter",
        return_value=mock_converter,
    )

    return {
        "converter": mock_converter,
        "converter_class": mock_converter_class,
        "result": mock_result,
        "document": mock_document,
    }


@pytest.fixture
def html_file(tmp_path):
    """Create a temporary HTML file for testing."""
    html_content = "<html><body><h1>Test</h1><p>Content</p></body></html>"
    file_path = tmp_path / "test.html"
    file_path.write_text(html_content, encoding="utf-8")
    return file_path


class TestHTMLReader:
    """Tests for HTMLReader class."""

    def test_init_creates_converter(self, mock_docling):
        """Test that HTMLReader initializes with a DocumentConverter."""
        reader = HTMLReader()
        assert reader.converter is not None
        mock_docling["converter_class"].assert_called_once()

    def test_load_data_file_not_found(self, mock_docling):
        """Test that FileNotFoundError is raised for missing files."""
        reader = HTMLReader()
        with pytest.raises(FileNotFoundError, match="HTML file not found"):
            reader.load_data(Path("/nonexistent/file.html"))

    def test_load_data_successful(self, mock_docling, html_file):
        """Test successful HTML file loading and conversion."""
        reader = HTMLReader()
        documents = reader.load_data(html_file)

        assert len(documents) == 1
        assert documents[0].text == "# Test Title\n\nTest content."
        assert documents[0].metadata["file_path"] == str(html_file)
        assert documents[0].metadata["file_name"] == "test.html"
        mock_docling["converter"].convert.assert_called_once_with(str(html_file))

    def test_load_data_with_extra_info(self, mock_docling, html_file):
        """Test that extra_info is included in document metadata."""
        reader = HTMLReader()
        extra_info = {"custom_key": "custom_value"}
        documents = reader.load_data(html_file, extra_info=extra_info)

        assert documents[0].metadata["custom_key"] == "custom_value"
        assert documents[0].metadata["file_path"] == str(html_file)

    def test_load_data_does_not_mutate_extra_info(self, mock_docling, html_file):
        """Test that the caller's extra_info dict is not mutated."""
        reader = HTMLReader()
        extra_info = {"custom_key": "custom_value"}
        original_keys = set(extra_info.keys())

        reader.load_data(html_file, extra_info=extra_info)

        # Verify extra_info was not mutated - should still have only original keys
        assert set(extra_info.keys()) == original_keys
        assert "file_path" not in extra_info
        assert "file_name" not in extra_info

    def test_load_data_conversion_error(self, mock_docling, html_file):
        """Test that RuntimeError is raised on conversion failure."""
        mock_docling["converter"].convert.side_effect = Exception("Conversion failed")
        reader = HTMLReader()

        with pytest.raises(RuntimeError, match="Failed to convert HTML file"):
            reader.load_data(html_file)


class TestConvertHtmlFileToMarkdown:
    """Tests for convert_html_file_to_markdown function."""

    def test_successful_conversion(self, mock_docling, html_file):
        """Test successful file conversion."""
        result = convert_html_file_to_markdown(html_file)
        assert result == "# Test Title\n\nTest content."

    def test_file_not_found(self, mock_docling):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            convert_html_file_to_markdown("/nonexistent/file.html")


class TestConvertHtmlStringToMarkdown:
    """Tests for convert_html_string_to_markdown function."""

    def test_empty_string_returns_empty(self):
        """Test that empty HTML content returns empty string."""
        result = convert_html_string_to_markdown("")
        assert result == ""

    def test_successful_conversion(self, mock_docling):
        """Test successful string conversion."""
        result = convert_html_string_to_markdown("<h1>Test</h1>")
        assert result == "# Test Title\n\nTest content."
        mock_docling["converter"].convert_string.assert_called_once()

    def test_conversion_with_document_name(self, mock_docling):
        """Test conversion with custom document name."""
        convert_html_string_to_markdown("<h1>Test</h1>", document_name="mydoc")
        call_kwargs = mock_docling["converter"].convert_string.call_args
        assert call_kwargs.kwargs["name"] == "mydoc"

    def test_conversion_error(self, mock_docling):
        """Test that RuntimeError is raised on conversion failure."""
        mock_docling["converter"].convert_string.side_effect = Exception("Parse error")

        with pytest.raises(RuntimeError, match="Failed to convert HTML string"):
            convert_html_string_to_markdown("<h1>Test</h1>")
