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
"""Tests for PDFReader class.

Two layers of coverage:

* Fast, fully-mocked tests exercise the plumbing around docling (path checks,
  metadata handling, error wrapping, the R7 empty-output warning).
* Real, un-mocked tests run the committed fixture PDF through actual docling to
  prove the conversion genuinely works (skipped when docling models are not
  cached). The HTML reader has no equivalent, which is why "does it work?" was
  previously unanswerable from its tests.
"""

import logging
from pathlib import Path

import pytest

from lightspeed_rag_content.pdf.pdf_reader import (
    EMPTY_OUTPUT_THRESHOLD,
    PDFReader,
    convert_pdf_file_to_markdown,
)
from tests.pdf.conftest import FIXTURE_HEADING, FIXTURE_MARKERS, requires_docling_models

# A body long enough to clear EMPTY_OUTPUT_THRESHOLD so the success-path tests do
# not incidentally trip the R7 warning.
SAMPLE_MARKDOWN = "# Test Title\n\nThis is a sufficiently long body of converted PDF content."


@pytest.fixture(name="mock_docling")
def mock_docling_fixture(mocker):
    """Mock docling's DocumentConverter inside the pdf_reader module."""
    mock_document = mocker.MagicMock()
    mock_document.export_to_markdown.return_value = SAMPLE_MARKDOWN

    mock_result = mocker.MagicMock()
    mock_result.document = mock_document

    mock_converter = mocker.MagicMock()
    mock_converter.convert.return_value = mock_result

    mock_converter_class = mocker.patch(
        "lightspeed_rag_content.pdf.pdf_reader.DocumentConverter",
        return_value=mock_converter,
    )
    return {
        "converter": mock_converter,
        "converter_class": mock_converter_class,
        "result": mock_result,
        "document": mock_document,
    }


@pytest.fixture(name="pdf_file")
def pdf_file_fixture(tmp_path):
    """Create a placeholder PDF file (contents irrelevant; docling is mocked)."""
    file_path = tmp_path / "test.pdf"
    file_path.write_bytes(b"%PDF-1.4 placeholder")
    return file_path


class TestPDFReader:
    """Tests for the PDFReader class (docling mocked)."""

    def test_init_creates_converter(self, mock_docling):
        """PDFReader initializes a DocumentConverter."""
        reader = PDFReader()
        assert reader.converter is not None
        mock_docling["converter_class"].assert_called_once()

    def test_load_data_file_not_found(self, mock_docling):
        """FileNotFoundError is raised for a missing file."""
        reader = PDFReader()
        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            reader.load_data(Path("/nonexistent/file.pdf"))

    def test_load_data_successful(self, mock_docling, pdf_file):
        """A successful conversion yields one Document with markdown and metadata."""
        reader = PDFReader()
        documents = reader.load_data(pdf_file)

        assert len(documents) == 1
        assert documents[0].text == SAMPLE_MARKDOWN
        assert documents[0].metadata["file_path"] == str(pdf_file)
        assert documents[0].metadata["file_name"] == "test.pdf"
        mock_docling["converter"].convert.assert_called_once_with(str(pdf_file))

    def test_load_data_with_extra_info(self, mock_docling, pdf_file):
        """extra_info is merged into the Document metadata."""
        reader = PDFReader()
        documents = reader.load_data(pdf_file, extra_info={"custom_key": "custom_value"})

        assert documents[0].metadata["custom_key"] == "custom_value"
        assert documents[0].metadata["file_path"] == str(pdf_file)

    def test_load_data_does_not_mutate_extra_info(self, mock_docling, pdf_file):
        """The caller's extra_info dict is not mutated."""
        reader = PDFReader()
        extra_info = {"custom_key": "custom_value"}

        reader.load_data(pdf_file, extra_info=extra_info)

        assert set(extra_info.keys()) == {"custom_key"}
        assert "file_path" not in extra_info
        assert "file_name" not in extra_info

    def test_load_data_conversion_error(self, mock_docling, pdf_file):
        """A docling failure is wrapped in RuntimeError with context."""
        mock_docling["converter"].convert.side_effect = Exception("Conversion failed")
        reader = PDFReader()

        with pytest.raises(RuntimeError, match="Failed to convert PDF file"):
            reader.load_data(pdf_file)

    def test_load_data_warns_on_empty_output(self, mock_docling, pdf_file, caplog):
        """R7: near-empty output (likely a scanned PDF) triggers a single warning."""
        mock_docling["document"].export_to_markdown.return_value = "x" * (
            EMPTY_OUTPUT_THRESHOLD - 1
        )
        reader = PDFReader()

        with caplog.at_level(logging.WARNING, logger="lightspeed_rag_content.pdf.pdf_reader"):
            reader.load_data(pdf_file)

        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        assert str(pdf_file) in warnings[0].getMessage()

    def test_load_data_no_warning_on_normal_output(self, mock_docling, pdf_file, caplog):
        """Sufficiently long output does not trigger the R7 warning."""
        reader = PDFReader()

        with caplog.at_level(logging.WARNING, logger="lightspeed_rag_content.pdf.pdf_reader"):
            reader.load_data(pdf_file)

        assert [r for r in caplog.records if r.levelname == "WARNING"] == []


class TestConvertPdfFileToMarkdown:
    """Tests for the convert_pdf_file_to_markdown convenience function."""

    def test_successful_conversion(self, mock_docling, pdf_file):
        """The helper returns the converted markdown text."""
        assert convert_pdf_file_to_markdown(pdf_file) == SAMPLE_MARKDOWN

    def test_file_not_found(self, mock_docling):
        """The helper propagates FileNotFoundError for a missing file."""
        with pytest.raises(FileNotFoundError):
            convert_pdf_file_to_markdown("/nonexistent/file.pdf")


@requires_docling_models
class TestPDFReaderRealDocling:
    """Real, un-mocked conversion of the committed fixture PDF through docling."""

    def test_real_conversion_extracts_marker_text(self, fixture_pdf):
        """The known marker strings survive a real docling conversion."""
        documents = PDFReader().load_data(fixture_pdf)

        assert len(documents) == 1
        text = documents[0].text
        for marker in FIXTURE_MARKERS:
            assert marker in text, f"expected {marker!r} in converted markdown"
        assert documents[0].metadata["file_name"] == "fixture.pdf"

    def test_real_conversion_infers_heading(self, fixture_pdf):
        """docling infers the larger first line as a Markdown heading for chunking."""
        text = convert_pdf_file_to_markdown(fixture_pdf)
        assert FIXTURE_HEADING in text
        # The heading line should carry a Markdown heading prefix so that
        # MarkdownNodeParser splits on it downstream.
        heading_line = next(line for line in text.splitlines() if FIXTURE_HEADING in line)
        assert heading_line.lstrip().startswith("#")
