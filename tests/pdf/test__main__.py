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
"""Tests for the PDF CLI module (docling mocked via PDFReader)."""

import argparse
from pathlib import Path

import pytest

from lightspeed_rag_content.pdf.__main__ import (
    get_argument_parser,
    main_batch,
    main_convert,
)

CONVERTED_MARKDOWN = "# Converted Markdown\n\nContent here."


@pytest.fixture(name="mock_pdf_reader")
def mock_pdf_reader_fixture(mocker):
    """Mock PDFReader for CLI tests."""
    mock_document = mocker.MagicMock()
    mock_document.text = CONVERTED_MARKDOWN

    mock_reader_instance = mocker.MagicMock()
    mock_reader_instance.load_data.return_value = [mock_document]

    mock_reader_class = mocker.patch(
        "lightspeed_rag_content.pdf.__main__.PDFReader",
        return_value=mock_reader_instance,
    )
    return {
        "reader_class": mock_reader_class,
        "reader": mock_reader_instance,
        "document": mock_document,
    }


@pytest.fixture(name="pdf_file")
def pdf_file_fixture(tmp_path):
    """Create a placeholder PDF file (docling is mocked)."""
    file_path = tmp_path / "test.pdf"
    file_path.write_bytes(b"%PDF-1.4 placeholder")
    return file_path


class TestPdfMainConvert:
    """Tests for main_convert."""

    def test_main_convert_success(self, mock_pdf_reader, pdf_file, tmp_path):
        """A single file is converted and written to the requested output path."""
        output_file = tmp_path / "output.md"
        args = argparse.Namespace(input_file=pdf_file, output_file=output_file)

        main_convert(args)

        assert output_file.read_text() == CONVERTED_MARKDOWN
        mock_pdf_reader["reader"].load_data.assert_called_once_with(pdf_file)

    def test_main_convert_default_output(self, mock_pdf_reader, pdf_file):
        """Without -o the output path defaults to the input with a .md suffix."""
        args = argparse.Namespace(input_file=pdf_file, output_file=None)

        main_convert(args)

        expected_output = pdf_file.with_suffix(".md")
        assert expected_output.read_text() == CONVERTED_MARKDOWN

    def test_main_convert_file_not_found(self, mock_pdf_reader, tmp_path):
        """A FileNotFoundError from the reader exits non-zero."""
        mock_pdf_reader["reader"].load_data.side_effect = FileNotFoundError("File not found")
        args = argparse.Namespace(input_file=tmp_path / "nonexistent.pdf", output_file=None)

        with pytest.raises(SystemExit) as exc_info:
            main_convert(args)

        assert exc_info.value.code == 1

    def test_main_convert_runtime_error(self, mock_pdf_reader, pdf_file, tmp_path):
        """A RuntimeError from the reader exits non-zero."""
        mock_pdf_reader["reader"].load_data.side_effect = RuntimeError("Conversion failed")
        args = argparse.Namespace(input_file=pdf_file, output_file=tmp_path / "output.md")

        with pytest.raises(SystemExit) as exc_info:
            main_convert(args)

        assert exc_info.value.code == 1


class TestPdfMainBatch:
    """Tests for main_batch."""

    def test_main_batch_success(self, mock_pdf_reader, tmp_path):
        """All PDFs in a directory are converted."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "file1.pdf").write_bytes(b"%PDF-1.4 a")
        (input_dir / "file2.pdf").write_bytes(b"%PDF-1.4 b")

        output_dir = tmp_path / "output"
        main_batch(argparse.Namespace(input_dir=input_dir, output_dir=output_dir))

        assert (output_dir / "file1.md").exists()
        assert (output_dir / "file2.md").exists()

    def test_main_batch_default_output_dir(self, mock_pdf_reader, tmp_path):
        """Without -o the output lands next to the input files."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "test.pdf").write_bytes(b"%PDF-1.4 a")

        main_batch(argparse.Namespace(input_dir=input_dir, output_dir=None))

        assert (input_dir / "test.md").exists()

    def test_main_batch_nonexistent_directory(self, mock_pdf_reader, tmp_path):
        """A missing input directory exits non-zero."""
        args = argparse.Namespace(input_dir=tmp_path / "nonexistent", output_dir=tmp_path / "out")

        with pytest.raises(SystemExit) as exc_info:
            main_batch(args)

        assert exc_info.value.code == 1

    def test_main_batch_no_pdf_files(self, mock_pdf_reader, tmp_path, caplog):
        """An empty directory logs a warning and does not raise."""
        input_dir = tmp_path / "empty"
        input_dir.mkdir()

        main_batch(argparse.Namespace(input_dir=input_dir, output_dir=None))

        assert "No PDF files found" in caplog.text

    def test_main_batch_with_errors(self, mock_pdf_reader, tmp_path):
        """A failing file makes the batch exit non-zero."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "good.pdf").write_bytes(b"%PDF-1.4 a")
        (input_dir / "bad.pdf").write_bytes(b"%PDF-1.4 b")

        mock_pdf_reader["reader"].load_data.side_effect = [
            [mock_pdf_reader["document"]],
            RuntimeError("Conversion failed"),
        ]

        args = argparse.Namespace(input_dir=input_dir, output_dir=tmp_path / "output")
        with pytest.raises(SystemExit) as exc_info:
            main_batch(args)

        assert exc_info.value.code == 1

    def test_main_batch_preserves_directory_structure(self, mock_pdf_reader, tmp_path):
        """Subdirectory structure is preserved in the output."""
        input_dir = tmp_path / "input"
        sub_dir = input_dir / "subdir"
        sub_dir.mkdir(parents=True)
        (sub_dir / "nested.pdf").write_bytes(b"%PDF-1.4 a")

        output_dir = tmp_path / "output"
        main_batch(argparse.Namespace(input_dir=input_dir, output_dir=output_dir))

        assert (output_dir / "subdir" / "nested.md").exists()


class TestGetArgumentParser:
    """Tests for get_argument_parser."""

    def test_returns_argument_parser(self):
        """The factory returns an ArgumentParser."""
        assert isinstance(get_argument_parser(), argparse.ArgumentParser)

    def test_convert_command(self):
        """The convert subcommand parses input and output paths."""
        args = get_argument_parser().parse_args(["convert", "-i", "in.pdf", "-o", "out.md"])
        assert args.command == "convert"
        assert args.input_file == Path("in.pdf")
        assert args.output_file == Path("out.md")

    def test_convert_command_default_output(self):
        """The convert subcommand allows omitting the output path."""
        args = get_argument_parser().parse_args(["convert", "-i", "in.pdf"])
        assert args.command == "convert"
        assert args.input_file == Path("in.pdf")
        assert args.output_file is None

    def test_batch_command(self):
        """The batch subcommand parses input and output directories."""
        args = get_argument_parser().parse_args(["batch", "-i", "./pdfs", "-o", "./md"])
        assert args.command == "batch"
        assert args.input_dir == Path("./pdfs")
        assert args.output_dir == Path("./md")

    def test_batch_command_default_output(self):
        """The batch subcommand allows omitting the output directory."""
        args = get_argument_parser().parse_args(["batch", "-i", "./pdfs"])
        assert args.command == "batch"
        assert args.input_dir == Path("./pdfs")
        assert args.output_dir is None

    def test_missing_command_raises_error(self):
        """No subcommand exits non-zero."""
        with pytest.raises(SystemExit):
            get_argument_parser().parse_args([])
