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
"""Tests for HTML CLI module."""

import argparse
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lightspeed_rag_content.html.__main__ import (
    get_argument_parser,
    main_batch,
    main_convert,
)


@pytest.fixture
def mock_html_reader(mocker):
    """Mock HTMLReader for CLI tests."""
    mock_document = MagicMock()
    mock_document.text = "# Converted Markdown\n\nContent here."

    mock_reader_instance = MagicMock()
    mock_reader_instance.load_data.return_value = [mock_document]

    mock_reader_class = mocker.patch(
        "lightspeed_rag_content.html.__main__.HTMLReader",
        return_value=mock_reader_instance,
    )
    return {
        "reader_class": mock_reader_class,
        "reader": mock_reader_instance,
        "document": mock_document,
    }


@pytest.fixture
def html_file(tmp_path):
    """Create a temporary HTML file for testing."""
    html_content = "<html><body><h1>Test</h1></body></html>"
    file_path = tmp_path / "test.html"
    file_path.write_text(html_content, encoding="utf-8")
    return file_path


class TestHtmlMainConvert:
    """Tests for main_convert function."""

    def test_main_convert_success(self, mock_html_reader, html_file, tmp_path):
        """Test successful single file conversion."""
        output_file = tmp_path / "output.md"
        args = argparse.Namespace(input_file=html_file, output_file=output_file)

        main_convert(args)

        assert output_file.exists()
        assert output_file.read_text() == "# Converted Markdown\n\nContent here."
        mock_html_reader["reader"].load_data.assert_called_once_with(html_file)

    def test_main_convert_default_output(self, mock_html_reader, html_file):
        """Test conversion with default output path (same name with .md extension)."""
        args = argparse.Namespace(input_file=html_file, output_file=None)

        main_convert(args)

        expected_output = html_file.with_suffix(".md")
        assert expected_output.exists()
        assert expected_output.read_text() == "# Converted Markdown\n\nContent here."

    def test_main_convert_file_not_found(self, mock_html_reader, tmp_path):
        """Test conversion with non-existent file."""
        mock_html_reader["reader"].load_data.side_effect = FileNotFoundError(
            "File not found"
        )
        args = argparse.Namespace(
            input_file=tmp_path / "nonexistent.html", output_file=None
        )

        with pytest.raises(SystemExit) as exc_info:
            main_convert(args)

        assert exc_info.value.code == 1

    def test_main_convert_runtime_error(self, mock_html_reader, html_file, tmp_path):
        """Test conversion with runtime error."""
        mock_html_reader["reader"].load_data.side_effect = RuntimeError(
            "Conversion failed"
        )
        args = argparse.Namespace(
            input_file=html_file, output_file=tmp_path / "output.md"
        )

        with pytest.raises(SystemExit) as exc_info:
            main_convert(args)

        assert exc_info.value.code == 1


class TestHtmlMainBatch:
    """Tests for main_batch function."""

    def test_main_batch_success(self, mock_html_reader, tmp_path):
        """Test successful batch conversion."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "file1.html").write_text("<h1>File 1</h1>", encoding="utf-8")
        (input_dir / "file2.htm").write_text("<h1>File 2</h1>", encoding="utf-8")

        output_dir = tmp_path / "output"
        args = argparse.Namespace(input_dir=input_dir, output_dir=output_dir)

        main_batch(args)

        assert output_dir.exists()
        assert (output_dir / "file1.md").exists()
        assert (output_dir / "file2.md").exists()

    def test_main_batch_default_output_dir(self, mock_html_reader, tmp_path):
        """Test batch conversion with default output directory."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "test.html").write_text("<h1>Test</h1>", encoding="utf-8")

        args = argparse.Namespace(input_dir=input_dir, output_dir=None)

        main_batch(args)

        # Output should be in the same directory as input
        assert (input_dir / "test.md").exists()

    def test_main_batch_nonexistent_directory(self, mock_html_reader, tmp_path):
        """Test batch conversion with non-existent input directory."""
        args = argparse.Namespace(
            input_dir=tmp_path / "nonexistent", output_dir=tmp_path / "output"
        )

        with pytest.raises(SystemExit) as exc_info:
            main_batch(args)

        assert exc_info.value.code == 1

    def test_main_batch_no_html_files(self, mock_html_reader, tmp_path, caplog):
        """Test batch conversion with no HTML files in directory."""
        input_dir = tmp_path / "empty"
        input_dir.mkdir()

        args = argparse.Namespace(input_dir=input_dir, output_dir=None)

        # Should not raise, just log warning
        main_batch(args)

        assert "No HTML files found" in caplog.text

    def test_main_batch_with_errors(self, mock_html_reader, tmp_path):
        """Test batch conversion with some files failing."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "good.html").write_text("<h1>Good</h1>", encoding="utf-8")
        (input_dir / "bad.html").write_text("<h1>Bad</h1>", encoding="utf-8")

        # Make every other call fail
        mock_html_reader["reader"].load_data.side_effect = [
            [mock_html_reader["document"]],
            RuntimeError("Conversion failed"),
        ]

        args = argparse.Namespace(input_dir=input_dir, output_dir=tmp_path / "output")

        with pytest.raises(SystemExit) as exc_info:
            main_batch(args)

        # Should exit with error code due to some failures
        assert exc_info.value.code == 1

    def test_main_batch_preserves_directory_structure(self, mock_html_reader, tmp_path):
        """Test that batch conversion preserves subdirectory structure."""
        input_dir = tmp_path / "input"
        sub_dir = input_dir / "subdir"
        sub_dir.mkdir(parents=True)
        (sub_dir / "nested.html").write_text("<h1>Nested</h1>", encoding="utf-8")

        output_dir = tmp_path / "output"
        args = argparse.Namespace(input_dir=input_dir, output_dir=output_dir)

        main_batch(args)

        assert (output_dir / "subdir" / "nested.md").exists()


class TestGetArgumentParser:
    """Tests for get_argument_parser function."""

    def test_returns_argument_parser(self):
        """Test that get_argument_parser returns an ArgumentParser."""
        parser = get_argument_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_convert_command(self):
        """Test parsing convert command."""
        parser = get_argument_parser()
        args = parser.parse_args(["convert", "-i", "input.html", "-o", "output.md"])

        assert args.command == "convert"
        assert args.input_file == Path("input.html")
        assert args.output_file == Path("output.md")

    def test_convert_command_default_output(self):
        """Test parsing convert command without output file."""
        parser = get_argument_parser()
        args = parser.parse_args(["convert", "-i", "input.html"])

        assert args.command == "convert"
        assert args.input_file == Path("input.html")
        assert args.output_file is None

    def test_batch_command(self):
        """Test parsing batch command."""
        parser = get_argument_parser()
        args = parser.parse_args(["batch", "-i", "./html_dir", "-o", "./md_dir"])

        assert args.command == "batch"
        assert args.input_dir == Path("./html_dir")
        assert args.output_dir == Path("./md_dir")

    def test_batch_command_default_output(self):
        """Test parsing batch command without output directory."""
        parser = get_argument_parser()
        args = parser.parse_args(["batch", "-i", "./html_dir"])

        assert args.command == "batch"
        assert args.input_dir == Path("./html_dir")
        assert args.output_dir is None

    def test_missing_command_raises_error(self):
        """Test that missing command raises SystemExit."""
        parser = get_argument_parser()

        with pytest.raises(SystemExit):
            parser.parse_args([])
