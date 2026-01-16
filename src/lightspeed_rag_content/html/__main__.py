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
"""CLI for converting HTML files to Markdown using docling.

This module provides a command-line interface for converting HTML files
to Markdown format. It can convert single files or batch convert entire
directories.

Usage:
    python -m lightspeed_rag_content.html convert -i input.html -o output.md
    python -m lightspeed_rag_content.html batch -i ./html_dir -o ./md_dir
"""

import argparse
import sys
from pathlib import Path

from lightspeed_rag_content.html.html_reader import HTMLReader
from lightspeed_rag_content.utils import (
    add_input_file_argument,
    run_cli_command,
    setup_cli_logging,
)

LOG = setup_cli_logging(__package__)


def main_convert(args: argparse.Namespace) -> None:
    """Convert a single HTML file to Markdown."""
    try:
        reader = HTMLReader()
        documents = reader.load_data(args.input_file)

        output_path = args.output_file
        if output_path is None:
            output_path = args.input_file.with_suffix(".md")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(documents[0].text, encoding="utf-8")

        LOG.info("Converted %s -> %s", args.input_file, output_path)
    except (FileNotFoundError, RuntimeError) as e:
        LOG.error(str(e))
        sys.exit(1)


def main_batch(args: argparse.Namespace) -> None:
    """Batch convert HTML files in a directory to Markdown."""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir

    if not input_dir.is_dir():
        LOG.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    reader = HTMLReader()
    html_files = list(input_dir.glob("**/*.html")) + list(input_dir.glob("**/*.htm"))

    if not html_files:
        LOG.warning("No HTML files found in %s", input_dir)
        return

    success_count = 0
    error_count = 0

    for html_file in html_files:
        try:
            # Preserve directory structure
            relative_path = html_file.relative_to(input_dir)
            output_file = output_dir / relative_path.with_suffix(".md")
            output_file.parent.mkdir(parents=True, exist_ok=True)

            documents = reader.load_data(html_file)
            output_file.write_text(documents[0].text, encoding="utf-8")

            LOG.info("Converted: %s -> %s", html_file.name, output_file.name)
            success_count += 1
        except (FileNotFoundError, RuntimeError) as e:
            LOG.error("Failed to convert %s: %s", html_file, e)
            error_count += 1

    LOG.info(
        "Batch conversion complete: %d succeeded, %d failed", success_count, error_count
    )

    if error_count > 0:
        sys.exit(1)


def get_argument_parser() -> argparse.ArgumentParser:
    """Get ArgumentParser for lightspeed_rag_content.html module."""
    parser = argparse.ArgumentParser(
        description="Convert HTML files to Markdown format using docling.",
        prog=__package__,
    )
    subparser = parser.add_subparsers(dest="command", required=True)

    # Single file conversion
    convert_parser = subparser.add_parser(
        "convert",
        help="Convert a single HTML file to Markdown.",
    )
    add_input_file_argument(convert_parser, "HTML file to convert.")
    convert_parser.add_argument(
        "-o",
        "--output-file",
        required=False,
        type=Path,
        help="Output Markdown file path. Defaults to input file with .md extension.",
    )

    # Batch conversion
    batch_parser = subparser.add_parser(
        "batch",
        help="Batch convert HTML files in a directory.",
    )
    batch_parser.add_argument(
        "-i",
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing HTML files to convert.",
    )
    batch_parser.add_argument(
        "-o",
        "--output-dir",
        required=False,
        type=Path,
        help="Output directory for Markdown files. Defaults to input directory.",
    )

    return parser


if __name__ == "__main__":
    run_cli_command(
        get_argument_parser(),
        {"convert": main_convert, "batch": main_batch},
    )
