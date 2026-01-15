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
"""Utilities for rag-content modules."""
import argparse
import logging
from collections.abc import Callable
from pathlib import Path

# Standard log format for CLI modules
CLI_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_cli_logging(package_name: str | None = None) -> logging.Logger:
    """Configure logging for CLI modules with a standard format.

    Args:
        package_name: The package name to use for the logger. If None,
            uses the root logger.

    Returns:
        A configured Logger instance for the specified package.
    """
    logging.basicConfig(
        level=logging.INFO,
        format=CLI_LOG_FORMAT,
    )
    return logging.getLogger(package_name)


def run_cli_command(
    parser: argparse.ArgumentParser,
    command_handlers: dict[str, Callable[[argparse.Namespace], None]],
) -> None:
    """Parse CLI arguments and dispatch to the appropriate command handler.

    Args:
        parser: The argument parser to use for parsing command line arguments.
        command_handlers: A dictionary mapping command names to handler functions.
            Each handler receives the parsed arguments Namespace.

    Raises:
        SystemExit: If the command is not found in handlers.
    """
    args = parser.parse_args()
    handler = command_handlers.get(args.command)
    if handler is None:
        raise SystemExit(f"Unknown command: {args.command}")
    handler(args)


def add_input_file_argument(
    parser: argparse.ArgumentParser,
    help_text: str = "Input file to process.",
) -> None:
    """Add a standard -i/--input-file argument to an argument parser.

    Args:
        parser: The argument parser to add the argument to.
        help_text: Help text for the argument.
    """
    parser.add_argument(
        "-i",
        "--input-file",
        required=True,
        type=Path,
        help=help_text,
    )


def get_common_arg_parser() -> argparse.ArgumentParser:
    """Provide common CLI arguments to document processing scripts."""
    parser = argparse.ArgumentParser(description="Embedding CLI for task execution")
    parser.add_argument(
        "-f", "--folder", help="Directory containing the plain text documentation"
    )
    parser.add_argument(
        "-md",
        "--model-dir",
        default="embeddings_model",
        help="Directory containing the embedding model.",
    )
    parser.add_argument("-mn", "--model-name", help="HF repo id of the embedding model")
    parser.add_argument(
        "-c", "--chunk", type=int, default=380, help="Chunk size for embedding"
    )
    parser.add_argument(
        "-l", "--overlap", type=int, default=0, help="Chunk overlap for embedding"
    )
    parser.add_argument(
        "-em",
        "--exclude-metadata",
        nargs="+",
        default=None,
        help="Metadata to be excluded during embedding",
    )
    parser.add_argument("-o", "--output", help="Vector DB output folder")
    parser.add_argument("-i", "--index", help="Product index")
    parser.add_argument(
        "-w",
        "--workers",
        default=None,
        type=int,
        help=(
            "Number of workers to parallelize the data loading. Set to a "
            "negative value by default, turning parallelism off"
        ),
    )
    parser.add_argument(
        "--vector-store-type",
        default="faiss",
        choices=["faiss", "postgres", "llamastack-faiss", "llamastack-sqlite-vec"],
        help="vector store type to be used.",
    )
    parser.add_argument(
        "--auto-chunking",
        action="store_false",
        default=True,
        dest="manual_chunking",
        help="How to do the chunking for llama-stack, manually like in "
        "llama-index or automatically using the RAG runtime tool.",
    )
    parser.add_argument(
        "-dt",
        "--document-type",
        dest="doc_type",
        default="text",
        choices=["text", "markdown", "html"],
        help="The type of the document which is to be added to the RAG.",
    )
    return parser
