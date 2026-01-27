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
"""HTML Reader using docling for conversion to Markdown.

This module provides an HTMLReader class that implements llama-index's BaseReader
interface, allowing HTML files to be read and converted to Markdown format using
the docling library.

Typical usage example:

    >>> from lightspeed_rag_content.html import HTMLReader
    >>> reader = HTMLReader()
    >>> documents = reader.load_data(Path("document.html"))

The reader can be used with llama-index's SimpleDirectoryReader via file_extractor:

    >>> from llama_index.core import SimpleDirectoryReader
    >>> reader = SimpleDirectoryReader(
    ...     "docs/",
    ...     file_extractor={".html": HTMLReader(), ".htm": HTMLReader()}
    ... )
    >>> docs = reader.load_data()
"""

import logging
from pathlib import Path
from typing import Any, Optional

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

LOG: logging.Logger = logging.getLogger(__name__)


class HTMLReader(BaseReader):
    """Read HTML files and convert them to Markdown using docling.

    This reader implements the llama-index BaseReader interface, making it
    compatible with SimpleDirectoryReader's file_extractor parameter.

    The reader uses docling's DocumentConverter to parse HTML files and
    export them as Markdown, preserving document structure where possible.

    Attributes:
        converter: The docling DocumentConverter instance used for conversion.
    """

    def __init__(self) -> None:
        """Initialize the HTMLReader with a docling DocumentConverter."""
        self.converter = DocumentConverter(
            allowed_formats=[InputFormat.HTML],
        )

    def load_data(  # pylint: disable=arguments-differ
        self,
        file: Path,
        extra_info: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Load and convert an HTML file to a Document.

        Args:
            file: Path to the HTML file to read.
            extra_info: Optional metadata to include in the Document.
            **kwargs: Additional keyword arguments (unused, for BaseReader compatibility).

        Returns:
            A list containing a single Document with the converted Markdown content.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            RuntimeError: If conversion fails.
        """
        del kwargs
        file_path = Path(file)

        if not file_path.exists():
            raise FileNotFoundError(f"HTML file not found: {file_path}")

        LOG.info("Converting HTML file to Markdown: %s", file_path)

        try:
            result = self.converter.convert(str(file_path))
            markdown_content = result.document.export_to_markdown()
        except Exception as exc:
            LOG.error("Failed to convert HTML file %s: %s", file_path, exc)
            raise RuntimeError(
                f"Failed to convert HTML file '{file_path}': {exc}"
            ) from exc

        LOG.debug("Successfully converted %s to Markdown", file_path)

        metadata = dict(extra_info) if extra_info else {}
        metadata["file_path"] = str(file_path)
        metadata["file_name"] = file_path.name

        return [Document(text=markdown_content, metadata=metadata)]


def convert_html_file_to_markdown(file_path: str | Path) -> str:
    """Convert an HTML file to Markdown format.

    This is a convenience function for standalone HTML to Markdown conversion.

    Args:
        file_path: Path to the HTML file to convert.

    Returns:
        The converted Markdown content as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If conversion fails.
    """
    reader = HTMLReader()
    documents = reader.load_data(Path(file_path))
    return documents[0].text


def convert_html_string_to_markdown(
    html_content: str, document_name: str | None = None
) -> str:
    """Convert an HTML string to Markdown format.

    Args:
        html_content: The HTML content to convert.
        document_name: Optional name for the document.

    Returns:
        The converted Markdown content as a string.

    Raises:
        RuntimeError: If conversion fails.
    """
    if not html_content:
        return ""

    LOG.info("Converting HTML string to Markdown")

    try:
        converter = DocumentConverter(allowed_formats=[InputFormat.HTML])
        result = converter.convert_string(
            content=html_content,
            format=InputFormat.HTML,
            name=document_name,
        )
        return result.document.export_to_markdown()
    except Exception as exc:
        LOG.error("Failed to convert HTML string: %s", exc)
        raise RuntimeError(f"Failed to convert HTML string: {exc}") from exc
