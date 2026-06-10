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
"""PDF Reader using docling for conversion to Markdown.

This module provides a PDFReader class that implements llama-index's BaseReader
interface, allowing PDF files to be read and converted to Markdown format using
the docling library. PDF parsing is docling's primary capability: it performs
layout analysis, reading-order detection, and table-structure recognition.

OCR is intentionally disabled: scanned or image-only PDFs are out of scope and
convert to empty or near-empty Markdown without erroring. When that happens,
load_data emits a warning so the condition is visible in the caller's logs rather
than degrading silently.

Typical usage example:

    >>> from lightspeed_rag_content.pdf import PDFReader
    >>> reader = PDFReader()
    >>> documents = reader.load_data(Path("document.pdf"))

The reader can be used with llama-index's SimpleDirectoryReader via file_extractor:

    >>> from llama_index.core import SimpleDirectoryReader
    >>> reader = SimpleDirectoryReader(
    ...     "docs/",
    ...     file_extractor={".pdf": PDFReader()}
    ... )
    >>> docs = reader.load_data()
"""

# The PDF reader deliberately mirrors the HTML reader's structure (see the BYOK
# PDF spec); a shared DoclingReader base is an explicit, deferred follow-up. Until
# then, suppress duplicate-code reports against html_reader.py.
# pylint: disable=duplicate-code

import logging
from pathlib import Path
from typing import Any, Final, Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

LOG: logging.Logger = logging.getLogger(__name__)

# Markdown shorter than this (after stripping) is treated as "empty output", the
# likely signature of a scanned/image-only PDF: OCR is disabled, so a PDF with no
# text layer extracts nothing. Module-level so it is easy to find and tune.
EMPTY_OUTPUT_THRESHOLD: Final[int] = 50


class PDFReader(BaseReader):
    """Read PDF files and convert them to Markdown using docling.

    This reader implements the llama-index BaseReader interface, making it
    compatible with SimpleDirectoryReader's file_extractor parameter.

    The reader uses docling's DocumentConverter with a PDF pipeline tuned for
    offline indexing: OCR off, table-structure recognition on in ACCURATE mode.

    Attributes:
        converter: The docling DocumentConverter instance used for conversion.
    """

    def __init__(self) -> None:
        """Initialize the PDFReader with a PDF-tuned docling DocumentConverter."""
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options = TableStructureOptions(
            mode=TableFormerMode.ACCURATE
        )
        self.converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            },
        )

    def load_data(  # pylint: disable=arguments-differ
        self,
        file: Path,
        extra_info: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Load and convert a PDF file to a Document.

        Args:
            file: Path to the PDF file to read.
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
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        LOG.info("Converting PDF file to Markdown: %s", file_path)

        try:
            result = self.converter.convert(str(file_path))
            markdown_content = result.document.export_to_markdown()
        except Exception as exc:
            LOG.error("Failed to convert PDF file %s: %s", file_path, exc)
            raise RuntimeError(f"Failed to convert PDF file '{file_path}': {exc}") from exc

        if len(markdown_content.strip()) < EMPTY_OUTPUT_THRESHOLD:
            # Most likely a scanned/image-only PDF (no extractable text, OCR
            # off). Surface it loudly instead of silently indexing an empty chunk.
            LOG.warning(
                "PDF produced little or no text (%d chars): %s. It may be scanned or "
                "image-only; OCR is disabled, so no text was extracted.",
                len(markdown_content.strip()),
                file_path,
            )
        else:
            LOG.debug("Successfully converted %s to Markdown", file_path)

        metadata = dict(extra_info) if extra_info else {}
        metadata["file_path"] = str(file_path)
        metadata["file_name"] = file_path.name

        return [Document(text=markdown_content, metadata=metadata)]


def convert_pdf_file_to_markdown(file_path: str | Path) -> str:
    """Convert a PDF file to Markdown format.

    This is a convenience function for standalone PDF to Markdown conversion.

    Args:
        file_path: Path to the PDF file to convert.

    Returns:
        The converted Markdown content as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If conversion fails.
    """
    reader = PDFReader()
    documents = reader.load_data(Path(file_path))
    return documents[0].text
