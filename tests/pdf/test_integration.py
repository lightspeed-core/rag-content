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
"""End-to-end integration test: a PDF directory builds a Faiss index.

This is the test the HTML suite never had: it drives the *real* pipeline a BYOK
user hits, with only the embedding model mocked.

    doc_type="pdf"
      -> DocumentProcessor.process() injects the default PDFReader extractor
      -> real docling converts fixture.pdf to Markdown
      -> MarkdownNodeParser chunks it
      -> chunks land in a faiss-backed node list

It asserts that the known marker strings from the source PDF survive all the way
into the indexed chunks. Skipped when docling models are not cached locally.
"""

import os
import shutil
from pathlib import Path

from lightspeed_rag_content import document_processor
from lightspeed_rag_content.metadata_processor import DefaultMetadataProcessor
from tests.conftest import RagMockEmbedding
from tests.pdf.conftest import FIXTURE_MARKERS, requires_docling_models


@requires_docling_models
def test_pdf_directory_builds_faiss_index_with_pdf_text(
    mocker, tmp_path, fixture_pdf, restore_llama_index_settings
):
    """A directory containing a PDF produces indexed chunks holding the PDF text."""
    # docling resolves its layout/table models from the real Hugging Face cache;
    # only the sentence-transformers embedding is mocked so no embedding model
    # download is needed.
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    mocker.patch.object(document_processor, "HuggingFaceEmbedding", new=RagMockEmbedding)

    docs_dir = tmp_path / "byok"
    docs_dir.mkdir()
    shutil.copy(fixture_pdf, docs_dir / "fixture.pdf")

    processor = document_processor.DocumentProcessor(
        chunk_size=380,
        chunk_overlap=0,
        model_name="mock-model",
        embeddings_model_dir=Path(hf_home),
        vector_store_type="faiss",
        doc_type="pdf",
    )
    # No file_extractor passed: process() must inject the docling PDFReader itself.
    processor.process(docs_dir, metadata=DefaultMetadataProcessor(hermetic_build=True))

    good_nodes = processor.db._good_nodes  # pylint: disable=protected-access
    assert good_nodes, "expected at least one indexed chunk from the PDF"

    node_text = "\n".join(node.text for node in good_nodes)
    for marker in FIXTURE_MARKERS:
        assert marker in node_text, f"expected {marker!r} in the indexed chunk text"
