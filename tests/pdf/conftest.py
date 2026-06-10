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
"""Shared fixtures and helpers for PDF tests."""

import os
from pathlib import Path

import pytest

# Marker strings baked into fixture.pdf (see generate_fixture.py).
FIXTURE_HEADING = "Red Hat OpenShift Lightspeed"
FIXTURE_MARKERS = ("Raleigh", "BYOK")


def _docling_models_available() -> bool:
    """Return True if docling's HF-cached models are present locally.

    Real PDF conversion downloads layout/table models from the Hugging Face hub
    on first use. CI images are expected to pre-cache them. When they are absent
    (and especially when offline), the real-conversion tests are skipped rather
    than failing, so the suite stays green without the multi-gigabyte download.
    """
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    hub = Path(hf_home) / "hub"
    return hub.exists() and any(hub.glob("models--docling-project--*"))


requires_docling_models = pytest.mark.skipif(
    not _docling_models_available(),
    reason="docling models are not cached locally; pre-cache them to run real PDF conversion",
)


@pytest.fixture(name="fixture_pdf")
def fixture_pdf_fixture() -> Path:
    """Path to the committed tiny text-extractable PDF fixture."""
    return Path(__file__).with_name("fixture.pdf")


@pytest.fixture(name="restore_llama_index_settings")
def restore_llama_index_settings_fixture():
    """Restore llama_index's global Settings.node_parser after the test.

    DocumentProcessor mutates the process-wide llama_index ``Settings`` singleton
    (it sets ``node_parser`` to a MarkdownNodeParser for markdown/html/pdf). A
    MarkdownNodeParser has no ``chunk_size``, so if it leaks into a later test the
    ``Settings.chunk_size`` setter raises. Each rag-content run is its own process
    in production, so this only matters for in-process test isolation. Reset to the
    default splitter on teardown so other test modules are unaffected.
    """
    from llama_index.core import Settings
    from llama_index.core.node_parser import SentenceSplitter

    try:
        yield
    finally:
        Settings.node_parser = SentenceSplitter()
