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
"""End-to-end integration test: AsciiDoc conversion preserves content.

Drives the real AsciidoctorConverter (requires ``asciidoctor`` to be installed)
and verifies that the converted output matches the hand-written Markdown
reference.  This validates the .adoc -> .md -> generate_embeddings.py path
that users follow inside the tool container.
"""

import shutil
from pathlib import Path

import pytest

from lightspeed_rag_content.asciidoc import AsciidoctorConverter

FIXTURE_DIR = Path(__file__).parent
FIXTURE_ADOC = FIXTURE_DIR / "fixture.adoc"
FIXTURE_MD = FIXTURE_DIR / "fixture.md"

requires_asciidoctor = pytest.mark.skipif(
    not shutil.which("ruby") or not shutil.which("asciidoctor"),
    reason="ruby and/or asciidoctor not installed (run 'bundle install' first)",
)

CONTENT_MARKERS = [
    "Getting Started with Lightspeed",
    "Prerequisites",
    "cluster administrator access",
    "OpenShift environment",
    "oc",
    "podman",
    "container registry",
    "Installation",
    "oc apply -f config.yaml",
    "pod is running",
    "service endpoint responds",
    "Review the logs for errors",
    "admin privileges",
    "Configuration",
    "config.yaml",
    "proxy_url",
    "upstream proxy",
    "timeout",
    "Connection timeout in seconds",
]


@requires_asciidoctor
class TestAsciidocIntegration:
    def test_converted_output_matches_reference(self, tmp_path):
        """The .adoc -> text conversion produces the same content as the .md reference."""
        output_file = tmp_path / "converted.md"

        converter = AsciidoctorConverter()
        converter.convert(FIXTURE_ADOC, output_file)

        converted = output_file.read_text(encoding="utf-8")
        reference = FIXTURE_MD.read_text(encoding="utf-8")

        assert converted.strip() == reference.strip()

    def test_converted_output_preserves_all_content(self, tmp_path):
        """Every key phrase from the source .adoc survives the conversion."""
        output_file = tmp_path / "converted.md"

        converter = AsciidoctorConverter()
        converter.convert(FIXTURE_ADOC, output_file)

        converted = output_file.read_text(encoding="utf-8")

        for marker in CONTENT_MARKERS:
            assert marker in converted, f"expected {marker!r} in converted text"
