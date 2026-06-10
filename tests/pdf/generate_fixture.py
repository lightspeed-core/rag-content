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
"""Zero-dependency generator for the tiny text-extractable PDF test fixture.

Run this to (re)generate ``fixture.pdf`` next to this file::

    python tests/pdf/generate_fixture.py

It writes a minimal, valid single-page PDF (correct xref offsets, no third-party
libraries) whose extracted text contains known marker strings the tests assert
on. The first line uses a larger font so docling infers it as a Markdown heading.
"""

import sys
from pathlib import Path

# Lines of (text, x, y, font_size) placed in absolute PDF user space (origin at
# the bottom-left of a 612x792 "letter" page).
FIXTURE_LINES: list[tuple[str, int, int, int]] = [
    ("Red Hat OpenShift Lightspeed", 72, 720, 18),
    ("The capital of OpenShift is Raleigh.", 72, 690, 12),
    ("BYOK ingestion proves this sentence came from the PDF.", 72, 660, 12),
]


def build_pdf(lines: list[tuple[str, int, int, int]]) -> bytes:
    """Build a minimal one-page PDF from absolutely-positioned text lines.

    Each tuple in ``lines`` is ``(text, x, y, font_size)`` placed in PDF user
    space. Returns the PDF file content as bytes.
    """
    parts = [b"BT"]
    for text, x, y, size in lines:
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        parts.append(b"/F1 %d Tf" % size)
        parts.append(b"1 0 0 1 %d %d Tm" % (x, y))
        parts.append(b"(%s) Tj" % escaped.encode("latin-1"))
    parts.append(b"ET")
    content = b"\n".join(parts)

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (index, obj)

    xref_pos = len(out)
    size = len(objects) + 1
    out += b"xref\n0 %d\n" % size
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        size,
        xref_pos,
    )
    return bytes(out)


def main() -> None:
    """Write the fixture PDF next to this script (or to argv[1])."""
    default = Path(__file__).with_name("fixture.pdf")
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    out_path.write_bytes(build_pdf(FIXTURE_LINES))
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
