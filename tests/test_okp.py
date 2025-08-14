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

import unittest
from unittest import mock

from lightspeed_rag_content import okp


class TestOKP(unittest.TestCase):
    """Test cases for OKP utility methods."""

    def test_metadata_has_url_and_title(self):
        """Test that the metadata has both URL and title."""
        md = {
            "title": "Example Title",
            "extra": {
                "reference_url": "https://fake.url/for/example",
            },
        }
        self.assertTrue(okp.metadata_has_url_and_title(md))

    def test_metadata_has_url_and_title_false(self):
        """Test that the metadata has both URL and title."""
        # No URL
        md = {
            "title": "Example Title",
            "extra": {},
        }
        self.assertFalse(okp.metadata_has_url_and_title(md))

        # No title
        md = {
            "extra": {
                "reference_url": "https://fake.url/for/example",
            },
        }
        self.assertFalse(okp.metadata_has_url_and_title(md))

    def test_is_file_related_to_projects(self):
        """Test if the file is related to specific projects."""
        metadata = {
            "extra": {
                "portal_product_names": ["Project Foo", "Project Bar"],
            },
        }
        projects = ["foo", "bar"]
        self.assertTrue(okp.is_file_related_to_projects(metadata, projects))

        projects = ["spongebob"]
        self.assertFalse(okp.is_file_related_to_projects(metadata, projects))

    def test_parse_metadata(self):
        """Test parsing metadata from a OKP file."""
        content = b"""
            +++
            title = '''Example Title'''
            path = "/errata/FAKE-1234"
            template = "erratum.html"
            [extra]
            document_kind="errata"
            original_title='''FAKE-1234 - Bugs in fake project'''
            solr_index="true"
            modified="2003-02-05T00:00:00Z"
            issued="2003-02-06T00:00:00Z"
            id="FAKE-1234"
            reference_url="https://fake.url/for/example"
            view_uri="/errata/FAKE-1234"
            portal_advisory_type="Bug Fix Advisory"
            portal_synopsis='''Bugs in fake project'''
            portal_severity="None"
            portal_product_names=["Product Foo","Product Bar"]
            portal_product_filter=["Product Foo|Product Bar|2|ia64","Product Foo|Product Bar|2|ia64"]
            +++
            """

        m = mock.mock_open(read_data=content)
        with mock.patch("builtins.open", m):
            metadata = okp.parse_metadata("fake_file.md")

        # Check if the metadata is parsed correctly
        expected_metadata = {
            "title": "Example Title",
            "path": "/errata/FAKE-1234",
            "template": "erratum.html",
            "extra": {
                "document_kind": "errata",
                "original_title": "FAKE-1234 - Bugs in fake project",
                "solr_index": "true",
                "modified": "2003-02-05T00:00:00Z",
                "issued": "2003-02-06T00:00:00Z",
                "id": "FAKE-1234",
                "reference_url": "https://fake.url/for/example",
                "view_uri": "/errata/FAKE-1234",
                "portal_advisory_type": "Bug Fix Advisory",
                "portal_synopsis": "Bugs in fake project",
                "portal_severity": "None",
                "portal_product_names": ["Product Foo", "Product Bar"],
                "portal_product_filter": [
                    "Product Foo|Product Bar|2|ia64",
                    "Product Foo|Product Bar|2|ia64",
                ],
            },
        }
        self.assertEqual(metadata, expected_metadata)

    @mock.patch("lightspeed_rag_content.okp.Path.glob")
    def test_yield_files_related_to_projects(self, mock_glob):
        """Test yielding files related to specific projects."""
        mock_glob.return_value = [
            "file1.md",
            "file2.md",
            "file3.md",  # Should be ignored, missing metadata
        ]

        okp.parse_metadata = mock.MagicMock(
            side_effect=[
                {
                    "title": "File 1",
                    "extra": {
                        "reference_url": "https://example.com/file1",
                        "portal_product_names": ["Project Foo"],
                    },
                },
                {
                    "title": "File 2",
                    "extra": {
                        "reference_url": "https://example.com/file2",
                        "portal_product_names": ["Project Bar"],
                    },
                },
                {
                    "title": "File 3",
                    "extra": {
                        "portal_product_names": ["Project Baz"],
                    },
                },
            ]
        )

        projects = ["foo", "bar"]
        files = list(okp.yield_files_related_to_projects("/fake", projects))

        # Check that the correct files are yielded
        self.assertEqual(len(files), 2)
        self.assertIn("file1.md", files)
        self.assertIn("file2.md", files)

        # Check that parse_metadata was called with the correct file paths
        okp.parse_metadata.assert_any_call("file1.md")
        okp.parse_metadata.assert_any_call("file2.md")


@mock.patch(
    "lightspeed_rag_content.okp.parse_metadata",
    return_value={
        "title": "Test Title",
        "extra": {
            "reference_url": "https://example.com",
        },
    },
)
class TestOKPMetadataProcessor(unittest.TestCase):
    """Test cases for OKPMetadataProcessor class."""

    def setUp(self):
        """Set up the test case."""
        self.okp_mp = okp.OKPMetadataProcessor()

    def test_url_function(self, mock_parse_metadata):
        """Test the URL function of OKPMetadataProcessor."""
        file_path = "/fake/path/errata_file.md"
        expected_url = "https://example.com"
        self.assertEqual(self.okp_mp.url_function(file_path), expected_url)

    def test_get_file_title(self, mock_parse_metadata):
        """Test the get_file_title function of OKPMetadataProcessor."""
        file_path = "/fake/path/errata_file.md"
        expected_title = "Test Title"
        self.assertEqual(self.okp_mp.get_file_title(file_path), expected_title)
