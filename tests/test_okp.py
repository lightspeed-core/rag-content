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

import pytest

from lightspeed_rag_content import okp


class TestOKP:
    """Test cases for OKP utility methods."""

    def test_metadata_has_url_and_title(self):
        """Test that the metadata has both URL and title."""
        md = {
            "title": "Example Title",
            "extra": {
                "reference_url": "https://fake.url/for/example",
            },
        }
        assert okp.metadata_has_url_and_title(md)

    def test_metadata_has_url_and_title_false(self):
        """Test that the metadata has both URL and title."""
        # No URL
        md = {
            "title": "Example Title",
            "extra": {},
        }
        assert not okp.metadata_has_url_and_title(md)

        # No title
        md = {
            "extra": {
                "reference_url": "https://fake.url/for/example",
            },
        }
        assert not okp.metadata_has_url_and_title(md)

    def test_is_file_related_to_projects(self):
        """Test if the file is related to specific projects."""
        metadata = {
            "extra": {
                "portal_product_names": ["Project Foo", "Project Bar"],
            },
        }
        projects = ["foo", "bar"]
        assert okp.is_file_related_to_projects(metadata, projects)

        projects = ["spongebob"]
        assert not okp.is_file_related_to_projects(metadata, projects)

    def test_parse_metadata(self, mocker):
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

        m = mocker.mock_open(read_data=content)
        mocker.patch("builtins.open", m)
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
        assert metadata == expected_metadata

    def test_yield_files_related_to_projects(self, mocker):
        """Test yielding files related to specific projects."""
        mock_glob = mocker.patch("lightspeed_rag_content.okp.Path.glob")
        mock_glob.return_value = [
            "file1.md",
            "file2.md",
            "file3.md",  # Should be ignored, missing metadata
        ]

        mock_parse_metadata = mocker.patch(
            "lightspeed_rag_content.okp.parse_metadata",
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
            ],
        )

        projects = ["foo", "bar"]
        files = list(okp.yield_files_related_to_projects("/fake", projects))

        # Check that the correct files are yielded
        assert len(files) == 2
        assert "file1.md" in files
        assert "file2.md" in files

        # Check that parse_metadata was called with the correct file paths
        mock_parse_metadata.assert_any_call("file1.md")
        mock_parse_metadata.assert_any_call("file2.md")


@pytest.fixture
def okp_mp():
    """Fixture for OKPMetadataProcessor."""
    return okp.OKPMetadataProcessor()


class TestOKPMetadataProcessor:
    """Test cases for OKPMetadataProcessor class."""

    @pytest.fixture(autouse=True)
    def mock_parse_metadata(self, mocker):
        """Mock the parse_metadata function for all tests in this class."""
        mocker.patch(
            "lightspeed_rag_content.okp.parse_metadata",
            return_value={
                "title": "Test Title",
                "extra": {
                    "reference_url": "https://example.com",
                },
            },
        )

    def test_url_function(self, okp_mp):
        """Test the URL function of OKPMetadataProcessor."""
        file_path = "/fake/path/errata_file.md"
        expected_url = "https://example.com"
        assert okp_mp.url_function(file_path) == expected_url

    def test_get_file_title(self, okp_mp):
        """Test the get_file_title function of OKPMetadataProcessor."""
        file_path = "/fake/path/errata_file.md"
        expected_title = "Test Title"
        assert okp_mp.get_file_title(file_path) == expected_title
