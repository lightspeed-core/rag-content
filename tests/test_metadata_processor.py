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

import logging

import pytest
import requests

from lightspeed_rag_content import metadata_processor


@pytest.fixture
def md_processor():
    """Fixture for MetadataProcessor."""
    return metadata_processor.MetadataProcessor()


@pytest.fixture
def processor_data():
    """Fixture for MetadataProcessor test data."""
    return {
        "file_path": "/fake/path/road-core",
        "url": "https://www.openstack.org",
        "title": "Road-Core title",
    }


class TestMetadataProcessor:
    """Test cases for the MetadataProcessor class."""

    def test_ping_url_200(self, md_processor, mocker, processor_data):
        """Test ping_url method returns True for successful HTTP 200 response."""
        mock_get = mocker.patch("requests.get")
        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = md_processor.ping_url(processor_data["url"])

        assert result is True

    def test_ping_url_404(self, md_processor, mocker, processor_data):
        """Test ping_url method returns False for HTTP 404 response."""
        mock_get = mocker.patch("requests.get")
        mock_response = mocker.MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = md_processor.ping_url(processor_data["url"])

        assert result is False

    def test_ping_url_exception(self, md_processor, mocker, processor_data):
        """Test ping_url method returns False when request raises exception."""
        mock_get = mocker.patch("requests.get")
        mock_get.side_effect = requests.exceptions.RequestException()

        result = md_processor.ping_url(processor_data["url"])

        assert result is False

    def test_get_file_title(self, md_processor, mocker, processor_data):
        """Test get_file_title method extracts title from file header."""
        mocker.patch(
            "builtins.open",
            new_callable=mocker.mock_open,
            read_data=f'# {processor_data["title"]}',
        )
        result = md_processor.get_file_title(processor_data["file_path"])

        assert processor_data["title"] == result

    def test_get_file_title_exception(self, md_processor, mocker, processor_data):
        """Test get_file_title method handles file access exceptions."""
        mock_file = mocker.patch("builtins.open", new_callable=mocker.mock_open)
        mock_file.side_effect = Exception("boom")

        result = md_processor.get_file_title(processor_data["file_path"])

        assert "" == result

    def test_populate(self, md_processor, mocker, processor_data):
        """Test populate method returns complete metadata when URL is reachable."""
        mock_url_func = mocker.patch.object(
            metadata_processor.MetadataProcessor, "url_function"
        )
        mock_get_title = mocker.patch.object(
            metadata_processor.MetadataProcessor, "get_file_title"
        )
        mock_ping_url = mocker.patch.object(
            metadata_processor.MetadataProcessor, "ping_url"
        )

        mock_url_func.return_value = processor_data["url"]
        mock_get_title.return_value = processor_data["title"]
        mock_ping_url.return_value = True

        result = md_processor.populate(processor_data["file_path"])

        expected_result = {
            "docs_url": processor_data["url"],
            "title": processor_data["title"],
            "url_reachable": True,
        }
        assert expected_result == result

    def test_populate_url_unreachable(
        self, md_processor, mocker, caplog, processor_data
    ):
        """Test populate method handles unreachable URLs and logs warning."""
        mock_url_func = mocker.patch.object(
            metadata_processor.MetadataProcessor, "url_function"
        )
        mock_get_title = mocker.patch.object(
            metadata_processor.MetadataProcessor, "get_file_title"
        )
        mock_ping_url = mocker.patch.object(
            metadata_processor.MetadataProcessor, "ping_url"
        )

        mock_url_func.return_value = processor_data["url"]
        mock_get_title.return_value = processor_data["title"]
        mock_ping_url.return_value = False

        with caplog.at_level(logging.WARNING):
            result = md_processor.populate(processor_data["file_path"])

        expected_result = {
            "docs_url": processor_data["url"],
            "title": processor_data["title"],
            "url_reachable": False,
        }
        assert expected_result == result
        assert "URL not reachable" in caplog.text
