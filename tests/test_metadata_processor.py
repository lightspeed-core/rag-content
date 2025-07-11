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

import requests

from lightspeed_rag_content import metadata_processor


class TestMetadataProcessor(unittest.TestCase):
    """Test cases for the MetadataProcessor class.

    This test suite covers the functionality of the MetadataProcessor class,
    including URL validation, file title extraction, and metadata population.
    """

    def setUp(self):
        """Set up test fixtures and common test data.

        Initializes a MetadataProcessor instance and defines common test data
        used across multiple test methods.
        """
        self.md_processor = metadata_processor.MetadataProcessor()
        self.file_path = "/fake/path/road-core"
        self.url = "https://www.openstack.org"
        self.title = "Road-Core title"

    @mock.patch("lightspeed_rag_content.metadata_processor.requests.get")
    def test_ping_url_200(self, mock_get):
        """Test ping_url method returns True for successful HTTP 200 response.

        Verifies that when a URL returns a 200 status code, the ping_url method
        correctly returns True indicating the URL is reachable.
        """
        mock_response = mock.MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = self.md_processor.ping_url(self.url)

        self.assertTrue(result)

    @mock.patch("lightspeed_rag_content.metadata_processor.requests.get")
    def test_ping_url_404(self, mock_get):
        """Test ping_url method returns False for HTTP 404 response.

        Verifies that when a URL returns a 404 status code, the ping_url method
        correctly returns False indicating the URL is not reachable.
        """
        mock_response = mock.MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = self.md_processor.ping_url(self.url)

        self.assertFalse(result)

    @mock.patch("lightspeed_rag_content.metadata_processor.requests.get")
    def test_ping_url_exception(self, mock_get):
        """Test ping_url method returns False when request raises exception.

        Verifies that when a request exception occurs (network error, timeout, etc.),
        the ping_url method handles it gracefully and returns False.
        """
        mock_get.side_effect = requests.exceptions.RequestException()

        result = self.md_processor.ping_url(self.url)

        self.assertFalse(result)

    @mock.patch(
        "builtins.open", new_callable=mock.mock_open, read_data="# Road-Core title"
    )
    def test_get_file_title(self, mock_file):
        """Test get_file_title method extracts title from file header.

        Verifies that the get_file_title method can successfully extract
        a title from a file's first line when it starts with '#'.
        """
        result = self.md_processor.get_file_title(self.file_path)

        self.assertEqual(self.title, result)

    @mock.patch("builtins.open", new_callable=mock.mock_open)
    def test_get_file_title_exception(self, mock_file):
        """Test get_file_title method handles file access exceptions.

        Verifies that when file reading fails (file not found, permission error, etc.),
        the get_file_title method returns an empty string.
        """
        mock_file.side_effect = Exception("boom")

        result = self.md_processor.get_file_title(self.file_path)

        self.assertEqual("", result)

    @mock.patch.object(metadata_processor.MetadataProcessor, "ping_url")
    @mock.patch.object(metadata_processor.MetadataProcessor, "get_file_title")
    @mock.patch.object(metadata_processor.MetadataProcessor, "url_function")
    def test_populate(self, mock_url_func, mock_get_title, mock_ping_url):
        """Test populate method returns complete metadata when URL is reachable.

        Verifies that the populate method correctly combines URL generation,
        title extraction, and URL validation to return a complete metadata
        dictionary when the URL is reachable.
        """
        mock_url_func.return_value = self.url
        mock_get_title.return_value = self.title
        mock_ping_url.return_value = True

        result = self.md_processor.populate(self.file_path)

        expected_result = {
            "docs_url": self.url,
            "title": self.title,
            "url_reachable": True,
        }
        self.assertEqual(expected_result, result)

    @mock.patch.object(metadata_processor.MetadataProcessor, "ping_url")
    @mock.patch.object(metadata_processor.MetadataProcessor, "get_file_title")
    @mock.patch.object(metadata_processor.MetadataProcessor, "url_function")
    def test_populate_url_unreachable(
        self, mock_url_func, mock_get_title, mock_ping_url
    ):
        """Test populate method handles unreachable URLs and logs warning.

        Verifies that when a generated URL is not reachable, the populate method
        returns metadata with url_reachable=False and logs an appropriate warning.
        """
        mock_url_func.return_value = self.url
        mock_get_title.return_value = self.title
        mock_ping_url.return_value = False

        with self.assertLogs(
            "lightspeed_rag_content.metadata_processor", level="WARNING"
        ) as log:
            result = self.md_processor.populate(self.file_path)

        expected_result = {
            "docs_url": self.url,
            "title": self.title,
            "url_reachable": False,
        }
        self.assertEqual(expected_result, result)
        self.assertIn("URL not reachable", log.output[0])
