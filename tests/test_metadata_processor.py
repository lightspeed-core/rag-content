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

    def setUp(self):
        self.md_processor = metadata_processor.MetadataProcessor()
        self.file_path = "/fake/path/road-core"
        self.url = "https://www.openstack.org"
        self.title = "Road-Core title"

    @mock.patch("lightspeed_rag_content.metadata_processor.requests.get")
    def test_ping_url_200(self, mock_get):
        mock_response = mock.MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = self.md_processor.ping_url(self.url)

        self.assertTrue(result)

    @mock.patch("lightspeed_rag_content.metadata_processor.requests.get")
    def test_ping_url_404(self, mock_get):
        mock_response = mock.MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = self.md_processor.ping_url(self.url)

        self.assertFalse(result)

    @mock.patch("lightspeed_rag_content.metadata_processor.requests.get")
    def test_ping_url_exception(self, mock_get):
        mock_get.side_effect = requests.exceptions.RequestException()

        result = self.md_processor.ping_url(self.url)

        self.assertFalse(result)

    @mock.patch(
        "builtins.open", new_callable=mock.mock_open, read_data="# Road-Core title"
    )
    def test_get_file_title(self, mock_file):
        result = self.md_processor.get_file_title(self.file_path)

        self.assertEqual(self.title, result)

    @mock.patch("builtins.open", new_callable=mock.mock_open)
    def test_get_file_title_exception(self, mock_file):
        mock_file.side_effect = Exception("boom")

        result = self.md_processor.get_file_title(self.file_path)

        self.assertEqual("", result)

    @mock.patch.object(metadata_processor.MetadataProcessor, "ping_url")
    @mock.patch.object(metadata_processor.MetadataProcessor, "get_file_title")
    @mock.patch.object(metadata_processor.MetadataProcessor, "url_function")
    def test_populate(self, mock_url_func, mock_get_title, mock_ping_url):
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
