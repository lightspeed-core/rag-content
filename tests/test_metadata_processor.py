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
from lightspeed_rag_content.metadata_processor import DefaultMetadataProcessor


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
        mock_get = mocker.patch("lightspeed_rag_content.metadata_processor.requests.get")
        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = md_processor.ping_url(processor_data["url"])

        assert result is True
        assert mock_get.call_count == 1
        assert mock_get.call_args == ((processor_data["url"],), {"timeout": 30})

    def test_ping_url_404(self, md_processor, mocker, processor_data):
        """Test ping_url method returns False for HTTP 404 response."""
        mock_get = mocker.patch("lightspeed_rag_content.metadata_processor.requests.get")
        mock_response = mocker.MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = md_processor.ping_url(processor_data["url"])

        assert result is False
        assert mock_get.call_count == 3
        for args, kwargs in mock_get.call_args_list:
            assert args == (processor_data["url"],)
            assert kwargs == {"timeout": 30}

    def test_ping_url_exception(self, md_processor, mocker, processor_data):
        """Test ping_url method returns False when request raises exception."""
        mock_get = mocker.patch("lightspeed_rag_content.metadata_processor.requests.get")
        mock_get.side_effect = requests.exceptions.RequestException()

        result = md_processor.ping_url(processor_data["url"])

        assert result is False
        assert mock_get.call_count == 3
        for args, kwargs in mock_get.call_args_list:
            assert args == (processor_data["url"],)
            assert kwargs == {"timeout": 30}

    def test_get_file_title(self, md_processor, mocker, processor_data):
        """Test get_file_title method extracts title from file header."""
        mocker.patch(
            "builtins.open",
            new_callable=mocker.mock_open,
            read_data=f'# {processor_data["title"]}\n',
        )
        result = md_processor.get_file_title(processor_data["file_path"])

        assert processor_data["title"] == result

    def test_get_file_title_frontmatter(self, md_processor, mocker, processor_data):
        """Test get_file_title method extracts title from YAML frontmatter."""
        mocker.patch(
            "builtins.open",
            new_callable=mocker.mock_open,
            read_data="---\n",
        )
        mock_fm = mocker.MagicMock()
        mock_fm.get.return_value = processor_data["title"]
        mocker.patch(
            "lightspeed_rag_content.metadata_processor.frontmatter.load",
            return_value=mock_fm,
        )

        result = md_processor.get_file_title(processor_data["file_path"])

        assert processor_data["title"] == result
        mock_fm.get.assert_called_once_with("title", "")

    def test_get_file_title_exception(self, md_processor, mocker, processor_data):
        """Test get_file_title method handles file access exceptions."""
        mock_file = mocker.patch("builtins.open", new_callable=mocker.mock_open)
        mock_file.side_effect = Exception("boom")

        result = md_processor.get_file_title(processor_data["file_path"])

        assert "" == result
        mock_file.assert_called_once_with(processor_data["file_path"], "r", encoding="utf-8")

    def test_populate(self, md_processor, mocker, processor_data):
        """Test populate method returns complete metadata when URL is reachable."""
        mocker.patch.object(md_processor, "_get_frontmatter_url", return_value=None)
        mock_url_func = mocker.patch.object(md_processor, "url_function")
        mock_get_title = mocker.patch.object(md_processor, "get_file_title")
        mock_ping_url = mocker.patch.object(md_processor, "ping_url")

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
        mock_url_func.assert_called_once_with(processor_data["file_path"])
        mock_get_title.assert_called_once_with(processor_data["file_path"])
        mock_ping_url.assert_called_once_with(processor_data["url"])

    def test_populate_url_unreachable(self, md_processor, mocker, caplog, processor_data):
        """Test populate method handles unreachable URLs and logs warning."""
        mocker.patch.object(md_processor, "_get_frontmatter_url", return_value=None)
        mock_url_func = mocker.patch.object(md_processor, "url_function")
        mock_get_title = mocker.patch.object(md_processor, "get_file_title")
        mock_ping_url = mocker.patch.object(md_processor, "ping_url")

        mock_url_func.return_value = processor_data["url"]
        mock_get_title.return_value = processor_data["title"]
        mock_ping_url.return_value = False

        with caplog.at_level(logging.WARNING, logger=metadata_processor.__name__):
            result = md_processor.populate(processor_data["file_path"])

        expected_result = {
            "docs_url": processor_data["url"],
            "title": processor_data["title"],
            "url_reachable": False,
        }
        assert expected_result == result
        assert "URL not reachable" in caplog.text
        assert processor_data["url"] in caplog.text
        assert processor_data["title"] in caplog.text
        mock_url_func.assert_called_once_with(processor_data["file_path"])
        mock_get_title.assert_called_once_with(processor_data["file_path"])
        mock_ping_url.assert_called_once_with(processor_data["url"])

    def test_populate_frontmatter_url(self, md_processor, mocker, processor_data):
        """Test populate uses frontmatter URL instead of url_function when available."""
        frontmatter_url = "https://docs.example.com/page"
        mocker.patch.object(md_processor, "_get_frontmatter_url", return_value=frontmatter_url)
        mock_url_func = mocker.patch.object(md_processor, "url_function")
        mock_get_title = mocker.patch.object(
            md_processor, "get_file_title", return_value=processor_data["title"]
        )
        mock_ping_url = mocker.patch.object(md_processor, "ping_url", return_value=True)

        result = md_processor.populate(processor_data["file_path"])

        assert result == {
            "docs_url": frontmatter_url,
            "title": processor_data["title"],
            "url_reachable": True,
        }
        mock_url_func.assert_not_called()
        mock_get_title.assert_called_once_with(processor_data["file_path"])
        mock_ping_url.assert_called_once_with(frontmatter_url)

    def test_populate_hermetic_build_skips_ping(self, mocker, processor_data):
        """Test populate skips URL ping when hermetic_build=True."""
        processor = metadata_processor.MetadataProcessor(hermetic_build=True)
        mocker.patch.object(processor, "_get_frontmatter_url", return_value=None)
        mocker.patch.object(processor, "url_function", return_value=processor_data["url"])
        mocker.patch.object(processor, "get_file_title", return_value=processor_data["title"])
        mock_ping_url = mocker.patch.object(processor, "ping_url")

        result = processor.populate(processor_data["file_path"])

        assert result == {
            "docs_url": processor_data["url"],
            "title": processor_data["title"],
            "url_reachable": True,
        }
        mock_ping_url.assert_not_called()


class TestDefaultMetadataProcessor:
    """Test cases for the DefaultMetadataProcessor class."""

    def test_url_function_returns_basename(self):
        """Test url_function returns just the filename, not the full path."""
        processor = DefaultMetadataProcessor()
        result = processor.url_function("/some/nested/path/document.md")
        assert result == "document.md"

    def test_url_function_filename_only(self):
        """Test url_function with a bare filename (no directory) is unchanged."""
        processor = DefaultMetadataProcessor()
        result = processor.url_function("document.md")
        assert result == "document.md"

    def test_url_function_deeply_nested_path(self):
        """Test url_function extracts basename from a deeply nested path."""
        processor = DefaultMetadataProcessor()
        result = processor.url_function("/a/b/c/d/e/file.md")
        assert result == "file.md"

    def test_is_concrete_class(self):
        """Test DefaultMetadataProcessor can be instantiated without subclassing."""
        processor = DefaultMetadataProcessor()
        assert isinstance(processor, DefaultMetadataProcessor)
        assert isinstance(processor, metadata_processor.MetadataProcessor)

    def test_importable_from_public_api(self):
        """Test DefaultMetadataProcessor is accessible from the package public API."""
        import lightspeed_rag_content

        assert hasattr(lightspeed_rag_content, "DefaultMetadataProcessor")
        assert lightspeed_rag_content.DefaultMetadataProcessor is DefaultMetadataProcessor

    def test_populate_uses_url_function_as_url(self, mocker):
        """Test populate delegates to url_function for the document URL."""
        processor = DefaultMetadataProcessor()
        mocker.patch.object(processor, "ping_url", return_value=True)
        mocker.patch.object(processor, "get_file_title", return_value="My Doc")

        result = processor.populate("/docs/getting-started.md")

        assert result["docs_url"] == "getting-started.md"
        assert result["title"] == "My Doc"
        assert result["url_reachable"] is True

    def test_populate_url_unreachable(self, mocker, caplog):
        """Test populate marks url_reachable False and logs a warning when URL is unreachable."""
        processor = DefaultMetadataProcessor()
        mocker.patch.object(processor, "ping_url", return_value=False)
        mocker.patch.object(processor, "get_file_title", return_value="My Doc")

        with caplog.at_level(logging.WARNING, logger=metadata_processor.__name__):
            result = processor.populate("/docs/getting-started.md")

        assert result["url_reachable"] is False
        assert "URL not reachable" in caplog.text
