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
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from lightspeed_rag_content.asciidoc.asciidoctor_converter import (
    RUBY_ASCIIDOC_DIR,
    AsciidoctorConverter,
)


@pytest.fixture
def converter_data():
    """Fixture for AsciidoctorConverter test data."""
    return {
        "valid_attributes_file": """---
        foo: bar
        """,
        "invalid_attributes_file": """---
        [[]
        """,
        "text_converter_path": RUBY_ASCIIDOC_DIR.joinpath("asciidoc_text_converter.rb"),
        "input_file": Path("input.adoc"),
        "output_file": Path("output.txt"),
        "attributes_file": Path("attributes.yaml"),
        "asciidoctor_cmd": "/usr/bin/asciidoctor",
    }


class TestAsciidoctorConverter:
    def test_convert(self, mocker, converter_data):
        mock_which = mocker.patch(
            "lightspeed_rag_content.asciidoc.asciidoctor_converter.shutil.which"
        )
        mock_run = mocker.patch(
            "lightspeed_rag_content.asciidoc.asciidoctor_converter.subprocess.run"
        )
        mock_which.return_value = converter_data["asciidoctor_cmd"]
        mocker.patch(
            "builtins.open",
            mocker.mock_open(read_data=converter_data["valid_attributes_file"]),
        )
        adoc_text_converter = AsciidoctorConverter(
            attributes_file=converter_data["attributes_file"]
        )
        adoc_text_converter.convert(
            converter_data["input_file"], converter_data["output_file"]
        )

        mock_run.assert_called_with(
            [
                converter_data["asciidoctor_cmd"],
                "-a",
                "foo=bar",
                "-r",
                str(converter_data["text_converter_path"].absolute()),
                "-b",
                "text",
                "-o",
                str(converter_data["output_file"].absolute()),
                "--trace",
                "--quiet",
                str(converter_data["input_file"].absolute()),
            ],
            check=True,
            capture_output=True,
        )

    def test_convert_custom_converter(self, mocker, converter_data):
        mock_which = mocker.patch(
            "lightspeed_rag_content.asciidoc.asciidoctor_converter.shutil.which"
        )
        mock_run = mocker.patch(
            "lightspeed_rag_content.asciidoc.asciidoctor_converter.subprocess.run"
        )
        mock_which.return_value = converter_data["asciidoctor_cmd"]
        custom_converter = Path("custom_converter")
        adoc_text_converter = AsciidoctorConverter(converter_file=custom_converter)
        adoc_text_converter.convert(
            converter_data["input_file"], converter_data["output_file"]
        )

        mock_run.assert_called_with(
            [
                converter_data["asciidoctor_cmd"],
                "-r",
                str(custom_converter.absolute()),
                "-b",
                "text",
                "-o",
                str(converter_data["output_file"].absolute()),
                "--trace",
                "--quiet",
                str(converter_data["input_file"].absolute()),
            ],
            check=True,
            capture_output=True,
        )

    def test_convert_overwrite_output_file(self, mocker, converter_data, caplog):
        mock_which = mocker.patch(
            "lightspeed_rag_content.asciidoc.asciidoctor_converter.shutil.which"
        )
        mocker.patch(
            "lightspeed_rag_content.asciidoc.asciidoctor_converter.subprocess.run"
        )
        mock_which.return_value = converter_data["asciidoctor_cmd"]
        adoc_text_converter = AsciidoctorConverter()

        mock_output_file = Mock()
        mock_output_file.exists.return_value = True

        with caplog.at_level(logging.WARNING):
            adoc_text_converter.convert(converter_data["input_file"], mock_output_file)
            assert "WARNING" in caplog.text

    def test_convert_new_output_file(self, mocker, converter_data):
        mock_which = mocker.patch(
            "lightspeed_rag_content.asciidoc.asciidoctor_converter.shutil.which"
        )
        mocker.patch(
            "lightspeed_rag_content.asciidoc.asciidoctor_converter.subprocess.run"
        )
        mock_which.return_value = converter_data["asciidoctor_cmd"]
        adoc_text_converter = AsciidoctorConverter()

        output_file = Mock()
        output_file.exists.return_value = False
        output_file.absolute.return_value = "/output.txt"

        adoc_text_converter.convert(converter_data["input_file"], output_file)
        output_file.parent.mkdir.assert_called_once()

    def test__get_converter_file(self):
        converter_file = AsciidoctorConverter._get_converter_file("text")
        assert converter_file == RUBY_ASCIIDOC_DIR.joinpath(
            "asciidoc_text_converter.rb"
        )

    def test__get_converter_file_asciidoctor_built_in_format(self):
        converter_file = AsciidoctorConverter._get_converter_file("html5")
        assert converter_file is None

    def test__get_converter_file_invalid_format(self):
        with pytest.raises(FileNotFoundError):
            AsciidoctorConverter._get_converter_file("invalid")

    def test__get_asciidoctor_path_missing(self, mocker):
        mock_which = mocker.patch(
            "lightspeed_rag_content.asciidoc.asciidoctor_converter.shutil.which"
        )
        mock_which.return_value = ""
        with pytest.raises(FileNotFoundError):
            AsciidoctorConverter()

    def test__get_attribute_list_valid_yaml(self, mocker, converter_data):
        m = mocker.patch(
            "builtins.open",
            mocker.mock_open(read_data=converter_data["valid_attributes_file"]),
        )
        AsciidoctorConverter._get_attribute_list(Path("valid.yaml"))
        m.assert_called_once()

    def test__get_attribute_list_invalid_yaml(self, mocker, converter_data):
        mocker.patch(
            "builtins.open",
            mocker.mock_open(read_data=converter_data["invalid_attributes_file"]),
        )
        with pytest.raises(yaml.YAMLError):
            AsciidoctorConverter._get_attribute_list(Path("invalid.yaml"))

    def test__get_attribute_list_empty_yaml(self, mocker):
        mocker.patch("builtins.open", mocker.mock_open(read_data=""))
        attributes = AsciidoctorConverter._get_attribute_list(Path("non_existing.yaml"))
        assert attributes == []
