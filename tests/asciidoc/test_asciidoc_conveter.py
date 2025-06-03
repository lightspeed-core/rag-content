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
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import yaml

from lightspeed_rag_content.asciidoc.asciidoctor_converter import (
    RUBY_ASCIIDOC_DIR,
    AsciidoctorConverter,
)


class TestAsciidoctorConverter(unittest.TestCase):
    def setUp(self):
        super().setUp()

        self.valid_attributes_file = """---
        foo: bar
        """

        self.invalid_attributes_file = """---
        [[]
        """

        self.text_converter_path = RUBY_ASCIIDOC_DIR.joinpath(
            "asciidoc_text_converter.rb"
        )
        self.input_file = Path("input.adoc")
        self.output_file = Path("output.txt")
        self.attributes_file = Path("attributes.yaml")
        self.asciidoctor_cmd = "/usr/bin/asciidoctor"

    @patch("lightspeed_rag_content.asciidoc.asciidoctor_converter.subprocess.run")
    @patch("lightspeed_rag_content.asciidoc.asciidoctor_converter.shutil.which")
    def test_convert(self, mock_which, mock_run):
        mock_which.return_value = self.asciidoctor_cmd
        with patch("builtins.open", mock_open(read_data=self.valid_attributes_file)):
            adoc_text_converter = AsciidoctorConverter(
                attributes_file=self.attributes_file
            )
            adoc_text_converter.convert(self.input_file, self.output_file)

        mock_run.assert_called_with(
            [
                self.asciidoctor_cmd,
                "-a",
                "foo=bar",
                "-r",
                str(self.text_converter_path.absolute()),
                "-b",
                "text",
                "-o",
                str(self.output_file.absolute()),
                "--trace",
                "--quiet",
                str(self.input_file.absolute()),
            ],
            check=True,
            capture_output=True,
        )

    @patch("lightspeed_rag_content.asciidoc.asciidoctor_converter.subprocess.run")
    @patch("lightspeed_rag_content.asciidoc.asciidoctor_converter.shutil.which")
    def test_convert_custom_converter(self, mock_which, mock_run):
        mock_which.return_value = self.asciidoctor_cmd
        custom_converter = Path("custom_converter")
        adoc_text_converter = AsciidoctorConverter(converter_file=custom_converter)
        adoc_text_converter.convert(self.input_file, self.output_file)

        mock_run.assert_called_with(
            [
                self.asciidoctor_cmd,
                "-r",
                str(custom_converter.absolute()),
                "-b",
                "text",
                "-o",
                str(self.output_file.absolute()),
                "--trace",
                "--quiet",
                str(self.input_file.absolute()),
            ],
            check=True,
            capture_output=True,
        )

    @patch("lightspeed_rag_content.asciidoc.asciidoctor_converter.subprocess.run")
    @patch("lightspeed_rag_content.asciidoc.asciidoctor_converter.shutil.which")
    def test_convert_overwrite_output_file(self, mock_which, mock_run):
        mock_which.return_value = self.asciidoctor_cmd
        adoc_text_converter = AsciidoctorConverter()

        mock_output_file = Mock()
        mock_output_file.exists.return_value = True

        with self.assertLogs() as logger:
            adoc_text_converter.convert(self.input_file, mock_output_file)
            warning_msgs = [output for output in logger.output if "WARNING" in output]
            self.assertTrue(len(warning_msgs) > 0)

    @patch("lightspeed_rag_content.asciidoc.asciidoctor_converter.subprocess.run")
    @patch("lightspeed_rag_content.asciidoc.asciidoctor_converter.shutil.which")
    def test_convert_new_output_file(self, mock_which, mock_run):
        mock_which.return_value = self.asciidoctor_cmd
        adoc_text_converter = AsciidoctorConverter()

        output_file = Mock()
        output_file.exists.return_value = False
        output_file.absolute.return_value = "/output.txt"

        adoc_text_converter.convert(self.input_file, output_file)
        output_file.parent.mkdir.assert_called_once()

    def test__get_converter_file(self):
        converter_file = AsciidoctorConverter._get_converter_file("text")
        self.assertEqual(
            converter_file, RUBY_ASCIIDOC_DIR.joinpath("asciidoc_text_converter.rb")
        )

    def test__get_converter_file_asciidoctor_built_in_format(self):
        converter_file = AsciidoctorConverter._get_converter_file("html5")
        self.assertEqual(converter_file, None)

    def test__get_converter_file_invalid_format(self):
        with self.assertRaises(FileNotFoundError):
            AsciidoctorConverter._get_converter_file("invalid")

    @patch("lightspeed_rag_content.asciidoc.asciidoctor_converter.shutil.which")
    def test__get_asciidoctor_path_missing(self, mock_which):
        mock_which.return_value = ""
        with self.assertRaises(FileNotFoundError):
            AsciidoctorConverter()

    def test__get_attribute_list_valid_yaml(self):
        with patch(
            "builtins.open", mock_open(read_data=self.valid_attributes_file)
        ) as m:
            AsciidoctorConverter._get_attribute_list("valid.yaml")
            m.assert_called_once()

    def test__get_attribute_list_invalid_yaml(self):
        with patch("builtins.open", mock_open(read_data=self.invalid_attributes_file)):
            with self.assertRaises(yaml.YAMLError):
                AsciidoctorConverter._get_attribute_list("invalid.yaml")

    def test__get_attribute_list_empty_yaml(self):
        with patch("builtins.open", mock_open(read_data="")):
            attributes = AsciidoctorConverter._get_attribute_list("non_existing.yaml")
            self.assertEqual(attributes, [])
