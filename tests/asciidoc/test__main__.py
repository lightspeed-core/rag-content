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
import argparse
import subprocess
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from lightspeed_rag_content.asciidoc.__main__ import (
    get_argument_parser,
    main_convert,
    main_get_structure,
)
from lightspeed_rag_content.asciidoc.asciidoctor_converter import RUBY_ASCIIDOC_DIR


class Test__main__(unittest.TestCase):
    def setUp(self):
        super().setUp()

        self.asciidoctor_cmd = "/usr/bin/asciidoctor"
        self.input_file = Path("input.adoc")
        self.output_file = Path("output.adoc")
        self.text_converter_file = RUBY_ASCIIDOC_DIR.joinpath("asciidoc_text_converter.rb")
        self.structure_dumper_file = RUBY_ASCIIDOC_DIR.joinpath("asciidoc_structure_dumper.rb")

    def get_mock_parsed_args(self) -> Mock:
        mock_args = Mock()
        mock_args.input_file = self.input_file
        mock_args.output_file = self.output_file
        mock_args.converter_file = self.text_converter_file
        mock_args.attributes_file = None
        mock_args.target_format = "text"

        return mock_args

    @patch("lightspeed_rag_content.asciidoc.asciidoctor_converter.subprocess.run")
    @patch("lightspeed_rag_content.asciidoc.asciidoctor_converter.shutil.which")
    def test_main_convert(self, mock_which, mock_run):
        mock_which.return_value = self.asciidoctor_cmd
        mock_args = self.get_mock_parsed_args()
        main_convert(mock_args)

        mock_run.assert_called_with(
            [
                "/usr/bin/asciidoctor",
                "-r",
                str(self.text_converter_file.absolute()),
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
    def test_main_convert_incorrect_cmd_error(self, mock_which, mock_run):
        mock_which.return_value = self.asciidoctor_cmd
        mock_run.side_effect = subprocess.CalledProcessError(cmd=self.asciidoctor_cmd, returncode=1)
        mock_args = self.get_mock_parsed_args()

        with self.assertRaises(SystemExit) as e:
            main_convert(mock_args)
            self.assertNotEqual(e.exception.code, 0)

    @patch("lightspeed_rag_content.asciidoc.asciidoctor_converter.shutil.which")
    def test_main_convert_missing_asciidoctor_cmd(self, mock_which):
        mock_which.return_value = ""
        mock_args = self.get_mock_parsed_args()

        with self.assertRaises(SystemExit) as e:
            main_convert(mock_args)
            self.assertNotEqual(e.exception.code, 0)

    @patch("lightspeed_rag_content.asciidoc.asciidoctor_converter.subprocess.run")
    @patch("lightspeed_rag_content.asciidoc.asciidoctor_converter.shutil.which")
    def test_main_get_structure(self, mock_which, mock_run):
        mock_which.return_value = "/usr/bin/ruby"
        mock_args = Mock()
        mock_args.input_file = self.input_file

        main_get_structure(mock_args)
        mock_run.assert_called_with(
            [
                "/usr/bin/ruby",
                str(self.structure_dumper_file),
                str(self.input_file.absolute()),
            ],
            check=True,
        )

    @patch("lightspeed_rag_content.asciidoc.asciidoctor_converter.subprocess.run")
    def test_main_incorrect_asciidoctor_cmd(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(cmd=self.asciidoctor_cmd, returncode=1)
        mock_args = Mock()
        mock_args.input_file = self.input_file

        with self.assertRaises(SystemExit) as e:
            main_get_structure(mock_args)
            self.assertNotEqual(e.exception.code, 0)

    @patch("lightspeed_rag_content.asciidoc.asciidoctor_converter.shutil.which")
    def test_main_missing_asciidoctor_cmd(self, mock_which):
        mock_which.return_value = ""
        mock_args = Mock()
        mock_args.input_file = self.input_file

        with self.assertRaises(SystemExit) as e:
            with self.assertLogs() as logger:
                main_get_structure(mock_args)
                self.assertNotEqual(e.exception.code, 0)

                error_msgs = [output for output in logger.output if "ERROR" in output]
                self.assertTrue(len(error_msgs) > 0)

    def test_get_argument_parser(self):
        args = get_argument_parser()

        self.assertIsInstance(args, argparse.ArgumentParser)
