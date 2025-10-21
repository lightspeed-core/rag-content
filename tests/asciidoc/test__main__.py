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
import logging
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from lightspeed_rag_content.asciidoc.__main__ import (
    get_argument_parser,
    main_convert,
    main_get_structure,
)
from lightspeed_rag_content.asciidoc.asciidoctor_converter import RUBY_ASCIIDOC_DIR


@pytest.fixture
def main_data():
    """Fixture for __main__ test data."""
    return {
        "asciidoctor_cmd": "/usr/bin/asciidoctor",
        "input_file": Path("input.adoc"),
        "output_file": Path("output.adoc"),
        "text_converter_file": RUBY_ASCIIDOC_DIR.joinpath("asciidoc_text_converter.rb"),
        "structure_dumper_file": RUBY_ASCIIDOC_DIR.joinpath(
            "asciidoc_structure_dumper.rb"
        ),
    }


def get_mock_parsed_args(main_data) -> Mock:
    mock_args = Mock()
    mock_args.input_file = main_data["input_file"]
    mock_args.output_file = main_data["output_file"]
    mock_args.converter_file = main_data["text_converter_file"]
    mock_args.attributes_file = None
    mock_args.target_format = "text"

    return mock_args


class Test__main__:
    def test_main_convert(self, mocker, main_data):
        mock_which = mocker.patch(
            "lightspeed_rag_content.asciidoc.asciidoctor_converter.shutil.which"
        )
        mock_run = mocker.patch(
            "lightspeed_rag_content.asciidoc.asciidoctor_converter.subprocess.run"
        )
        mock_which.return_value = main_data["asciidoctor_cmd"]
        mock_args = get_mock_parsed_args(main_data)
        main_convert(mock_args)

        mock_run.assert_called_with(
            [
                "/usr/bin/asciidoctor",
                "-r",
                str(main_data["text_converter_file"].absolute()),
                "-b",
                "text",
                "-o",
                str(main_data["output_file"].absolute()),
                "--trace",
                "--quiet",
                str(main_data["input_file"].absolute()),
            ],
            check=True,
            capture_output=True,
        )

    def test_main_convert_incorrect_cmd_error(self, mocker, main_data):
        mock_which = mocker.patch(
            "lightspeed_rag_content.asciidoc.asciidoctor_converter.shutil.which"
        )
        mock_run = mocker.patch(
            "lightspeed_rag_content.asciidoc.asciidoctor_converter.subprocess.run"
        )
        mock_which.return_value = main_data["asciidoctor_cmd"]
        mock_run.side_effect = subprocess.CalledProcessError(
            cmd=main_data["asciidoctor_cmd"], returncode=1
        )
        mock_args = get_mock_parsed_args(main_data)

        with pytest.raises(SystemExit) as e:
            main_convert(mock_args)
            assert e.value.code != 0

    def test_main_convert_missing_asciidoctor_cmd(self, mocker, main_data):
        mock_which = mocker.patch(
            "lightspeed_rag_content.asciidoc.asciidoctor_converter.shutil.which"
        )
        mock_which.return_value = ""
        mock_args = get_mock_parsed_args(main_data)

        with pytest.raises(SystemExit) as e:
            main_convert(mock_args)
            assert e.value.code != 0

    def test_main_get_structure(self, mocker, main_data):
        mock_which = mocker.patch(
            "lightspeed_rag_content.asciidoc.asciidoctor_converter.shutil.which"
        )
        mock_run = mocker.patch(
            "lightspeed_rag_content.asciidoc.asciidoctor_converter.subprocess.run"
        )
        mock_which.return_value = "/usr/bin/ruby"
        mock_args = Mock()
        mock_args.input_file = main_data["input_file"]

        main_get_structure(mock_args)
        mock_run.assert_called_with(
            [
                "/usr/bin/ruby",
                str(main_data["structure_dumper_file"]),
                str(main_data["input_file"].absolute()),
            ],
            check=True,
        )

    def test_main_incorrect_asciidoctor_cmd(self, mocker, main_data):
        mock_run = mocker.patch(
            "lightspeed_rag_content.asciidoc.asciidoctor_converter.subprocess.run"
        )
        mock_run.side_effect = subprocess.CalledProcessError(
            cmd=main_data["asciidoctor_cmd"], returncode=1
        )
        mock_args = Mock()
        mock_args.input_file = main_data["input_file"]

        with pytest.raises(SystemExit) as e:
            main_get_structure(mock_args)
            assert e.value.code != 0

    def test_main_missing_asciidoctor_cmd(self, mocker, main_data, caplog):
        mock_which = mocker.patch(
            "lightspeed_rag_content.asciidoc.asciidoctor_converter.shutil.which"
        )
        mock_which.return_value = ""
        mock_args = Mock()
        mock_args.input_file = main_data["input_file"]

        with pytest.raises(SystemExit) as e:
            with caplog.at_level(logging.ERROR):
                main_get_structure(mock_args)
                assert e.value.code != 0
                assert "ERROR" in caplog.text

    def test_get_argument_parser(self):
        args = get_argument_parser()

        assert isinstance(args, argparse.ArgumentParser)
