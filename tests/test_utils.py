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
import pytest

from lightspeed_rag_content import utils


class TestUtils:
    """Test cases for the metadata processor."""

    def test_get_common_arg_parser(self):
        """Test we get a valid arg parser from get_common_arg_parser."""
        parser = utils.get_common_arg_parser()

        assert isinstance(parser, argparse.ArgumentParser)

    def test_arg_parser_vector_store_type(self):
        """Test arg parser works with valid vector-store-type."""
        parser = utils.get_common_arg_parser()

        for vector_store_type in (
            "faiss",
            "postgres",
            "llamastack-faiss",
            "llamastack-sqlite-vec",
        ):
            args = parser.parse_args(["--vector-store-type", vector_store_type])
            assert args.vector_store_type == vector_store_type

    def test_arg_parser_vector_store_type_incorrect(self):
        """Test arg parser fails with incorrect vector-store-type."""
        parser = utils.get_common_arg_parser()

        for vector_store_type in ("faisss", "lamastack-faiss"):
            with pytest.raises(SystemExit):
                parser.parse_args(["--vector-store-type", vector_store_type])

    def test_arg_parser_auto_chunking(self):
        """Test that arg parser's manual chunking toggle works."""
        parser = utils.get_common_arg_parser()

        args = parser.parse_args(["--auto-chunking"])
        assert not args.manual_chunking

    def test_arg_parser_auto_chunking_default(self):
        """Test that manual chunking is the default in the arg parser."""
        parser = utils.get_common_arg_parser()

        args = parser.parse_args([])
        assert args.manual_chunking

    def test_run_cli_command_unknown_command(self):
        """Test run_cli_command raises SystemExit for unknown commands."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        subparsers.add_parser("known")

        handlers = {"known": lambda args: None}

        # Simulate parsing an unknown command by mocking parse_args
        with pytest.raises(SystemExit) as exc_info:
            parser_with_unknown = argparse.ArgumentParser()
            parser_with_unknown.parse_args = lambda: argparse.Namespace(
                command="unknown"
            )
            utils.run_cli_command(parser_with_unknown, handlers)

        assert "Unknown command: unknown" in str(exc_info.value)

    def test_run_cli_command_executes_handler(self, mocker):
        """Test run_cli_command executes the correct handler."""
        mock_handler = mocker.Mock()
        parser = argparse.ArgumentParser()
        parser.parse_args = lambda: argparse.Namespace(command="test")

        utils.run_cli_command(parser, {"test": mock_handler})

        mock_handler.assert_called_once()

    def test_setup_cli_logging(self):
        """Test setup_cli_logging returns a logger."""
        import logging

        logger = utils.setup_cli_logging("test_package")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_package"

    def test_add_input_file_argument(self):
        """Test add_input_file_argument adds the expected argument."""
        from pathlib import Path

        parser = argparse.ArgumentParser()
        utils.add_input_file_argument(parser, help_text="Test file input")

        args = parser.parse_args(["-i", "/tmp/test.txt"])
        assert args.input_file == Path("/tmp/test.txt")
