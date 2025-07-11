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
import unittest

from lightspeed_rag_content import utils


class TestUtils(unittest.TestCase):
    """Test cases for the metadata processor."""

    def test_get_common_arg_parser(self):
        """Test we get a valid arg parser from get_common_arg_parser."""
        parser = utils.get_common_arg_parser()

        self.assertIsInstance(parser, argparse.ArgumentParser)

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
            self.assertEqual(args.vector_store_type, vector_store_type)

    def test_arg_parser_vector_store_type_incorrect(self):
        """Test arg parser fails with incorrect vector-store-type."""
        parser = utils.get_common_arg_parser()

        for vector_store_type in ("faisss", "lamastack-faiss"):
            self.assertRaises(
                SystemExit,
                parser.parse_args,
                ["--vector-store-type", vector_store_type],
            )

    def test_arg_parser_auto_chunking(self):
        """Test that arg parser's manual chunking toggle works."""
        parser = utils.get_common_arg_parser()

        args = parser.parse_args(["--auto-chunking"])
        self.assertFalse(args.manual_chunking)

    def test_arg_parser_auto_chunking_default(self):
        """Test that manual chunking is the default in the arg parser."""
        parser = utils.get_common_arg_parser()

        args = parser.parse_args([])
        self.assertTrue(args.manual_chunking)
