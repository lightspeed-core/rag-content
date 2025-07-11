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

import functools
from unittest import TestCase as UTTestCase
from unittest import mock

from llama_index.core.embeddings.mock_embed_model import MockEmbedding


class RagMockEmbedding(MockEmbedding):
    """Mock class for HuggingFaceEmbedding."""

    def __init__(self, model_name="ABC"):
        """Initialize the mock class."""
        super().__init__(embed_dim=768, model_name=model_name)

    def get_text_embedding(self, text):
        """Simulate the text embedding with the right size."""
        return "A" * 768


def subtest(key, values):
    """Trivial decorator to create subtests.

    This helps with simple unit test parameterization.

    Example:
    -------
    @subtest("vector_store_type", ("llamastack-faiss", "llamastack-sqlite"))
    def test_init_llama_stack(self, vector_store_type):
        ...

    """

    def decorator(func):
        @functools.wraps(func)
        def wrap(self, *args, **kwargs):
            for value in values:
                params = {key: value}
                with self.subTest(**params):
                    return func(self, *args, **kwargs, **params)

        return wrap

    return decorator


class TestCase(UTTestCase):
    """Basic project's unit test case class."""

    def patch_object(self, obj, attr_name, *args, **kwargs):
        """Use python mock to mock an object attribute.

        Mocks the specified objects attribute with the given value.
        Automatically performs 'addCleanup' for the mock.
        """
        patcher = mock.patch.object(obj, attr_name, *args, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def patch(self, path, *args, **kwargs):
        """Use python mock to mock a path with automatic cleanup."""
        patcher = mock.patch(path, *args, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result
