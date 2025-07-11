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

import os
import unittest
from unittest import mock

from lightspeed_rag_content import document_processor
from tests import utils


class TestConfig(unittest.TestCase):
    def test_config(self):
        config = document_processor._Config(
            chunk_size=380,
            chunk_overlap=0,
            model_name="sentence-transformers/all-mpnet-base-v2",
            embeddings_model_dir="./embeddings_model",
        )
        self.assertEqual(380, config.chunk_size)
        self.assertEqual(0, config.chunk_overlap)
        self.assertEqual("sentence-transformers/all-mpnet-base-v2",
                         config.model_name)
        self.assertEqual("./embeddings_model", config.embeddings_model_dir)


@mock.patch("lightspeed_rag_content.document_processor.HuggingFaceEmbedding",
            new=utils.MockEmbedding)
class TestDocumentProcessor(utils.TestCase):
    def setUp(self):
        self.params = dict(
            chunk_size=380,
            chunk_overlap=0,
            model_name="sentence-transformers/all-mpnet-base-v2",
            embeddings_model_dir="./embeddings_model",
            num_workers=10,
        )
        self.log = self.patch_object(document_processor, "LOG")
        self.indexdb = self.patch_object(document_processor, "_LlamaIndexDB")
        self.llamadb = self.patch_object(document_processor, "_LlamaStackDB")

    def test_init_default(self):
        doc_processor = document_processor.DocumentProcessor(**self.params)

        self.log.warning.assert_not_called()
        self.indexdb.assert_called_once_with(doc_processor.config)

        self.assertIsNotNone(doc_processor)

        self.params.update(  # Add default values
            embedding_dimension=None,  # Not calculated because class is mocked
            manual_chunking=True,
            table_name=None,
            vector_store_type="faiss",
        )
        self.assertEqual(self.params, doc_processor.config._Config__attributes)
        self.assertEqual(0, doc_processor._num_embedded_files)

        self.assertEqual(self.params["embeddings_model_dir"],
                         os.environ["HF_HOME"])
        self.assertEqual('1', os.environ["TRANSFORMERS_OFFLINE"])

        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    @utils.subtest("vector_store_type", ("faiss", "postgres"))
    def test_init_llama_index(self, vector_store_type):
        params = self.params.copy()
        params["vector_store_type"] = vector_store_type

        doc_processor = document_processor.DocumentProcessor(**params)
        self.log.warning.assert_not_called()
        self.indexdb.assert_called_once_with(doc_processor.config)

        self.assertIsNotNone(doc_processor)

        params.update(  # Add default values
            embedding_dimension=None,  # Not calculated because class is mocked
            manual_chunking=True,
            table_name=None,
            vector_store_type=vector_store_type,
        )
        self.assertEqual(params, doc_processor.config._Config__attributes)
        self.assertEqual(0, doc_processor._num_embedded_files)

        self.assertEqual(params["embeddings_model_dir"],
                         os.environ["HF_HOME"])
        self.assertEqual('1', os.environ["TRANSFORMERS_OFFLINE"])

        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        self.indexdb.reset_mock()

    @utils.subtest("vector_store_type",
                   ("llamastack-faiss", "llamastack-sqlite"))
    def test_init_llama_stack(self, vector_store_type):
        params = self.params.copy()
        params["vector_store_type"] = vector_store_type

        doc_processor = document_processor.DocumentProcessor(**params)
        self.log.warning.assert_not_called()
        self.llamadb.assert_called_once_with(doc_processor.config)

        self.assertIsNotNone(doc_processor)

        params.update(  # Add default values
            embedding_dimension=None,  # Not calculated because class is mocked
            manual_chunking=True,
            table_name=None,
        )
        self.assertEqual(params, doc_processor.config._Config__attributes)
        self.assertEqual(0, doc_processor._num_embedded_files)

        self.assertEqual(params["embeddings_model_dir"],
                        os.environ["HF_HOME"])
        self.assertEqual('1', os.environ["TRANSFORMERS_OFFLINE"])

        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    def test__check_config_faiss_auto_chunking(self):
        config = document_processor._Config(
            vector_store_type="faiss",
            manual_chunking=False,
            table_name=None,
        )
        document_processor.DocumentProcessor._check_config(config)
        self.log.warning.assert_called_once_with(mock.ANY)

    def test__check_config_faiss_table_name(self):
        config = document_processor._Config(
            vector_store_type="faiss",
            manual_chunking=True,
            table_name='table_name',
        )
        document_processor.DocumentProcessor._check_config(config)
        self.log.warning.assert_called_once_with(mock.ANY)

    def test_process(self):
        doc_processor = document_processor.DocumentProcessor(**self.params)

        metadata = mock.Mock()
        docs = list(range(5))

        with mock.patch.object(document_processor,
                               "SimpleDirectoryReader") as reader:
            reader.return_value.load_data.return_value = docs

            doc_processor.process(mock.sentinel.docs_dir,
                                  metadata,
                                  mock.sentinel.required_exts,
                                  mock.sentinel.file_extractor)

            reader.assert_called_once_with(
                str(mock.sentinel.docs_dir),
                recursive=True,
                file_metadata=metadata.populate,
                required_exts=mock.sentinel.required_exts,
                file_extractor=mock.sentinel.file_extractor,
            )

            doc_processor.db.add_docs.assert_called_once_with(docs)
            self.assertEqual(len(docs), doc_processor._num_embedded_files)

    def test_save(self):
        doc_processor = document_processor.DocumentProcessor(**self.params)

        doc_processor.save(mock.sentinel.index, mock.sentinel.output_dir)

        doc_processor.db.save.assert_called_once_with(
            mock.sentinel.index, mock.sentinel.output_dir, 0, mock.ANY)
