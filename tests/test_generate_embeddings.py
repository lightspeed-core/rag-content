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

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Load generate_embeddings as a module from the scripts/ directory (it is not
# installed as a package, so we load it directly from its file path).
_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "generate_embeddings.py"
_spec = importlib.util.spec_from_file_location("generate_embeddings", _SCRIPT_PATH)
_generate_embeddings = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_generate_embeddings)


class TestGenerateEmbeddingsCLI:
    """Test cases for the generate_embeddings.py CLI argument parsing."""

    def _parse(self, args: list[str]):
        """Helper: parse a list of CLI args via the script's _parse_args()."""
        with patch("sys.argv", ["generate_embeddings.py"] + args):
            return _generate_embeddings._parse_args()

    def test_required_args_parsed(self):
        """Test that -f, -o and -i are correctly parsed."""
        args = self._parse(["-f", "/input", "-o", "/output", "-i", "my-index"])
        assert args.folder == "/input"
        assert args.output == "/output"
        assert args.index == "my-index"

    def test_default_values(self):
        """Test that defaults match the prebuilt CPU image constants."""
        args = self._parse(["-f", "/input", "-o", "/output", "-i", "my-index"])
        assert args.chunk_size == _generate_embeddings.DEFAULT_CHUNK_SIZE
        assert args.chunk_overlap == _generate_embeddings.DEFAULT_CHUNK_OVERLAP
        assert args.model_name == _generate_embeddings.DEFAULT_MODEL_NAME
        assert args.model_dir == _generate_embeddings.DEFAULT_MODEL_DIR
        assert args.vector_store == _generate_embeddings.DEFAULT_VECTOR_STORE
        assert args.doc_type == _generate_embeddings.DEFAULT_DOC_TYPE

    def test_override_all_optional_args(self):
        """Test that all optional flags override their defaults."""
        args = self._parse([
            "-f", "/input", "-o", "/output", "-i", "my-index",
            "-c", "512",
            "-v", "50",
            "-m", "custom/model",
            "-d", "/custom/embeddings",
            "-s", "faiss",
            "-t", "asciidoc",
        ])
        assert args.chunk_size == 512
        assert args.chunk_overlap == 50
        assert args.model_name == "custom/model"
        assert args.model_dir == "/custom/embeddings"
        assert args.vector_store == "faiss"
        assert args.doc_type == "asciidoc"

    @pytest.mark.parametrize("missing_flag,remaining", [
        ("-f", ["-o", "/output", "-i", "my-index"]),
        ("-o", ["-f", "/input", "-i", "my-index"]),
        ("-i", ["-f", "/input", "-o", "/output"]),
    ])
    def test_missing_required_arg_exits(self, missing_flag, remaining):
        """Test that omitting any required flag causes SystemExit."""
        with pytest.raises(SystemExit):
            self._parse(remaining)


class TestGenerateEmbeddingsMain:
    """Test cases for the generate_embeddings.py main() orchestration."""

    def test_main_calls_process_and_save(self, mocker):
        """Test main() wires DefaultMetadataProcessor and DocumentProcessor correctly."""
        mock_doc_processor = MagicMock()
        mock_dp_class = mocker.patch.object(
            _generate_embeddings,
            "DocumentProcessor",
            return_value=mock_doc_processor,
        )
        mock_metadata = MagicMock()
        mocker.patch.object(
            _generate_embeddings,
            "DefaultMetadataProcessor",
            return_value=mock_metadata,
        )

        with patch(
            "sys.argv",
            ["generate_embeddings.py", "-f", "/in", "-o", "/out", "-i", "idx"],
        ):
            _generate_embeddings.main()

        mock_dp_class.assert_called_once_with(
            chunk_size=_generate_embeddings.DEFAULT_CHUNK_SIZE,
            chunk_overlap=_generate_embeddings.DEFAULT_CHUNK_OVERLAP,
            model_name=_generate_embeddings.DEFAULT_MODEL_NAME,
            embeddings_model_dir=_generate_embeddings.DEFAULT_MODEL_DIR,
            vector_store_type=_generate_embeddings.DEFAULT_VECTOR_STORE,
            doc_type=_generate_embeddings.DEFAULT_DOC_TYPE,
        )
        mock_doc_processor.process.assert_called_once_with("/in", metadata=mock_metadata)
        mock_doc_processor.save.assert_called_once_with("idx", "/out")
