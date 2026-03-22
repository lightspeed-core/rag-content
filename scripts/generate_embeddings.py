"""Utility script for generating a vector database from Markdown files.

Provides a ready-made CLI so that users do not need to write a custom
Python script to drive the embedding pipeline.  The default settings
match the prebuilt CPU image defaults; override them with the flags below.

Usage::

    python generate_embeddings.py -f /input -o /output -i my-index

    # override chunk size and use a custom model directory
    python generate_embeddings.py -f /input -o /output -i my-index \\
        -c 512 -v 50 -d /my-embeddings

"""

import argparse

from lightspeed_rag_content.document_processor import DocumentProcessor
from lightspeed_rag_content.metadata_processor import DefaultMetadataProcessor

DEFAULT_CHUNK_SIZE = 380
DEFAULT_CHUNK_OVERLAP = 0
DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_MODEL_DIR = "/rag-content/embeddings_model"
DEFAULT_VECTOR_STORE = "llamastack-faiss"
DEFAULT_DOC_TYPE = "markdown"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a vector database from Markdown files.",
    )
    parser.add_argument(
        "-f", "--folder", required=True, help="Input directory with document files"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output directory for the vector DB"
    )
    parser.add_argument(
        "-i", "--index", required=True, help="Index name for the vector store"
    )
    parser.add_argument(
        "-c",
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Token chunk size (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "-v",
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Token chunk overlap (default: {DEFAULT_CHUNK_OVERLAP})",
    )
    parser.add_argument(
        "-m",
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"Embedding model name (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "-d",
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help=f"Local directory containing the embedding model (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "-s",
        "--vector-store",
        default=DEFAULT_VECTOR_STORE,
        help=f"Vector store type (default: {DEFAULT_VECTOR_STORE})",
    )
    parser.add_argument(
        "-t",
        "--doc-type",
        default=DEFAULT_DOC_TYPE,
        help=f"Document type (default: {DEFAULT_DOC_TYPE})",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    metadata_processor = DefaultMetadataProcessor()

    document_processor = DocumentProcessor(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        model_name=args.model_name,
        embeddings_model_dir=args.model_dir,
        vector_store_type=args.vector_store,
        doc_type=args.doc_type,
    )

    document_processor.process(args.folder, metadata=metadata_processor)
    document_processor.save(args.index, args.output)


if __name__ == "__main__":
    main()
