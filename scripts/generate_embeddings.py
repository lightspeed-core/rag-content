r"""Utility script for generating a vector database from Markdown files.

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
import os

from lightspeed_rag_content.document_processor import DocumentProcessor
from lightspeed_rag_content.image_builder import DEFAULT_BASE_IMAGE, build_image_archive
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
    parser.add_argument("-f", "--folder", required=True, help="Input directory with document files")
    parser.add_argument("-o", "--output", required=True, help="Output directory for the vector DB")
    parser.add_argument("-i", "--index", required=True, help="Index name for the vector store")
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
    parser.add_argument(
        "--output-image",
        default=None,
        help="Path for the output image archive (.tar). When set, the vector DB "
        "(and optionally the embedding model) is packaged into a container image "
        "that can be loaded with 'podman load' or 'docker load'.",
    )
    parser.add_argument(
        "--image-name",
        default="rag-content-output",
        help="Repository name embedded in the image archive (default: rag-content-output)",
    )
    parser.add_argument(
        "--image-tag",
        default="latest",
        help="Tag embedded in the image archive (default: latest)",
    )
    parser.add_argument(
        "--exclude-model",
        dest="include_model",
        action="store_false",
        default=True,
        help="Exclude the embedding model from the output image (included by default)",
    )
    parser.add_argument(
        "--base-image",
        default=None,
        help="Base container image for the output image archive " "(default: UBI 9 full, ubi9/ubi)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the embedding pipeline using CLI arguments."""
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

    if args.output_image:
        extra_dirs = None
        if args.include_model:
            if not os.path.isdir(args.model_dir):
                raise FileNotFoundError(
                    f"Embedding model directory not found: {args.model_dir}. "
                    "Use --exclude-model to build the image without it."
                )
            extra_dirs = {args.model_dir: "/rag/embeddings_model"}
        build_image_archive(
            vector_db_dir=args.output,
            output_tar_path=args.output_image,
            image_name=args.image_name,
            image_tag=args.image_tag,
            extra_dirs=extra_dirs,
            base_image=args.base_image or DEFAULT_BASE_IMAGE,
        )


if __name__ == "__main__":
    main()
