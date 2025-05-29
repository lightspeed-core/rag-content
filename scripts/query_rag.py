"""Utility script for querying RAG database."""

import argparse
import os

from llama_index.core import Settings, load_index_from_storage
from llama_index.core.llms.utils import resolve_llm
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility script for querying RAG database"
    )
    parser.add_argument(
        "-p",
        "--db-path",
        required=True,
        help="path to the vector db",
    )
    parser.add_argument("-x", "--product-index", required=True, help="product index")
    parser.add_argument(
        "-m", "--model-path", required=True, help="path to the embedding model"
    )
    parser.add_argument("-q", "--query", type=str, required=True, help="query to run")
    parser.add_argument("-k", "--top-k", type=int, default=1, help="similarity_top_k")
    parser.add_argument("-n", "--node", help="retrieve node")
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.0,
        help="Minimal score for top node retrieved",
    )
    args = parser.parse_args()

    os.environ["TRANSFORMERS_CACHE"] = args.model_path
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    Settings.llm = resolve_llm(None)
    Settings.embed_model = HuggingFaceEmbedding(model_name=args.model_path)

    storage_context = StorageContext.from_defaults(
        vector_store=FaissVectorStore.from_persist_dir(args.db_path),
        persist_dir=args.db_path,
    )
    vector_index = load_index_from_storage(
        storage_context=storage_context,
        index_id=args.product_index,
    )
    if args.node is not None:
        print(storage_context.docstore.get_node(args.node))
    else:
        retriever = vector_index.as_retriever(similarity_top_k=args.top_k)
        nodes = retriever.retrieve(args.query)
        if len(nodes) == 0:
            print(f"No nodes retrieved for query: {args.query}")
            exit(1)
        if args.threshold > 0.0 and nodes[0].score < args.threshold:
            print(
                f"Score {nodes[0].score} of the top retrieved node for query '{args.query}' "
                f"didn't cross the minimal threshold {args.threshold}."
            )
            exit(1)
        for n in nodes:
            print(n)
