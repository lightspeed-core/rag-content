"""Utility script for querying RAG database."""

import argparse
import importlib
import json
import logging
import os
import sys
import tempfile
from typing import Any, cast

import yaml
from llama_index.core import Settings, load_index_from_storage
from llama_index.core.llms.utils import resolve_llm
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore


def _llama_index_query(args: argparse.Namespace) -> None:  # noqa: C901
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
        node = storage_context.docstore.get_node(args.node)
        if isinstance(node, TextNode):
            result = {
                "query": args.query,
                "type": "single_node",
                "node_id": args.node,
                "node": {
                    "id": node.node_id,
                    "text": node.text,
                    "metadata": node.metadata if hasattr(node, "metadata") else {},
                },
            }
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print(node)
        else:
            logging.warning(
                f"Node {args.node} is not a TextNode, type: {type(node).__name__}"
            )
            if args.json:
                result = {
                    "query": args.query,
                    "type": "single_node",
                    "node_id": args.node,
                    "error": f"Node is not a TextNode (type: {type(node).__name__})",
                }
                print(json.dumps(result, indent=2))
            exit(1)
    else:
        retriever = vector_index.as_retriever(similarity_top_k=args.top_k)
        nodes = retriever.retrieve(args.query)

        if len(nodes) == 0:
            logging.warning(f"No nodes retrieved for query: {args.query}")
            if args.json:
                result = {
                    "query": args.query,
                    "top_k": args.top_k,
                    "threshold": args.threshold,
                    "nodes": [],
                }
                print(json.dumps(result, indent=2))
            exit(1)

        if args.threshold > 0.0 and nodes[0].score < args.threshold:
            logging.warning(
                f"Score {nodes[0].score} of the top retrieved node for query '{args.query}' "
                f"didn't cross the minimal threshold {args.threshold}."
            )
            if args.json:
                result = {
                    "query": args.query,
                    "top_k": args.top_k,
                    "threshold": args.threshold,
                    "nodes": [],
                }
                print(json.dumps(result, indent=2))
            exit(1)

        # Format results
        result = {
            "query": args.query,
            "top_k": args.top_k,
            "threshold": args.threshold,
            "nodes": [],
        }
        for node in nodes:  # type: ignore
            if isinstance(node, NodeWithScore):
                base_node = cast("NodeWithScore", node)
                text = getattr(base_node, "text", None)
                if text is None:
                    text = base_node.node.get_content() or ""
                node_data = {
                    "id": node.node_id,
                    "score": node.score,
                    "text": text,
                    "metadata": node.metadata,
                }
                result["nodes"].append(node_data)
            else:
                logging.debug(
                    f"Skipping node of type {type(node).__name__}, expected NodeWithScore"
                )

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            for n in nodes:
                print("=" * 80)
                print(n)


def _get_db_path_dict(vector_type: str, config: dict[str, Any]) -> dict[str, Any]:
    """Return the dict where db_path key is from our llama-stack config."""
    try:
        provider_config: dict[str, Any] = config["providers"]["vector_io"][0]["config"]

        # New llama-stack 0.3.x format
        if "persistence" in provider_config:
            backend_name = provider_config["persistence"]["backend"]
            result: dict[str, Any] = config["storage"]["backends"][backend_name]
            return result

        # Old 0.2.x format
        if vector_type == "llamastack-faiss":
            kvstore: dict[str, Any] = provider_config["kvstore"]
            return kvstore
        return provider_config
    except (KeyError, IndexError) as e:
        raise ValueError(f"Invalid configuration structure: {e}")


def _get_chunk_text(chunk: Any) -> str:
    """Extract text content from a chunk object."""
    if hasattr(chunk, "content"):
        if isinstance(chunk.content, str):
            return chunk.content
        if isinstance(chunk.content, list):
            return " ".join(
                c.text if hasattr(c, "text") else str(c) for c in chunk.content
            )
    return str(chunk)


def _llama_stack_query(args: argparse.Namespace) -> None:  # noqa: C901
    tmp_dir = tempfile.TemporaryDirectory(prefix="ls-rag-")
    os.environ["LLAMA_STACK_CONFIG_DIR"] = tmp_dir.name

    cfg = yaml.safe_load(
        open(os.path.join(args.db_path, "llama-stack.yaml"), "r", encoding="utf-8")
    )

    db_dict = _get_db_path_dict(args.vector_store_type, cfg)
    db_filename = os.path.basename(db_dict["db_path"])
    db_dict["db_path"] = os.path.realpath(os.path.join(args.db_path, db_filename))

    if args.model_path:
        model_path = os.path.realpath(args.model_path)
        # New 0.3.x format
        if "registered_resources" in cfg and "models" in cfg["registered_resources"]:
            cfg["registered_resources"]["models"][0]["provider_model_id"] = model_path
            if "vector_stores" in cfg["registered_resources"]:
                for vs in cfg["registered_resources"]["vector_stores"]:
                    vs["embedding_model"] = f"sentence-transformers/{model_path}"
        # Old 0.2.x format
        elif "models" in cfg:
            cfg["models"][0]["provider_model_id"] = model_path

    cfg_file = os.path.join(tmp_dir.name, "llama-stack.yaml")
    yaml.safe_dump(cfg, open(cfg_file, "w", encoding="utf-8"))

    lib_client = importlib.import_module("llama_stack.core.library_client")
    with lib_client.LlamaStackAsLibraryClient(cfg_file) as client:
        query_cfg = {
            "max_chunks": args.top_k,
            "mode": "vector",  # "vector", "keyword", or "hybrid". Default "vector"
            "score_threshold": 0,
        }
        res = client.vector_io.query(
            vector_store_id=args.product_index, query=args.query, params=query_cfg
        )

        if len(res.chunks) == 0:
            logging.warning(f"No chunks retrieved for query: {args.query}")
            if args.json:
                result = {
                    "query": args.query,
                    "top_k": args.top_k,
                    "threshold": args.threshold,
                    "nodes": [],
                }
                print(json.dumps(result, indent=2))
            exit(1)

        threshold = args.threshold
        if threshold > 0.0 and res.scores and res.scores[0] < threshold:
            logging.warning(
                f"Score {res.scores[0]} of the top retrieved node for query '{args.query}' "
                f"didn't cross the minimal threshold {threshold}."
            )
            if args.json:
                result = {
                    "query": args.query,
                    "top_k": args.top_k,
                    "threshold": args.threshold,
                    "nodes": [],
                }
                print(json.dumps(result, indent=2))
            exit(1)

        # Format results
        result = {
            "query": args.query,
            "top_k": args.top_k,
            "threshold": args.threshold,
            "nodes": [],
        }

        for chunk, score in zip(res.chunks, res.scores):
            node_data = {
                "id": chunk.chunk_id if hasattr(chunk, "chunk_id") else "",
                "score": score,
                "text": _get_chunk_text(chunk),
                "metadata": chunk.metadata if hasattr(chunk, "metadata") else {},
            }
            result["nodes"].append(node_data)

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            for chunk, score in zip(res.chunks, res.scores):
                print("=" * 80)
                chunk_id = chunk.chunk_id if hasattr(chunk, "chunk_id") else ""
                print(
                    f"Chunk ID: {chunk_id}\nScore: {score}\nText:\n{_get_chunk_text(chunk)}"
                )


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
    parser.add_argument(
        "--vector-store-type",
        default="auto",
        choices=["auto", "faiss", "llamastack-faiss", "llamastack-sqlite-vec"],
        help="vector store type to be used.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format",
    )

    args = parser.parse_args()

    if args.json:
        # In JSON mode, only show ERROR or higher to avoid polluting JSON output
        logging.basicConfig(
            level=logging.ERROR,
            format="%(levelname)s: %(message)s",
            stream=sys.stderr,  # Send logs to stderr to keep stdout clean for JSON
        )
    else:
        # In normal mode, show info and above
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.json:
        logging.info("Command line used: " + " ".join(sys.argv))

    vector_store_type = args.vector_store_type
    if args.vector_store_type == "auto":
        if os.path.exists(os.path.join(args.db_path, "metadata.json")):
            args.vector_store_type = "faiss"
        elif os.path.exists(os.path.join(args.db_path, "sqlite-vec_store.db")):
            args.vector_store_type = "llamastack-sqlite-vec"
        elif os.path.exists(os.path.join(args.db_path, "faiss_store.db")):
            args.vector_store_type = "llamastack-faiss"
        else:
            logging.error(f"Cannot recognize the DB in {args.db_path}")
            exit(1)

    if args.vector_store_type == "faiss":
        _llama_index_query(args)
    else:
        _llama_stack_query(args)
