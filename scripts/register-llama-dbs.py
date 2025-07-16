#!/usr/bin/env python3
"""Utility to register llama-stack dbs based on the vector-io contents.

If you copy a vector-io DB file to a new llama-stack installation and add it
to the llama-stack configuration file as the source of a vector-io provider
you will notice when you run llama-stack that it doesn't have any of the
vector-dbs that exist in that file.  You need to manually register the DBs.

This script is meant to help in that endeavor by automatically registering
all the vector-dbs present in your vector-io sources.

Use --config parameter to override the default configuration file (run.yaml)
where to look for the vector-io entries.
"""

import argparse
import importlib
import json
import logging
import os

# from pathlib import Path
import sqlite3
import tempfile
from typing import Any

import yaml

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def discover_faiss_dbs(db_path: str) -> list[dict[str, Any]]:
    """Discover vector-dbs in a FAISS database file.

    For FAISS, the database uses SQLite as kvstore, so we query it directly.
    Example of the database:
    ```
    sqlite> select key from kvstore;
    faiss_index:v3::os-docs
    vector_dbs:v3::os-docs

    sqlite> select key, value from kvstore where key='vector_dbs:v3::os-docs';
    vector_dbs:v3::os-docs|{"identifier":"os-docs","provider_resource_id":"os-docs","provider_id":"faiss","type":"vector_db","owner":null,"embedding_model":"sentence-transformers/all-mpnet-base-v2","embedding_dimension":768}
    ```

    The method returns a list of dictionaries with the provider information for
    each of the database:
    ```
    [
        {
            "identifier": "os-docs",
            "provider_resource_id": "os-docs",
            "provider_id": "faiss",
            "type": "vector_db",
            "owner": null,
            "embedding_model": "sentence-transformers/all-mpnet-base-v2",
            "embedding_dimension": 768
        },
    ]
    ```
    """
    vector_dbs = []

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("select key from kvstore where key like 'vector_dbs%'")
        db_keys = cursor.fetchall()

        for (db_key,) in db_keys:
            try:
                cursor.execute("SELECT value FROM kvstore WHERE key = ?", (db_key,))
                (data,) = cursor.fetchone()
                db_info = json.loads(data)
                vector_dbs.append(db_info)
            except sqlite3.Error:
                # Column might not exist in this table
                pass

        conn.close()

    except Exception as e:
        logger.warning(f"Could not inspect FAISS database {db_path}: {e}")

    return vector_dbs


def discover_sqlite_vec_dbs(db_path: str) -> list[dict[str, Any]]:
    """Discover vector-dbs in a SQLite-vec database file.

    For sqlite-vec, the database uses SQLite, so we query it directly.
    Example of the database:
    ```
    sqlite> select * from vector_dbs;
    os-docs|{"identifier":"os-docs","provider_resource_id":"os-docs","provider_id":"vectorio","type":"vector_db","owner":null,"embedding_model":"sentence-transformers/all-mpnet-base-v2","embedding_dimension":768}
    ```

    The method returns a list of dictionaries with the provider information for
    each of the database:
    ```
    [
        {
            "identifier": "os-docs",
            "provider_resource_id": "os-docs",
            "provider_id": "vectorio",
            "type": "vector_db",
            "owner": null,
            "embedding_model": "sentence-transformers/all-mpnet-base-v2",
            "embedding_dimension": 768
        },
    ]
    ```
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM vector_dbs")

        vector_dbs = [json.loads(metadata) for db_id, metadata in cursor.fetchall()]
        conn.close()

    except Exception as e:
        logger.warning(f"Could not inspect SQLite-vec database {db_path}: {e}")
        vector_dbs = []

    return vector_dbs


def discover_vector_dbs(provider_type: str, db_path: str) -> list[dict[str, Any]]:
    """Discover vector-dbs in a database file based on provider type."""
    if not os.path.exists(db_path):
        logger.warning(f"Database file does not exist: {db_path}")
        return []

    if provider_type == "inline::faiss":
        return discover_faiss_dbs(db_path)

    if provider_type == "inline::sqlite-vec":
        return discover_sqlite_vec_dbs(db_path)

    logger.warning(f"Unknown provider type: {provider_type}")
    return []


def load_config(config_path: str) -> Any:
    """Load and parse the llama-stack configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def _check_vector_dbs(config: dict[str, Any], vector_dbs: dict[str, Any]) -> None:
    """Check if DB metadata aligns with the llama-stack configuration ."""
    embedding_models = {
        model["model_id"]: model
        for model in config["models"]
        if model.get("model_type") == "embedding"
    }

    for provider_id, dbs_info in vector_dbs.items():
        # TODO: Should be change the database? Fail? Have a config option for this?

        for db_info in dbs_info:
            if db_info["provider_id"] != provider_id:
                logger.warning(
                    "Provider id in DB (%s) doesn't match llama-stack "
                    "configuration (%s)",
                    db_info["provider_id"],
                    provider_id,
                )

            model = db_info["embedding_model"]
            if model not in embedding_models:
                logger.warning(
                    "Embedding model in DB (%s) not present in llama-stack "
                    "configuration",
                    db_info["embedding_model"],
                )
                continue

            db_dimension = db_info.get("embedding_dimension")
            cfg_dimension = (
                embedding_models[model].get("metadata", {}).get("embedding_dimension")
            )

            if db_dimension and cfg_dimension and db_dimension != cfg_dimension:
                logger.warning(
                    "Embedding model dimension in DB (%s) doesn't match the "
                    "llama-stack configuration (%s)",
                    db_dimension,
                    cfg_dimension,
                )


def register_vector_dbs(
    llama_cfg_dir: str, config: dict[str, Any], vector_dbs_to_register: dict[str, Any]
) -> None:
    """Register vector-dbs with llama-stack."""
    if not vector_dbs_to_register:
        logger.info("No vector-dbs to register")
        return

    # We need to set llama-stack's config dir so registered DBs will be added
    if llama_cfg_dir:
        os.environ["LLAMA_STACK_CONFIG_DIR"] = llama_cfg_dir

    # Write config to temp file
    temp_config_file = tempfile.NamedTemporaryFile(suffix=".yaml")

    # Reduce the config to the minimum which is all the vector_io, the models
    # they use and the intefence providers for those models.
    provider_model = {
        model["provider_id"]: model
        for model in config["models"]
        if model["model_type"] == "embedding"
    }
    min_config = {
        "version": "2",
        "image_name": config.get("image_name", "ollama"),
        "apis": ["inference", "vector_io", "models"],
        "providers": {
            "vector_io": config["providers"]["vector_io"],
            "inference": [
                inf
                for inf in config["providers"]["inference"]
                if inf["provider_id"] in provider_model
            ],
        },
        "models": list(provider_model.values()),
    }

    yaml.safe_dump(min_config, temp_config_file.file, encoding="utf-8")
    temp_config_file.flush()

    # Import and initialize llama-stack
    try:
        llama_stack = importlib.import_module("llama_stack")
        client = llama_stack.distribution.library_client.LlamaStackAsLibraryClient(
            temp_config_file.name
        )
        client.initialize()

        # Register each vector-db that is not already registered
        existing_dbs = {db.identifier: db for db in client.vector_dbs.list()}
        for provider_id, dbs_info in vector_dbs_to_register.items():
            ids = [db["identifier"] for db in dbs_info]
            # llama-stack breaks our logger
            # logger.info(f"Registering vector-dbs for provider {provider_id}: {ids}")
            print(f"Registering vector-dbs for provider {provider_id}: {ids}")

            for db_info in dbs_info:
                db_id = db_info["identifier"]
                # TODO: Compare the config of the existing DBs
                if db_info["provider_resource_id"] in existing_dbs:
                    # logger.info(f"Skipping {db_id}, already present")
                    print(f"Skipping {db_id}, already present")
                    continue

                # TODO: Confirm the mapping is correct
                params = {
                    "vector_db_id": db_id,
                    "provider_id": db_info["provider_id"],
                    "provider_vector_db_id": db_info["provider_resource_id"],
                    "embedding_model": db_info["embedding_model"],
                }
                if "embedding_dimension" in db_info:
                    params["embedding_dimension"] = db_info["embedding_dimension"]

                try:
                    client.vector_dbs.register(**params)
                    # logger.info(f"Successfully registered vector-db: {db_id}")
                    print(f"Successfully registered vector-db: {db_id}")
                except Exception as e:
                    # logger.error(f"Failed to register vector-db {db_id}: {e}")
                    print(f"Failed to register vector-db {db_id}: {e}")

    except Exception as e:
        # logger.error(f"Failed to initialize llama-stack client: {e}")
        print(f"Failed to initialize llama-stack client: {e}")
        raise


def _parse_args() -> Any:
    """Create arg parser and parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Register in llama-stack all vector-dbs from configured vector-io "
            "providers"
        )
    )
    parser.add_argument(
        "--config",
        default="run.yaml",
        help="Path to llama-stack configuration file (default: run.yaml)",
    )
    parser.add_argument(
        "--llama-cfg-dir",
        default=os.environ.get(
            "LLAMA_STACK_CONFIG_DIR", os.path.expanduser("~/.llama")
        ),
        help="Path to llama-stack configuration directory where the databases will be registered",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be registered without actually registering",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()
    return args


def _get_vector_dbs(
    config_filename: str, config: dict[str, Any]
) -> dict[str, list[Any]]:
    """Get the metadata for the vector-dbs to register."""
    # Find vector-io providers
    vector_io_providers = []
    if "providers" in config and "vector_io" in config["providers"]:
        vector_io_providers = config["providers"]["vector_io"]

    vector_dbs_to_register: dict[str, list[Any]] = {}

    if not vector_io_providers:
        logger.warning("No vector-io providers found in configuration")
        return vector_dbs_to_register

    # Discover vector-dbs in each provider
    for provider in vector_io_providers:
        provider_id = provider.get("provider_id", "unknown")
        provider_type = provider.get("provider_type", "unknown")

        logger.info(f"Processing provider: {provider_id} (type: {provider_type})")

        # Get database path
        db_path = None
        if "config" in provider:
            if "db_path" in provider["config"]:
                db_path = provider["config"]["db_path"]
            elif (
                "kvstore" in provider["config"]
                and "db_path" in provider["config"]["kvstore"]
            ):
                db_path = provider["config"]["kvstore"]["db_path"]

        if not db_path:
            logger.warning(f"No database path found for provider {provider_id}")
            continue

        # Make path absolute if it's relative
        if not os.path.isabs(db_path):
            config_dir = os.path.dirname(os.path.abspath(config_filename))
            db_path = os.path.realpath(os.path.join(config_dir, db_path))

        logger.info(f"Discovering vector-dbs in: {db_path}")

        # Discover vector-dbs
        vector_dbs = discover_vector_dbs(provider_type, db_path)

        if vector_dbs:
            vector_dbs_to_register[provider_id] = vector_dbs
            logger.info(f"Found {len(vector_dbs)} vector-dbs: {vector_dbs}")
        else:
            logger.info("No vector-dbs found in this provider")

    return vector_dbs_to_register


def main() -> int:
    """Register vector-dbs from llama-stack configuration."""
    args = _parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config)

        vector_dbs_to_register = _get_vector_dbs(args.config, config)

        # Register vector-dbs
        if not vector_dbs_to_register:
            logger.info("No vector-dbs found to register")
            return 0

        total_dbs = sum(len(dbs) for dbs in vector_dbs_to_register.values())
        logger.info(f"Found {total_dbs} vector-dbs to register")
        _check_vector_dbs(config, vector_dbs_to_register)

        if args.dry_run:
            logger.info("DRY RUN - Would register the following vector-dbs:")
            for provider_id, dbs in vector_dbs_to_register.items():
                for info in dbs:
                    logger.info(
                        "  Provider id: %s - resource_id: %s",
                        provider_id,
                        info["provider_resource_id"],
                    )
        else:
            register_vector_dbs(args.llama_cfg_dir, config, vector_dbs_to_register)

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
