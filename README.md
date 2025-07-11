# RAG Content

RAG Content provides a shared codebase for generating vector databases.
It serves as the core framework for Lightspeed-related projects (e.g., OpenShift
Lightspeed, OpenStack Lightspeed, etc.) to generate their own vector databases
that can be used for RAG.

## Installing the Python Library

The ``lightspeed_rag_content`` library is not available via pip, but it's included:
   - in the base [container image](#via-container-image) or
   - it can be installed [via UV](#via-uv).

### Via uv

To install the library via uv, do:

1. Run the command ``uv sync``

    ```bash
    uv sync
    ```

2. Test if the library can be imported (expect `lightspeed-rag-content` in the output):

    ```bash
    uv run python -c "import lightspeed_rag_content; print(lightspeed_rag_content.__name__)"
    ```

### Via Container Image

The base container image can be manually generated or pulled from a container
registry at `ghcr.io/lightspeed-core/rag-content-cpu:latest`. To build the image locally,
follow these steps:

1. Install the requirements: `make` and `podman`.
2. Generate the container image:

    ```bash
    podman build -t localhost/lightspeed-rag-content-cpu:latest .
    ```

3. The `lightspeed_rag_content` and its dependencies will be installed in the
image (expect `lightspeed-rag-content` in the output):
    ```bash
    podman run localhost/lightspeed-rag-content-cpu:latest python -c "import lightspeed_rag_content; print(lightspeed_rag_content.__name__)"
    ```


## Generating the Vector Database

You can generate the vector database either using

1. [Llama-Index Faiss Vector Store](#faiss-vector-store)
2. [Llama-Index Postgres (PGVector) Vector Store](#postgres-pgvector-vector-store)
3. [Llama-Stack Faiss Vector-IO](#llama-stack-faiss)
4. [Llama-Stack SQLite-vec Vector-IO](#llama-stack-sqlite-vec)

Llama-Index approaches require you to download the embedding model, and we also
recommend it for Llama-Stack targets even though it should work even without
manually downloading the model, model-download.

All cases require you to prepare documentation in text format that is going to
be chunked and map to embeddings generated using the model:

1. Download the embedding model
([sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2))
from HuggingFace as follows:

    ```bash
   mkdir ./embeddings_model
   pdm run python ./scripts/download_embeddings_model.py -l ./embeddings_model/ -r sentence-transformers/all-mpnet-base-v2
    ```

2. Prepare dummy documentation:

   ```bash
   mkdir -p ./custom_docs/0.1
   echo "Vector Database is an efficient way how to provide information to LLM" > ./custom_docs/0.1/info.txt
   ```

3. Prepare a custom script (`./custom_processor.py`) for populating the vector
database. We provide an example of how such a script might look like using the
`lightspeed_rag_content` library. Note that in your case the script will be
different:

    ```python
    from lightspeed_rag_content.metadata_processor import MetadataProcessor
    from lightspeed_rag_content.document_processor import DocumentProcessor
    from lightspeed_rag_content import utils

    class CustomMetadataProcessor(MetadataProcessor):

        def __init__(self, url):
            self.url = url

        def url_function(self, file_path: str) -> str:
            # Return a URL for the file, so it can be referenced when used
            # in an answer
            return self.url

    if __name__ == "__main__":
        parser = utils.get_common_arg_parser()
        args = parser.parse_args()

        # Instantiate custom Metadata Processor
        metadata_processor = CustomMetadataProcessor("https://www.redhat.com")

        # Instantiate Document Processor
        document_processor = DocumentProcessor(
            chunk_size=args.chunk,
            chunk_overlap=args.overlap,
            model_name=args.model_name,
            embeddings_model_dir=args.model_dir,
            num_workers=args.workers,
            vector_store_type=args.vector_store_type,
        )

        # Load and embed the documents, this method can be called multiple times
        # for different sets of documents
        document_processor.process(args.folder, metadata=metadata_processor)

        # Save the new vector database to the output directory
        document_processor.save(args.index, args.output)
    ```

### Faiss Vector Store

Generate the documentation using the script from the previous section
([Generating the Vector Database](#generating-the-vector-database)):

```bash
uv run ./custom_processor.py -o ./vector_db/custom_docs/0.1 -f ./custom_docs/0.1/ -md embeddings_model/ -mn sentence-transformers/all-mpnet-base-v2 -i custom_docs-0_1
```

Once the command is done, you can find the vector database at `./vector_db`, the
embedding model at `./embeddings_model`, and the Index ID set to `custom-docs-0_1`.


### Postgres (PGVector) Vector Store

To generate a vector database stored in Postgres (PGVector), run the following
commands:

1. Start Postgres with the pgvector extension by running:

    ```bash
    make start-postgres-debug
    ```

    The `data` folder of Postgres is created at `./postgresql/data`. This command
    also creates `./output` for the output directory, in which the metadata is saved.

2. Run:

    ```bash
    POSTGRES_USER=postgres \
    POSTGRES_PASSWORD=somesecret \
    POSTGRES_HOST=localhost \
    POSTGRES_PORT=15432 \
    POSTGRES_DATABASE=postgres \
    uv run python ./custom_processor.py \
     -o ./output \
     -f custom_docs/0.1/ \
     -md embeddings_model/ \
     -mn sentence-transformers/all-mpnet-base-v2 \
     -i custom_docs-0_1 \
     --vector-store-type postgres
    ```

    Which generates embeddings on PostgreSQL, which can be used for RAG, and
    `metadata.json` in `./output`. Generated embeddings are stored in the
    `data_table_name` table.

    ```bash
    $ podman exec -it pgvector bash
    $ psql -U postgres
    psql (16.4 (Debian 16.4-1.pgdg120+2))
    Type "help" for help.

    postgres=# \dt
                     List of relations
     Schema |          Name          | Type  |  Owner
    --------+------------------------+-------+----------
     public | data_table_name        | table | postgres
    (1 row)
    ```

### Llama-Stack Faiss

The process is basically the same as in the
[Llama-Index Faiss Vector Store](#faiss-vector-store) but passing the
`--vector-store-type` parameter; so you generate the documentation using the
`custom_processor.py` script from earlier section
 ([Generating the Vector Database](#generating-the-vector-database)):

```bash
pdm run ./custom_processor.py \
  -o ./vector_db/custom_docs/0.1 \
  -f ./custom_docs/0.1/ \
  -md embeddings_model/ \
  -mn sentence-transformers/all-mpnet-base-v2 \
  -i custom_docs-0_1 \
  --vector-store-type=llamastack-faiss
```

Once the command is done, you can find the vector database at
`./vector_db/custom_docs/0.1` with the name `faiss_store.db` as well as a
barebones llama-stack configuration file named `llama-stack.yaml` for
reference, since it's not necessary for the final deployment.

The vector-io will be named `custom-docs-0_1`:

```yaml
providers:
 vector_io:
   - provider_id: custom-docs-0_1
     provider_type: inline::faiss
     config:
       kvstore:
         type: sqlite
         namespace: null
         db_path: /home/<user>/rag-content/vector_db/custom_docs/0.1/faiss_store.db
```

Once we have a database we can use script `query_rag.py` to check some results:

```bash
python scripts/query_rag.py \
  -p vector_db/custom_docs/0.1 \
  -x custom-docs-0_1 \
  -m embeddings_model \
  -k 5 \
  -q "how can I configure a cinder backend"
```

### Llama-Stack SQLite-vec

The process is the same as in the
[Llama-Stack Faiss](#llama-stack-faiss) but passing a different value on the
`--vector-store-type` parameter; so you generate the documentation using the
`custom_processor.py` script from earlier section
 ([Generating the Vector Database](#generating-the-vector-database)):

```bash
pdm run ./custom_processor.py \
  -o ./vector_db/custom_docs/0.1 \
  -f ./custom_docs/0.1/ \
  -md embeddings_model/ \
  -mn sentence-transformers/all-mpnet-base-v2 \
  -i custom_docs-0_1 \
  --vector-store-type=llamastack-sqlite-vec
```

Once the command is done, you can find the vector database at
`./vector_db/custom_docs/0.1` with the name `sqlitevec_store.db` as well as a
barebones llama-stack configuration file named `llama-stack.yaml` for
reference, since it's not necessary for the final deployment.

The vector-io will be named `custom-docs-0_1`:

```yaml
providers:
 vector_io:
   - provider_id: custom-docs-0_1
     provider_type: inline::sqlite-vec
     config:
       db_path: /home/<user>/rag-content/vector_db/custom_docs/0.1/sqlitevec_store.db
```

Once we have a database we can use script `query_rag.py` to check some results:

```bash
python scripts/query_rag.py \
  -p vector_db/custom_docs/0.1 \
  -x custom-docs-0_1 \
  -m embeddings_model \
  -k 5 \
  -q "how can I configure a cinder backend"
```

## Update lockfiles

The lock file is used in this repository:

```
uv.lock
```

The lock file needs to be regenerated when new updates (dependencies) are available. Use
following commands in order to do it:

```
uv lock --upgrade
uv sync
```

## `requirements*` Files Generation for Konflux

To generate all `requirements*` files:

```
requirements-build.in
requirements-build.txt
requirements.txt
```

The following command must be executed:

```bash
scripts/generate_packages_to_prefetch.py
```

# License

This project is licensed under the Apache License 2.0. See the **LICENSE** file
for details.
