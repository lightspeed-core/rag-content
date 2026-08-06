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

"""Configuration templates for llama-stack and Lightspeed Core Stack output."""

# llama-stack configuration templates

LLAMA_STACK_TEMPLATE = """version: 2
image_name: starter

apis:
- agents
- files
- inference
- safety
- tool_runtime
- vector_io

server:
  port: 8321

providers:
  inference:
  - config: {{}}
    provider_id: sentence-transformers
    provider_type: inline::sentence-transformers
  files:
  - config:
      metadata_store:
        table_name: files_metadata
        backend: sql_default
      storage_dir: /tmp/files
    provider_id: meta-reference-files
    provider_type: inline::localfs
  agents:
  - config:
      persistence:
        agent_state:
          namespace: agents_state
          backend: kv_default
        responses:
          table_name: agents_responses
          backend: sql_default
    provider_id: meta-reference
    provider_type: inline::meta-reference
  tool_runtime:
  - config: {{}}
    provider_id: rag-runtime
    provider_type: inline::rag-runtime
  vector_io:
  - config:
      {vector_io_cfg}
    provider_id: {index_id}
    provider_type: {provider_type_prefix}::{provider_type}
storage:
  backends:
    kv_rag:
      type: kv_sqlite
      db_path: {kv_db_path}
    kv_default:
      type: kv_sqlite
      db_path: /tmp/kv_store.db
    sql_default:
      type: sql_sqlite
      db_path: /tmp/sql_store.db
  stores:
    metadata:
      namespace: registry
      backend: kv_default
    inference:
      table_name: inference_store
      backend: sql_default
    conversations:
      table_name: openai_conversations
      backend: sql_default
registered_resources:
  models:
  - metadata:
      embedding_dimension: {dimension}
    model_id: {model_name}
    provider_id: sentence-transformers
    provider_model_id: {model_name_or_dir}
    model_type: embedding
  vector_stores: []
  shields: []
  datasets: []
  scoring_fns: []
  benchmarks: []
  tool_groups:
  - toolgroup_id: builtin::rag
    provider_id: rag-runtime
"""

LLAMA_STACK_VECTOR_STORES_TEMPLATE = """vector_stores:
  - embedding_dimension: {dimension}
    embedding_model: sentence-transformers/{model_name_or_dir}
    provider_id: {vector_io_provider_id}
    vector_store_id: {vector_store_id}"""

LLAMA_STACK_VECTOR_IO_CONFIG_SQLITE = """persistence:
        namespace: vector_io::{provider_type}
        backend: kv_rag"""

LLAMA_STACK_VECTOR_IO_CONFIG_PGVECTOR = """persistence:
        namespace: vector_io::{provider_type}
        backend: kv_default
      host: ${{env.POSTGRES_HOST}}
      port: ${{env.POSTGRES_PORT}}
      db: ${{env.POSTGRES_DATABASE}}
      user: ${{env.POSTGRES_USER}}
      password: ${{env.POSTGRES_PASSWORD}}"""

LLAMA_STACK_CFG_FILENAME = "llama-stack.yaml"

# Lightspeed Core Stack configuration templates

LCS_CFG_FILENAME = "lightspeed-stack.yaml"

LCS_BASE_TEMPLATE = """\
name: Lightspeed Core Stack (LCS)
service:
  host: 0.0.0.0
  port: 8080
  base_url: http://localhost:8080
  auth_enabled: false
  workers: 1
  color_log: true
  access_log: true
llama_stack:
  use_as_library_client: true
  library_client_config_path: {llama_stack_config_path}
  # api_key: custom-key  # Uncomment if your llama-stack requires authentication
  # To use a remote llama-stack service instead of library mode, set:
  # use_as_library_client: false
  # url: http://localhost:8321
user_data_collection:
  feedback_enabled: true
  feedback_storage: "/tmp/data/feedback"
  transcripts_enabled: true
  transcripts_storage: "/tmp/data/transcripts"
conversation_cache:
  type: "sqlite"
  sqlite:
    db_path: "/tmp/data/conversation-cache.db"
authentication:
  module: "noop"

"""

LCS_FAISS_BYOK_TEMPLATE = """\
rag:
  byok:
    stores:
      - rag_id: {index_id}
        backend: faiss
        embedding_model: {model_name}
        embedding_dimension: {dimension}
        vector_db_id: {vector_store_id}
        db_path: ${{env.RAG_DB_PATH:={db_path}}}
  retrieval:
    # inline:
    #   sources:
    #     - {index_id}
    tool:
      sources:
        - {index_id}
"""

LCS_PGVECTOR_BYOK_TEMPLATE = """\
rag:
  byok:
    stores:
      - rag_id: {index_id}
        backend: pgvector
        embedding_model: {model_name}
        embedding_dimension: {dimension}
        vector_db_id: {vector_store_id}
        host: ${{env.POSTGRES_HOST}}
        port: ${{env.POSTGRES_PORT}}
        db: ${{env.POSTGRES_DATABASE}}
        user: ${{env.POSTGRES_USER}}
        password: ${{env.POSTGRES_PASSWORD}}
  retrieval:
    # inline:
    #   sources:
    #     - {index_id}
    tool:
      sources:
        - {index_id}
"""
