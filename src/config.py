import os
from dataclasses import dataclass, field
from typing import Dict, Optional, List


@dataclass
class Settings:
    # --- Directory ---
    root_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir: str = root_dir + "/dickens/Musique"

    # --- Chunking ---
    chunk_token_size: int = 2000
    chunk_overlap_token_size: int = 50

    # --- Model ---
    tiktoken_model_name: str = "gpt-4o-mini"
    llm_model_name: str = "gpt-4o-mini-2024-07-18"
    embedding_model_name: str = "BAAI/bge-m3"
    llm_model_max_token_size: int = 30000
    embedding_dim: int = 1024
    embedding_max_token_size: int = 8192

    # --- LLM Client ---
    llm_base_url: str = os.environ.get("LLM_BASE_URL", "")
    llm_api_key: str = os.environ.get("LLM_API_KEY", "")
    llm_timeout: float = 460.0

    # --- LLM Parameters ---
    llm_max_tokens: Optional[int] = 8192
    llm_temperature: Optional[float] = 0.7
    llm_top_p: Optional[float] = None
    llm_presence_penalty: Optional[float] = None
    llm_extra_body: Optional[Dict] = None

    # --- Embedding Client ---
    embedding_base_url: Optional[str] = os.environ.get("EMBEDDING_BASE_URL", "")
    embedding_api_key: Optional[str] = os.environ.get("EMBEDDING_BASE_KEY", "")
    embedding_timeout: float = 120.0

    # --- Rerank Model ---
    rerank_base_url: str = os.environ.get("RERANK_BASE_URL", "")
    rerank_model_name: str = "Pro/BAAI/bge-reranker-v2-m3"
    rerank_api_key: str = os.environ.get("RERANK_API_KEY", "")
    rerank_timeout: float = 60.0
    rerank_max_retries: int = 20
    rerank_max_chunks_per_doc: int = 8196
    rerank_overlap_tokens: int = 50

    # --- HTTP Client ---
    httpx_timeout: float = 120.0
    httpx_max_connections: int = 600
    httpx_max_keepalive_connections: int = 300

    # --- Extraction & Summary ---
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500

    # --- Concurrency & Storage ---
    async_semaphore_limit: int = 200
    embedding_async_semaphore_limit: int = 200
    embedding_batch_num: int = 32

    # --- Graph ---
    node2vec_params: Dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    # --- Beam Search ---
    beam_search_max_hops: int = 3
    beam_search_beam_width: int = 15
    beam_search_alpha: float = 0.5
    beam_search_beta: float = 1.5
    beam_search_decay: float = 0.9

    # --- KG Query ---
    kg_max_token_size: int = 4000
    kg_max_token_size_for_beam_search: int = 4000

    # Cross-modal fusion weights
    ent_to_chunk_wt: float = 0.5
    rel_to_chunk_wt: float = 0.5
    chunk_to_ent_wt: float = 0.5
    chunk_to_rel_wt: float = 0.5

    # Top-k for different retrieval types
    kg_etop_k: int = 60
    kg_rtop_k: int = 60
    kg_top_k: int = 60

    def __post_init__(self):
        os.makedirs(self.data_dir, exist_ok=True)

        if self.embedding_base_url is None:
            self.embedding_base_url = self.llm_base_url
        if self.embedding_api_key is None:
            self.embedding_api_key = self.llm_api_key

    @property
    def effective_embedding_base_url(self) -> str:
        return self.embedding_base_url or self.llm_base_url

    @property
    def effective_embedding_api_key(self) -> str:
        return self.embedding_api_key or self.llm_api_key


settings = Settings()
