from dataclasses import dataclass, field
from typing import (
    TypedDict,
    Union,
    Literal,
    Generic,
    TypeVar,
    Optional,
    Dict,
    Any,
    List,
    Tuple,
    Set,
)

import numpy as np

from src.utils import EmbeddingFunc

TextChunkSchema = TypedDict(
    "TextChunkSchema",
    {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int},
)

T = TypeVar("T")


@dataclass
class QueryParam:
    mode: Literal[
        "naive",
        "dec_kg",
        "kg_query",
        "beam_query",
        "corprorag",
        "SGN",
    ] = "global"
    only_need_context: bool = False
    response_type: str = "Multi-paragraph"

    # "Multiple Paragraphs"
    top_k: int = 60
    max_token_for_text_unit: int = 4000
    max_token_for_global_context: int = 4000
    max_token_for_local_context: int = 4000


@dataclass
class StorageNameSpace:
    namespace: str
    global_config: Dict[str, Any]

    async def index_done_callback(self) -> None:
        """commit the storage operations after indexing"""
        pass

    async def query_done_callback(self) -> None:
        """commit the storage operations after querying"""
        pass


@dataclass
class BaseVectorStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc
    meta_fields: Set[str] = field(default_factory=set)

    async def query(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        raise NotImplementedError

    async def upsert(self, data: Dict[str, Dict[str, Any]]) -> List[Any]:
        """Use 'content' field from value for embedding, use key as id.
        If embedding_func is None, use 'embedding' field from value
        """
        raise NotImplementedError

    # Extended interface for entity/relationship lookup
    async def get_entity_id(self, name: str) -> Optional[str]:
        """Get entity ID by name. Return None if not found."""
        return None

    async def get_rel_id(self, src: str, tgt: str) -> Optional[str]:
        """Get relationship ID by source and target. Return None if not found."""
        return None

    async def get_entity_ids(self, names: List[str]) -> Dict[str, Optional[str]]:
        """Get multiple entity IDs by names."""
        return {name: await self.get_entity_id(name) for name in names}


@dataclass
class BaseKVStorage(Generic[T], StorageNameSpace):
    async def all_keys(self) -> List[str]:
        raise NotImplementedError

    async def get_by_id(self, id: str) -> Optional[T]:
        raise NotImplementedError

    async def get_by_ids(
        self, ids: List[str], fields: Optional[Set[str]] = None
    ) -> List[Optional[T]]:
        raise NotImplementedError

    async def filter_keys(self, data: List[str]) -> Set[str]:
        """return un-exist keys"""
        raise NotImplementedError

    async def upsert(self, data: Dict[str, T]) -> Dict[str, T]:
        raise NotImplementedError

    async def drop(self) -> None:
        raise NotImplementedError


@dataclass
class BaseGraphStorage(StorageNameSpace):
    async def has_node(self, node_id: str) -> bool:
        raise NotImplementedError

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        raise NotImplementedError

    async def node_degree(self, node_id: str) -> int:
        raise NotImplementedError

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        raise NotImplementedError

    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    async def get_node_edges(
        self, source_node_id: str
    ) -> Optional[List[Tuple[str, str]]]:
        raise NotImplementedError

    async def get_nodes(self, node_ids: List[str]) -> List[Optional[Dict[str, Any]]]:
        raise NotImplementedError

    async def get_edges(
        self, edge_pairs: List[Tuple[str, str]]
    ) -> List[Optional[Dict[str, Any]]]:
        raise NotImplementedError

    async def get_neighbors(self, node_id: str) -> List[str]:
        raise NotImplementedError

    async def get_nodes_edges(
        self, node_ids: List[str]
    ) -> Dict[str, Optional[List[Tuple[str, str]]]]:
        raise NotImplementedError

    async def get_nodes_by_ids(
        self, node_ids: List[str], include_neighbors: bool = False
    ) -> Dict[str, Any]:
        raise NotImplementedError

    async def upsert_node(self, node_id: str, node_data: Dict[str, Any]) -> None:
        raise NotImplementedError

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: Dict[str, Any]
    ) -> None:
        raise NotImplementedError

    async def clustering(self, algorithm: str) -> Any:
        raise NotImplementedError

    async def embed_nodes(self, algorithm: str) -> Tuple[np.ndarray, List[str]]:
        raise NotImplementedError("Node embedding is not used in lightrag.")

    # Extended interface for chunk-based queries
    async def find_by_chunk(
        self, chunk_id: str
    ) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[Tuple[str, str, Dict[str, Any]]]]:
        """Find nodes and edges by chunk ID.
        Returns: (nodes, edges) where nodes=[(node_id, node_data), ...] and edges=[(src, tgt, edge_data), ...]
        """
        raise NotImplementedError("find_by_chunk not implemented in base class")
