import asyncio
import html
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Union, cast, Tuple, List, Dict, Set, Optional
import networkx as nx
import numpy as np
from nano_vectordb import NanoVectorDB

from src.utils import load_json, logger, write_json
from src.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
)


@dataclass
class JsonKVStorage(BaseKVStorage):
    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        try:
            self._data = load_json(self._file_name) or {}
            logger.info(f"Load KV {self.namespace} with {len(self._data)} data")
        except Exception as e:
            logger.warning(f"Failed to load KV file {self._file_name}: {e}")
            self._data = {}

    async def all_keys(self) -> List[str]:
        return list(self._data.keys())

    async def index_done_callback(self) -> None:
        try:
            write_json(self._data, self._file_name)
        except Exception as e:
            logger.error(f"Failed to save KV file {self._file_name}: {e}")
            raise

    async def get_by_id(self, id):
        return self._data.get(id, None)

    async def get_by_ids(self, ids, fields=None):
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        return [
            (
                {k: v for k, v in self._data[id].items() if k in fields}
                if self._data.get(id, None)
                else None
            )
            for id in ids
        ]

    async def filter_keys(self, data: List[str]) -> Set[str]:
        return set([s for s in data if s not in self._data])

    async def upsert(self, data: Dict[str, Dict[str, Any]]):
        left_data = {k: v for k, v in data.items() if k not in self._data}
        self._data.update(left_data)
        return left_data

    async def drop(self) -> None:
        self._data = {}


@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.2
    _idx_kv: Optional[JsonKVStorage] = None

    def __post_init__(self):
        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        try:
            self._client = NanoVectorDB(
                self.embedding_func.embedding_dim, storage_file=self._client_file_name
            )
        except Exception as e:
            logger.error(f"Failed to initialize NanoVectorDB: {e}")
            raise

        self.cosine_better_than_threshold = self.global_config.get(
            "cosine_better_than_threshold", self.cosine_better_than_threshold
        )

        # Initialize reverse index for entities and relationships
        if self.namespace in ["entities", "relationships"]:
            self._init_reverse_index()

    def _init_reverse_index(self):
        """Initialize reverse index for entity/relationship lookups"""
        try:
            # 1) Create rev-idx KV
            self._idx_kv = JsonKVStorage(
                namespace=f"{self.namespace}_rev_idx",
                global_config=self.global_config,
            )

            # 2) Load from KV to memory
            raw_entity = self._idx_kv._data.get("entity_name2id", {})
            self._entity_name2id = raw_entity.copy()
            raw_rel = self._idx_kv._data.get("rel_pair2id", {})
            self._rel_pair2id = {}
            for key, rid in raw_rel.items():
                # key format "src||tgt"
                try:
                    src, tgt = key.split("||", 1)
                    self._rel_pair2id[(src, tgt)] = rid
                except ValueError:
                    # Backward compatibility: key may be tuple
                    if isinstance(key, (list, tuple)) and len(key) == 2:
                        self._rel_pair2id[(key[0], key[1])] = rid

            # 3) If both are empty, build index from file and write back
            if not self._entity_name2id and not self._rel_pair2id:
                self._load_reverse_index()
                # Async write back
                asyncio.get_event_loop().create_task(self._save_rev_idx())
        except Exception as e:
            logger.error(f"Failed to initialize reverse index: {e}")
            self._entity_name2id = {}
            self._rel_pair2id = {}

    def _load_reverse_index(self):
        """Build reverse index from vector.json file"""
        if not os.path.exists(self._client_file_name):
            return
        try:
            with open(self._client_file_name, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for record in raw.get("data", []):
                rid = record["__id__"]
                if "entity_name" in record:
                    name = record["entity_name"]
                    self._entity_name2id[name] = rid
                if "src_id" in record and "tgt_id" in record:
                    src = record["src_id"]
                    tgt = record["tgt_id"]
                    self._rel_pair2id[(src, tgt)] = rid
        except Exception as e:
            logger.error(
                f"Failed to load reverse index from {self._client_file_name}: {e}"
            )

    async def _save_rev_idx(self):
        """Serialize in-memory reverse index and save to KV"""
        if self._idx_kv is None:
            return
        try:
            # entity_name2id is str->str
            self._idx_kv._data["entity_name2id"] = self._entity_name2id
            # rel_pair2id: tuple key to "src||tgt"
            serial_rel = {
                f"{src}||{tgt}": rid for (src, tgt), rid in self._rel_pair2id.items()
            }
            self._idx_kv._data["rel_pair2id"] = serial_rel
            await self._idx_kv.index_done_callback()
        except Exception as e:
            logger.error(f"Failed to save reverse index: {e}")

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []

        try:
            list_data = [
                {
                    "__id__": k,
                    **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
                }
                for k, v in data.items()
            ]
            contents = [v["content"] for v in data.values()]
            batches = [
                contents[i : i + self._max_batch_size]
                for i in range(0, len(contents), self._max_batch_size)
            ]
            embeddings_list = await asyncio.gather(
                *[self.embedding_func(batch) for batch in batches]
            )
            embeddings = np.concatenate(embeddings_list)
            for i, d in enumerate(list_data):
                d["__vector__"] = embeddings[i]
            results = self._client.upsert(datas=list_data)

            # Sync in-memory reverse index
            if hasattr(self, "_entity_name2id") or hasattr(self, "_rel_pair2id"):
                await self._update_reverse_index(list_data)
            return results
        except Exception as e:
            logger.error(f"Failed to upsert data to vector DB: {e}")
            raise

    async def _update_reverse_index(self, list_data: List[Dict[str, Any]]):
        """Update reverse index after upsert"""
        updated_entity = False
        updated_rel = False
        for d in list_data:
            if "entity_name" in d:
                nm = d["entity_name"]
                self._entity_name2id[nm] = d["__id__"]
                updated_entity = True
            if "src_id" in d and "tgt_id" in d:
                src = d["src_id"]
                tgt = d["tgt_id"]
                self._rel_pair2id[(src, tgt)] = d["__id__"]
                updated_rel = True
        # Save all at once
        if updated_entity or updated_rel:
            await self._save_rev_idx()

    async def query(self, query: str, top_k=5):
        try:
            embedding = await self.embedding_func([query])
            embedding = embedding[0]
            results = self._client.query(
                query=embedding,
                top_k=top_k,
                better_than_threshold=self.cosine_better_than_threshold,
            )
            results = [
                {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]}
                for dp in results
            ]
            return results
        except Exception as e:
            logger.error(f"Failed to query vector DB: {e}")
            return []

    # Get entity_id by entity name
    async def get_entity_id(self, name: str) -> Optional[str]:
        if not hasattr(self, "_entity_name2id"):
            return None
        return self._entity_name2id.get(name)

    # Get rel_id by src and tgt
    async def get_rel_id(self, src: str, tgt: str) -> Optional[str]:
        if not hasattr(self, "_rel_pair2id"):
            return None
        return self._rel_pair2id.get((src, tgt))

    async def get_entity_ids(self, names: List[str]) -> Dict[str, Optional[str]]:
        return {n: await self.get_entity_id(n) for n in names}

    async def index_done_callback(self) -> None:
        try:
            self._client.save()
        except Exception as e:
            logger.error(f"Failed to save vector DB: {e}")
            raise


@dataclass
class NetworkXStorage(BaseGraphStorage):
    # KV for storing index
    _idx_kv: Optional[JsonKVStorage] = None

    @staticmethod
    def load_nx_graph(file_name: str) -> Optional[nx.Graph]:
        if os.path.exists(file_name):
            try:
                return nx.read_graphml(file_name)
            except Exception as e:
                logger.error(f"Failed to load graph from {file_name}: {e}")
                return None
        return None

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name: str):
        try:
            logger.info(
                f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
            )
            nx.write_graphml(graph, file_name)
        except Exception as e:
            logger.error(f"Failed to write graph to {file_name}: {e}")
            raise

    @staticmethod
    def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Return the largest connected component of the graph, with nodes and edges sorted in a stable way.
        """
        from graspologic.utils import largest_connected_component

        graph = graph.copy()
        graph = cast(nx.Graph, largest_connected_component(graph))
        node_mapping = {
            node: html.unescape(node.upper().strip()) for node in graph.nodes()
        }  # type: ignore
        graph = nx.relabel_nodes(graph, node_mapping)
        return NetworkXStorage._stabilize_graph(graph)

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Ensure an undirected graph with the same relationships will always be read the same way.
        """
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

        sorted_nodes = graph.nodes(data=True)
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

        fixed_graph.add_nodes_from(sorted_nodes)
        edges = list(graph.edges(data=True))

        if not graph.is_directed():

            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    temp = source
                    source = target
                    target = temp
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]

        def _get_edge_key(source: Any, target: Any) -> str:
            return f"{source} -> {target}"

        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

        fixed_graph.add_edges_from(edges)
        return fixed_graph

    def __post_init__(self):
        self._graphml_xml_file = os.path.join(
            self.global_config["working_dir"], f"graph_{self.namespace}.graphml"
        )
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        self._graph = preloaded_graph or nx.Graph()

        # Initialize chunk index
        self._init_chunk_index()

        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    def _init_chunk_index(self):
        """Initialize chunk-based index"""
        try:
            # 1) Create inverted index KV
            self._idx_kv = JsonKVStorage(
                namespace=f"chunkId2kg_idx",
                global_config=self.global_config,
            )
            # 2) Load index and convert list to set
            raw = self._idx_kv._data
            # chunk_to_nodes: {chunk: [node_id, ...], ...}
            self._chunk_to_nodes = defaultdict(set)
            for chunk, lst in raw.get("chunk_to_nodes", {}).items():
                self._chunk_to_nodes[chunk] = set(lst)
            # chunk_to_edges: {chunk: [[u,v], ...], ...}
            self._chunk_to_edges = defaultdict(set)
            for chunk, lst in raw.get("chunk_to_edges", {}).items():
                self._chunk_to_edges[chunk] = set((u, v) for u, v in lst)
            # 3) If empty, build from graph and write back
            if not self._chunk_to_nodes and not self._chunk_to_edges:
                self._build_index()
                # Serialize and write back
                self._idx_kv._data["chunk_to_nodes"] = {
                    k: list(v) for k, v in self._chunk_to_nodes.items()
                }
                self._idx_kv._data["chunk_to_edges"] = {
                    k: [list(e) for e in v] for k, v in self._chunk_to_edges.items()
                }
                # Async write JSON
                asyncio.get_event_loop().create_task(self._idx_kv.index_done_callback())
        except Exception as e:
            logger.error(f"Failed to initialize chunk index: {e}")
            self._chunk_to_nodes = defaultdict(set)
            self._chunk_to_edges = defaultdict(set)

    async def index_done_callback(self) -> None:
        try:
            NetworkXStorage.write_nx_graph(self._graph, self._graphml_xml_file)

            # Serialize and save in-memory inverted index to KV
            if self._idx_kv is not None:
                self._idx_kv._data["chunk_to_nodes"] = {
                    k: list(v) for k, v in self._chunk_to_nodes.items()
                }
                self._idx_kv._data["chunk_to_edges"] = {
                    k: [list(e) for e in v] for k, v in self._chunk_to_edges.items()
                }
                await self._idx_kv.index_done_callback()
        except Exception as e:
            logger.error(f"Failed in index_done_callback: {e}")
            raise

    async def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> Union[dict, None]:
        return self._graph.nodes.get(node_id)

    # Batch get nodes info - new
    async def get_nodes(self, node_ids: List[str]) -> List[Union[dict, None]]:
        return [self._graph.nodes.get(node_id) for node_id in node_ids]

    async def node_degree(self, node_id: str) -> int:
        if not self._graph.has_node(node_id):
            return 0
        return self._graph.degree(node_id)

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        src_degree = self._graph.degree(src_id) if self._graph.has_node(src_id) else 0
        tgt_degree = self._graph.degree(tgt_id) if self._graph.has_node(tgt_id) else 0
        return src_degree + tgt_degree

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        edge_data = self._graph.edges.get((source_node_id, target_node_id))
        if edge_data is None:
            # Try reverse search if not found
            edge_data = self._graph.edges.get((target_node_id, source_node_id))
        return edge_data

    # Batch get edges info - new
    async def get_edges(
        self, edge_pairs: List[Tuple[str, str]]
    ) -> List[Union[dict, None]]:
        result = []
        for source_id, target_id in edge_pairs:
            edge_data = self._graph.edges.get((source_id, target_id))
            if edge_data is None:
                # Try reverse search if not found
                edge_data = self._graph.edges.get((target_id, source_id))
            result.append(edge_data)
        return result

    # Get all neighbor node IDs of the given node - new
    async def get_neighbors(self, node_id: str) -> List[str]:
        if not self._graph.has_node(node_id):
            return []
        return list(self._graph.neighbors(node_id))

    async def get_node_edges(
        self, source_node_id: str
    ) -> Union[List[Tuple[str, str]], None]:
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id))
        return None

    # Batch get all incident edges of nodes - new
    async def get_nodes_edges(
        self, node_ids: List[str]
    ) -> Dict[str, Union[List[Tuple[str, str]], None]]:
        result = {}
        for node_id in node_ids:
            if self._graph.has_node(node_id):
                result[node_id] = list(self._graph.edges(node_id))
            else:
                result[node_id] = None
        return result

    # Batch get nodes info, optionally with neighbors - new
    async def get_nodes_by_ids(
        self, node_ids: List[str], include_neighbors: bool = False
    ) -> Dict[str, Any]:
        """Batch get node info, optionally including neighbors

        Args:
            node_ids: List of node IDs
            include_neighbors: Whether to include neighbors for each node

        Returns:
            Dict as {node_id: {data: node_data, neighbors: [neighbor_ids]}}
        """
        result = {}
        for node_id in node_ids:
            node_data = self._graph.nodes.get(node_id)
            if node_data is not None:
                node_info = {"data": node_data}
                if include_neighbors:
                    node_info["neighbors"] = await self.get_neighbors(node_id)
                result[node_id] = node_info
            else:
                result[node_id] = None
        return result

    # Override upsert_node, keep index in sync
    async def upsert_node(self, node_id: str, node_data: Dict[str, Any]) -> None:
        try:
            self._graph.add_node(node_id, **node_data)
            src = node_data.get("source_id")
            if isinstance(src, str):
                for chunk in src.split("<SEP>"):
                    self._chunk_to_nodes[chunk].add(node_id)
        except Exception as e:
            logger.error(f"Failed to upsert node {node_id}: {e}")
            raise

    # Override upsert_edge, keep index in sync
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: Dict[str, Any]
    ) -> None:
        try:
            self._graph.add_edge(source_node_id, target_node_id, **edge_data)
            src = edge_data.get("source_id")
            if isinstance(src, str):
                for chunk in src.split("<SEP>"):
                    self._chunk_to_edges[chunk].add((source_node_id, target_node_id))
        except Exception as e:
            logger.error(
                f"Failed to upsert edge {source_node_id}->{target_node_id}: {e}"
            )
            raise

    async def embed_nodes(self, algorithm: str) -> Tuple[np.ndarray, List[str]]:
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self._node_embed_algorithms[algorithm]()

    async def _node2vec_embed(self):
        from graspologic import embed

        embeddings, nodes = embed.node2vec_embed(
            self._graph,
            **self.global_config["node2vec_params"],
        )

        nodes_ids = [self._graph.nodes[node_id]["id"] for node_id in nodes]
        return embeddings, nodes_ids

    def _build_index(self):
        """Scan all nodes/edges of self._graph and build source_id inverted index"""
        SEP = "<SEP>"
        # Build node index
        for n, attrs in self._graph.nodes(data=True):
            src = attrs.get("source_id")
            if isinstance(src, str):
                for chunk in src.split(SEP):
                    self._chunk_to_nodes[chunk].add(n)
        # Build edge index
        for u, v, attrs in self._graph.edges(data=True):
            src = attrs.get("source_id")
            if isinstance(src, str):
                for chunk in src.split(SEP):
                    self._chunk_to_edges[chunk].add((u, v))

    # Use inverted index for fast query
    async def find_by_chunk(self, chunk_id: str):
        try:
            nodes = [
                (n, self._graph.nodes[n])
                for n in self._chunk_to_nodes.get(chunk_id, ())
            ]
            edges = [
                (u, v, self._graph.edges[u, v])
                for u, v in self._chunk_to_edges.get(chunk_id, ())
            ]
            return nodes, edges
        except Exception as e:
            logger.error(f"Failed to find by chunk {chunk_id}: {e}")
            return [], []
