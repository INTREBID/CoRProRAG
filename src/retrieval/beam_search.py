"""Beam search for multi-hop reasoning on knowledge graphs."""

import asyncio
import random

from src.utils import (
    truncate_list_by_token_size_async,
    truncate_by_token_size_async,
    normalize_list,
)
from src.llm import rerank_texts
from src.retrieval.graph_ops import get_one_hop_path


def _initialize_beam(entities, beam_width=15):
    """Initialize beam structure."""
    start_nodes = entities[:beam_width]
    return [
        {
            "nodes": [node],
            "edges": [],
            "paths": [],
            "weighted_score": node["weighted_score"],
            "is_used": False,
        }
        for node in start_nodes
    ]


def _extract_current_nodes_from_beam(beam):
    """Extract current terminal nodes from beam."""
    current_nodes = []
    unique_current_nodes = set()

    for b in beam:
        if not b["nodes"]:
            continue
        the_node_name = b["nodes"][-1]["entity_name"]
        the_path_score = b["weighted_score"]
        if the_node_name not in unique_current_nodes:
            current_nodes.append(
                {"entity_name": the_node_name, "weighted_score": the_path_score}
            )
            unique_current_nodes.add(the_node_name)

    return current_nodes


def _filter_visited_paths(hop_paths, visited_entities):
    """Filter out already-visited paths."""
    filtered_hop_paths = []
    for path in hop_paths:
        if path["tgt_name"] not in visited_entities:
            filtered_hop_paths.append(path)
            visited_entities.add(path["tgt_name"])
    return filtered_hop_paths


def _calculate_text_score(path, text_chunks):
    """Compute text score S_text(r, e) by summing max chunk scores for relation and entity."""
    rel_scores = [
        chunk["weighted_score"]
        for chunk in text_chunks[:30]
        if chunk["id"] in path["rel_chunk_id"]
    ]
    max_rel_score = max(rel_scores) if rel_scores else 0.0

    ent_scores = [
        chunk["weighted_score"]
        for chunk in text_chunks[:30]
        if chunk["id"] in path["ent_chunk_id"]
    ]
    max_ent_score = max(ent_scores) if ent_scores else 0.0

    return max_rel_score + max_ent_score


async def _calculate_path_scores(
    query,
    filtered_hop_paths,
    text_chunks,
    current_nodes,
    hop=0,
    alpha=1.0,
    beta=1.0,
    decay=0.9,
    rerank_texts_func=rerank_texts,
):
    """
    Calculate path scores combining structural and text scores with decay.

    S_hop = alpha * S_struct + beta * S_text
    score(p_h) = score(p_{h-1}) + decay^(h-1) * S_hop
    """
    if not filtered_hop_paths:
        return []

    sample_size = min(500, len(filtered_hop_paths))

    random.seed(52)
    sampled_paths = random.sample(filtered_hop_paths, sample_size)
    hop_path_strs = [
        p["src_name"] + p["description"] + p["tgt_description"] for p in sampled_paths
    ]

    hop_ranks = await rerank_texts_func(query=query, texts=hop_path_strs)

    normalize_list(hop_ranks, key="relevance_score")

    for path in sampled_paths:
        path["text_score"] = _calculate_text_score(path, text_chunks)
    normalize_list(sampled_paths, key="text_score")

    decay_factor = decay**hop

    for path, rank in zip(sampled_paths, hop_ranks):
        s_struct = rank["relevance_score"]
        s_text = path["text_score"]
        s_hop = alpha * s_struct + beta * s_text

        prev_score = 0.0
        for node in current_nodes:
            if node["entity_name"] == path["src_name"]:
                prev_score = node["weighted_score"]
                break

        path["weighted_score"] = prev_score + decay_factor * s_hop

    normalize_list(sampled_paths, key="weighted_score")

    return sorted(sampled_paths, key=lambda x: x["weighted_score"], reverse=True)


def _create_new_beam_item(beam_item, path):
    """Create a new beam item by extending with a path."""
    new_nodes = beam_item["nodes"].copy()
    new_nodes.append(
        {
            "entity_name": path["tgt_name"],
            "description": path["tgt_description"],
        }
    )

    new_edges = beam_item["edges"].copy()
    new_edges.append(
        {
            "src_name": path["src_name"],
            "description": path["description"],
            "tgt_name": path["tgt_name"],
            "keywords": path["keywords"],
        }
    )

    new_paths = beam_item["paths"].copy()
    new_paths.append(path)

    return {
        "nodes": new_nodes,
        "edges": new_edges,
        "paths": new_paths,
        "weighted_score": path["weighted_score"],
        "is_used": False,
    }


def _update_beam(beam, filtered_hop_paths, beam_width=15):
    """Update beam with new scored paths."""
    new_beam = []

    top_paths = filtered_hop_paths[:beam_width]

    for path in top_paths:
        src_entity = path["src_name"]
        for b in beam:
            if b["nodes"] and b["nodes"][-1]["entity_name"] == src_entity:
                new_beam.append(_create_new_beam_item(b, path))
                b["is_used"] = True

    if len(new_beam) < beam_width:
        remaining = beam_width - len(new_beam)
        unused = [b for b in beam if not b["is_used"]]
        unused.sort(key=lambda x: x["weighted_score"], reverse=True)
        new_beam.extend(unused[:remaining])

    return new_beam[:beam_width]


async def beam_search(
    query,
    entities,
    text_chunks,
    knowledge_graph_inst,
    max_hops=3,
    beam_width=15,
    alpha=1.0,
    beta=1.0,
    decay=0.9,
    rerank_texts_func=rerank_texts,
):
    """
    Multi-hop beam search on the knowledge graph.

    Iteratively expands terminal nodes, scores paths using structural
    and text relevance with decay, and retains top-W paths.
    """
    beam = _initialize_beam(entities, beam_width)
    visited_entities = {node["entity_name"] for node in entities[:beam_width]}

    for hop in range(max_hops):
        current_nodes = _extract_current_nodes_from_beam(beam)
        if not current_nodes:
            break

        hop_paths = await get_one_hop_path(
            [node["entity_name"] for node in current_nodes], knowledge_graph_inst
        )

        filtered_hop_paths = _filter_visited_paths(hop_paths, visited_entities)
        if not filtered_hop_paths:
            break

        scored_paths = await _calculate_path_scores(
            query,
            filtered_hop_paths,
            text_chunks,
            current_nodes,
            hop=hop,
            alpha=alpha,
            beta=beta,
            decay=decay,
            rerank_texts_func=rerank_texts_func,
        )

        beam = _update_beam(beam, scored_paths, beam_width)

    return beam


def _extract_from_complete_paths(beam, final_entities, final_relationships, max_hops):
    """Extract entities and relations from complete beam paths."""
    if beam is None:
        return

    for b in beam:
        if len(b["nodes"]) >= 1:
            for node in b["nodes"]:
                if node["entity_name"] not in [
                    e["entity_name"] for e in final_entities
                ]:
                    final_entities.append(node)

            for edge in b["edges"]:
                if edge["description"] not in [
                    e["description"] for e in final_relationships
                ]:
                    final_relationships.append(edge)


def _format_context(entities, relationships, chunks):
    """Format context string from entities, relationships, and chunks."""
    str_entities = "\n".join(
        [f"{e['entity_name']}: {e['description']}" for e in entities]
    )
    str_relationships = "\n".join(
        [f"{r['src_name']} {r['description']} {r['tgt_name']}" for r in relationships]
    )
    str_chunks = "\n".join([f"{c['content']}" for c in chunks])

    return str_chunks + "\n\n" + str_relationships + "\n\n" + str_entities


async def process_beam_output(
    beam,
    entities,
    relationships,
    text_chunks,
    max_hops,
    max_entities_tokens=4000,
    max_relationships_tokens=4000,
    max_chunks_tokens=4000,
):
    """Process beam search output into truncated context."""
    entities = entities or []
    relationships = relationships or []
    text_chunks = text_chunks or []

    final_entities = entities[:15]
    final_relationships = relationships[:15]

    _extract_from_complete_paths(beam, final_entities, final_relationships, max_hops)

    final_entities = await truncate_list_by_token_size_async(
        final_entities, lambda x: x["description"], max_token_size=max_entities_tokens
    )
    final_relationships = await truncate_list_by_token_size_async(
        final_relationships,
        lambda x: x["description"],
        max_token_size=max_relationships_tokens,
    )
    final_chunks = await truncate_by_token_size_async(
        text_chunks, lambda x: x["content"], max_token_size=max_chunks_tokens
    )

    context = _format_context(final_entities, final_relationships, final_chunks)

    return final_entities, final_relationships, final_chunks, context
