"""Fusion retrieval: combining knowledge graph and text retrieval."""

import asyncio
import json

from src.utils import (
    logger,
    parse_json_response,
    normalize_list,
)
from src.llm import rerank_texts
from src.prompt import PROMPTS


async def _extract_keywords_relationships(query, use_model_func):
    """Extract keywords and relationships from the query."""
    extraction_prompt = PROMPTS["keywords_relationship_extraction"].format(query=query)
    llm_result = await use_model_func(prompt=extraction_prompt)

    try:
        return await parse_json_response(llm_result, use_model_func)
    except json.JSONDecodeError:
        llm_result = await use_model_func(prompt=extraction_prompt)
        try:
            return await parse_json_response(llm_result, use_model_func)
        except json.JSONDecodeError:
            print("Failed to extract keywords and relationships!")
            return None


async def _parallel_retrieval(
    query,
    keywords_relationships,
    chunks_vdb,
    entities_vdb,
    relationships_vdb,
    top_k,
    etop_k,
):
    """Retrieve chunks, entities, and relations in parallel."""
    keywords_str = ",".join(keywords_relationships["keywords"])
    relationships_list = keywords_relationships["relationships"]

    chunks_task = asyncio.create_task(chunks_vdb.query(query, top_k=top_k))
    entities_task = asyncio.create_task(entities_vdb.query(keywords_str, top_k=etop_k))
    relations_tasks = [
        asyncio.create_task(relationships_vdb.query(rel_query, top_k=15))
        for rel_query in relationships_list
    ]

    gathered = await asyncio.gather(chunks_task, entities_task, *relations_tasks)
    chunks_result, entities_result, *relations_results = gathered

    unique_items = {}
    for sublist in relations_results:
        for item in sublist:
            item_id = item["id"]
            if (
                item_id not in unique_items
                or item["distance"] > unique_items[item_id]["distance"]
            ):
                unique_items[item_id] = item

    return chunks_result, entities_result, list(unique_items.values())


async def _parallel_retrieval_with_rerank(
    query,
    keywords_relationships,
    chunks_vdb,
    entities_vdb,
    relationships_vdb,
    top_k,
    etop_k,
    rerank_texts_func=rerank_texts,
):
    """Retrieve chunks, entities, and relations in parallel with reranking."""
    keywords_str = ",".join(keywords_relationships["keywords"])
    relationships_list = keywords_relationships["relationships"]

    chunks_task = asyncio.create_task(chunks_vdb.query(query, top_k=top_k))
    entities_task = asyncio.create_task(entities_vdb.query(keywords_str, top_k=etop_k))
    relations_tasks = [
        asyncio.create_task(relationships_vdb.query(rel_query, top_k=15))
        for rel_query in relationships_list
    ]

    gathered = await asyncio.gather(chunks_task, entities_task, *relations_tasks)
    chunks_result, entities_result, *relations_results = gathered

    unique_items = {}
    for sublist in relations_results:
        for item in sublist:
            item_id = item["id"]
            if (
                item_id not in unique_items
                or item["distance"] > unique_items[item_id]["distance"]
            ):
                unique_items[item_id] = item
    relations_result = list(unique_items.values())

    chunk_texts = [chunk.get("content", "") for chunk in chunks_result]
    entity_texts = [entity.get("entity_name", "") for entity in entities_result]
    relation_texts = [
        f"{rel.get('src_id', '')} -> {rel.get('tgt_id', '')}"
        for rel in relations_result
    ]

    async def empty_rerank():
        return []

    rerank_tasks = [
        rerank_texts_func(query=query, texts=chunk_texts)
        if chunk_texts
        else empty_rerank(),
        rerank_texts_func(query=query, texts=entity_texts)
        if entity_texts
        else empty_rerank(),
        rerank_texts_func(query=query, texts=relation_texts)
        if relation_texts
        else empty_rerank(),
    ]

    (
        chunk_rerank_scores,
        entity_rerank_scores,
        relation_rerank_scores,
    ) = await asyncio.gather(*rerank_tasks)

    for i, chunk in enumerate(chunks_result):
        if i < len(chunk_rerank_scores):
            chunk["__metrics__"] = chunk_rerank_scores[i].get("relevance_score", 0.0)

    for i, entity in enumerate(entities_result):
        if i < len(entity_rerank_scores):
            entity["__metrics__"] = entity_rerank_scores[i].get("relevance_score", 0.0)

    for i, relation in enumerate(relations_result):
        if i < len(relation_rerank_scores):
            relation["__metrics__"] = relation_rerank_scores[i].get(
                "relevance_score", 0.0
            )

    return chunks_result, entities_result, relations_result


def _initialize_retrieval_dicts(chunks_result, entities_result, relations_results):
    """Initialize chunk, entity, and relation score dicts."""
    text_dict = {
        chunk["id"]: {
            "id": chunk["id"],
            "weighted_score": chunk["__metrics__"],
        }
        for chunk in chunks_result[:15]
    }
    entities_dict = {
        entity["id"]: {
            "id": entity["id"],
            "entity_name": entity["entity_name"],
            "weighted_score": entity["__metrics__"],
        }
        for entity in entities_result[:20]
    }
    relationships_dict = {
        relation["id"]: {
            "id": relation["id"],
            "src_name": relation["src_id"],
            "tgt_name": relation["tgt_id"],
            "weighted_score": relation["__metrics__"],
        }
        for relation in relations_results[:5]
    }

    return text_dict, entities_dict, relationships_dict


async def _add_entity_relation_descriptions(entities, relationships, kg_graph):
    """Add descriptions for entities and relations in parallel."""

    async def get_entity_description(entity):
        entity_name = entity["entity_name"]
        entity_content = await kg_graph.get_node(entity_name)
        entity["description"] = entity_content.get("description", "")

    async def get_relationship_description(relationship):
        src_name = relationship["src_name"]
        tgt_name = relationship["tgt_name"]
        relationship_content = await kg_graph.get_edge(src_name, tgt_name)
        relationship["description"] = relationship_content.get("description", "")

    await asyncio.gather(
        *[get_entity_description(entity) for entity in entities],
        *[get_relationship_description(relationship) for relationship in relationships],
    )


async def _compute_chunk_to_entity_contributions(
    text_dict, knowledge_graph, entity_vdb, chunk_to_ent_wt
):
    """Compute chunk-to-entity score contributions."""
    c_to_e_contrib = {}

    for cid, chunk_item in text_dict.items():
        score = chunk_item["weighted_score"]
        ents, _ = await knowledge_graph.find_by_chunk(cid)

        for entity_name, entity_attr in ents:
            eid = await entity_vdb.get_entity_id(name=entity_name)
            contrib = score * chunk_to_ent_wt
            if eid not in c_to_e_contrib:
                c_to_e_contrib[eid] = {
                    "weighted_score": contrib,
                    "entity_name": entity_name,
                }
            else:
                c_to_e_contrib[eid]["weighted_score"] += contrib

    return c_to_e_contrib


async def _compute_chunk_to_relation_contributions(
    text_dict, knowledge_graph, relationship_vdb, chunk_to_rel_wt
):
    """Compute chunk-to-relation score contributions."""
    c_to_r_contrib = {}

    for cid, chunk_item in text_dict.items():
        score = chunk_item["weighted_score"]
        _, rels = await knowledge_graph.find_by_chunk(cid)

        for src, tgt, rel_attr in rels:
            rid = await relationship_vdb.get_rel_id(src=src, tgt=tgt)
            if rid is None:
                rid = await relationship_vdb.get_rel_id(src=tgt, tgt=src)
            contrib = score * chunk_to_rel_wt
            if rid not in c_to_r_contrib:
                c_to_r_contrib[rid] = {
                    "weighted_score": contrib,
                    "src_name": src,
                    "tgt_name": tgt,
                }
            else:
                c_to_r_contrib[rid]["weighted_score"] += contrib

    return c_to_r_contrib


async def _compute_entity_to_chunk_contributions(
    entities_dict, knowledge_graph, ent_to_chunk_wt
):
    """Compute entity-to-chunk score contributions."""
    e_to_c_contrib = {}

    for eid, entity_item in entities_dict.items():
        score = entity_item["weighted_score"]
        entity_name = entity_item["entity_name"]
        node = await knowledge_graph.get_node(entity_name)
        chunk_id_list = node.get("source_id", "")
        chunk_id_list = chunk_id_list.split("<SEP>") if chunk_id_list else []

        for chunk_id in chunk_id_list:
            contrib = score * ent_to_chunk_wt
            if chunk_id not in e_to_c_contrib:
                e_to_c_contrib[chunk_id] = contrib
            else:
                e_to_c_contrib[chunk_id] += contrib

    return e_to_c_contrib


async def _compute_relation_to_chunk_contributions(
    relationships_dict, knowledge_graph, rel_to_chunk_wt
):
    """Compute relation-to-chunk score contributions."""
    r_to_c_contrib = {}

    for rid, relation_item in relationships_dict.items():
        score = relation_item["weighted_score"]
        src_name = relation_item["src_name"]
        tgt_name = relation_item["tgt_name"]
        edge = await knowledge_graph.get_edge(src_name, tgt_name)
        chunk_id_list = edge.get("source_id", "")
        chunk_id_list = chunk_id_list.split("<SEP>") if chunk_id_list else []

        for chunk_id in chunk_id_list:
            contrib = score * rel_to_chunk_wt
            if chunk_id not in r_to_c_contrib:
                r_to_c_contrib[chunk_id] = contrib
            else:
                r_to_c_contrib[chunk_id] += contrib

    return r_to_c_contrib


def _apply_contributions_to_dicts(
    text_dict,
    entities_dict,
    relationships_dict,
    c_to_e_contrib,
    c_to_r_contrib,
    e_to_c_contrib,
    r_to_c_contrib,
):
    """Apply all cross-contributions to the score dicts."""
    for chunk_id, contrib in e_to_c_contrib.items():
        if chunk_id not in text_dict:
            text_dict[chunk_id] = {"id": chunk_id, "weighted_score": contrib}
        else:
            text_dict[chunk_id]["weighted_score"] += contrib

    for chunk_id, contrib in r_to_c_contrib.items():
        if chunk_id not in text_dict:
            text_dict[chunk_id] = {"id": chunk_id, "weighted_score": contrib}
        else:
            text_dict[chunk_id]["weighted_score"] += contrib

    for eid, contrib in c_to_e_contrib.items():
        if eid and eid not in entities_dict:
            entities_dict[eid] = {
                "id": eid,
                "entity_name": contrib["entity_name"],
                "weighted_score": contrib["weighted_score"],
            }
        elif eid:
            entities_dict[eid]["weighted_score"] += contrib["weighted_score"]

    for rid, contrib in c_to_r_contrib.items():
        if rid not in relationships_dict:
            relationships_dict[rid] = {
                "id": rid,
                "src_name": contrib["src_name"],
                "tgt_name": contrib["tgt_name"],
                "weighted_score": contrib["weighted_score"],
            }
        else:
            relationships_dict[rid]["weighted_score"] += contrib["weighted_score"]


def _sort_and_truncate_dicts(text_dict, entities_dict, relationships_dict, top_k=60):
    """Sort and truncate dicts, keeping top-k entries."""
    text_dict = dict(
        sorted(text_dict.items(), key=lambda x: x[1]["weighted_score"], reverse=True)[
            :top_k
        ]
    )
    entities_dict = dict(
        sorted(
            entities_dict.items(), key=lambda x: x[1]["weighted_score"], reverse=True
        )[:top_k]
    )
    relationships_dict = dict(
        sorted(
            relationships_dict.items(),
            key=lambda x: x[1]["weighted_score"],
            reverse=True,
        )[:top_k]
    )
    return text_dict, entities_dict, relationships_dict


async def _finalize_results(
    text_dict, entities_dict, relationships_dict, text_chunks_db, knowledge_graph
):
    """Finalize results: fetch chunk content and descriptions."""
    ranked_chunks = list(text_dict.values())
    chunks_ids = [chunk["id"] for chunk in ranked_chunks[:60]]
    ranked_chunks_content = await text_chunks_db.get_by_ids(chunks_ids)
    for chunk, chunk_content in zip(ranked_chunks, ranked_chunks_content):
        chunk["content"] = chunk_content["content"]

    ranked_entities = list(entities_dict.values())
    ranked_relationships = list(relationships_dict.values())

    await _add_entity_relation_descriptions(
        ranked_entities, ranked_relationships, knowledge_graph
    )

    return ranked_chunks, ranked_entities, ranked_relationships


async def process_retrieval_results(
    chunks_result, entities_result, relations_results, knowledge_graph
):
    """Process retrieval results: fetch details and normalize scores."""
    chunks = []
    entities = []
    relationships = []

    for chunk in chunks_result:
        chunks.append(
            {
                "id": chunk["id"],
                "weighted_score": chunk["__metrics__"],
                "content": chunk["content"],
            }
        )

    entity_name_list = [entity["entity_name"] for entity in entities_result]
    entities_with_content = await knowledge_graph.get_nodes(entity_name_list)
    for entity, entity_content in zip(entities_result, entities_with_content):
        entities.append(
            {
                "id": entity["id"],
                "entity_name": entity["entity_name"],
                "weighted_score": entity["__metrics__"],
                "description": entity_content["description"],
            }
        )

    relationship_key_list = [
        (relation["src_id"], relation["tgt_id"]) for relation in relations_results
    ]
    relationships_with_content = await knowledge_graph.get_edges(relationship_key_list)
    for relation, relation_content in zip(
        relations_results, relationships_with_content
    ):
        relationships.append(
            {
                "id": relation["id"],
                "src_name": relation["src_id"],
                "tgt_name": relation["tgt_id"],
                "weighted_score": relation["__metrics__"],
                "description": relation_content["description"],
            }
        )

    normalize_list(list_objs=entities, key="weighted_score")
    normalize_list(list_objs=relationships, key="weighted_score")
    normalize_list(list_objs=chunks, key="weighted_score")

    return chunks, entities, relationships


async def fuse_kg_text(
    query,
    chunks_vdb,
    text_chunks_db,
    knowledge_graph,
    entity_vdb,
    relationship_vdb,
    use_model_func,
    hyperparams=None,
):
    """
    Fuse knowledge graph and text retrieval with cross-weighting.

    Returns ranked chunks, entities, and relationships.
    """

    if hyperparams is None:
        hyperparams = {}

    max_token_size = hyperparams.get("max_token_size", 4000)
    ent_to_chunk_wt = hyperparams.get("ent_to_chunk_wt", 0.5)
    rel_to_chunk_wt = hyperparams.get("rel_to_chunk_wt", 0.5)
    chunk_to_ent_wt = hyperparams.get("chunk_to_ent_wt", 0.5)
    chunk_to_rel_wt = hyperparams.get("chunk_to_rel_wt", 0.5)
    etop_k = hyperparams.get("etop_k", 60)
    rtop_k = hyperparams.get("rtop_k", 60)
    top_k = hyperparams.get("top_k", 60)

    iter_times = 1

    try:
        keywords_relationships = await _extract_keywords_relationships(
            query, use_model_func
        )
    except Exception:
        logger.error("Failed to extract keywords and relationships!")
        return "\nFailed to extract keywords and relationships!\n"

    chunks_result, entities_result, relations_results = await _parallel_retrieval(
        query,
        keywords_relationships,
        chunks_vdb,
        entity_vdb,
        relationship_vdb,
        top_k,
        etop_k,
    )

    chunks, entities, relationships = await process_retrieval_results(
        chunks_result, entities_result, relations_results, knowledge_graph
    )

    text_dict, entities_dict, relationships_dict = _initialize_retrieval_dicts(
        chunks_result, entities_result, relations_results
    )

    for i in range(iter_times):
        (
            c_to_e_contrib,
            c_to_r_contrib,
            e_to_c_contrib,
            r_to_c_contrib,
        ) = await asyncio.gather(
            _compute_chunk_to_entity_contributions(
                text_dict, knowledge_graph, entity_vdb, chunk_to_ent_wt
            ),
            _compute_chunk_to_relation_contributions(
                text_dict, knowledge_graph, relationship_vdb, chunk_to_rel_wt
            ),
            _compute_entity_to_chunk_contributions(
                entities_dict, knowledge_graph, ent_to_chunk_wt
            ),
            _compute_relation_to_chunk_contributions(
                relationships_dict, knowledge_graph, rel_to_chunk_wt
            ),
        )

        _apply_contributions_to_dicts(
            text_dict,
            entities_dict,
            relationships_dict,
            c_to_e_contrib,
            c_to_r_contrib,
            e_to_c_contrib,
            r_to_c_contrib,
        )

        text_dict, entities_dict, relationships_dict = _sort_and_truncate_dicts(
            text_dict, entities_dict, relationships_dict, 50
        )

        normalize_list(text_dict.values(), "weighted_score")
        normalize_list(entities_dict.values(), "weighted_score")
        normalize_list(relationships_dict.values(), "weighted_score")

        ranked_chunks, ranked_entities, ranked_relations = await _finalize_results(
            text_dict,
            entities_dict,
            relationships_dict,
            text_chunks_db,
            knowledge_graph,
        )

        return (
            ranked_chunks,
            ranked_entities,
            ranked_relations,
            chunks,
            entities,
            relationships,
        )
