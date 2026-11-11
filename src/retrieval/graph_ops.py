"""Graph operations: entity/relation retrieval and path finding."""

import asyncio

from src.utils import (
    logger,
    truncate_list_by_token_size_async,
    truncate_by_token_size_async,
    split_chunk_ids,
)
from src.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_edges = set()
    for this_edges in all_related_edges:
        all_edges.update([tuple(sorted(e)) for e in this_edges])
    all_edges = list(all_edges)
    all_edges_pack = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]
    )
    all_edges_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
    )
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = await truncate_list_by_token_size_async(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )
    return all_edges_data


async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    entity_names = set()
    for e in edge_datas:
        entity_names.add(e["src_id"])
        entity_names.add(e["tgt_id"])

    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(entity_name) for entity_name in entity_names]
    )

    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(entity_name) for entity_name in entity_names]
    )
    node_datas = [
        {**n, "entity_name": k, "rank": d}
        for k, n, d in zip(entity_names, node_datas, node_degrees)
    ]

    node_datas = await truncate_list_by_token_size_async(
        node_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )

    return node_datas


async def get_chunks_by_query(
    query,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    top_k,
    max_token_size,
):
    results = await chunks_vdb.query(query, top_k=top_k)
    if not len(results):
        return "\nFailed to retrieve text chunks!\n"
    chunks_ids = [r["id"] for r in results[:40]]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)

    maybe_trun_chunks = await truncate_by_token_size_async(
        chunks,
        key=lambda x: x["content"],
        max_token_size=max_token_size,
    )
    logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
    section = "\n--New Chunk--\n".join([c["content"] for c in maybe_trun_chunks])
    return section


async def find_integrated_entity_relation(
    ll_keywords,
    hl_keywords,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    query_param: QueryParam,
):
    ll_task = process_low_level_keywords(
        ll_keywords, knowledge_graph_inst, entities_vdb, query_param
    )
    hl_task = process_high_level_keywords(
        hl_keywords, knowledge_graph_inst, relationships_vdb, query_param
    )

    ll_results, hl_results = await asyncio.gather(ll_task, hl_task)

    ll_node_datas, ll_relations = ll_results
    hl_entities, hl_edge_datas = hl_results

    # Merge entities, deduplicate
    merged_entities = {}

    for entity in ll_node_datas:
        entity_name = entity["entity_name"]
        merged_entities[entity_name] = entity

    for entity in hl_entities:
        entity_name = entity["entity_name"]
        if entity_name in merged_entities:
            continue
        else:
            merged_entities[entity_name] = entity

    # Merge relations, deduplicate
    merged_relations = {}

    for relation in ll_relations:
        relation_key = (relation["src_tgt"][0], relation["src_tgt"][1])
        merged_relations[relation_key] = {
            "src_id": relation["src_tgt"][0],
            "tgt_id": relation["src_tgt"][1],
            "description": relation["description"],
            "keywords": relation["keywords"],
            "weight": relation["weight"],
        }

    for relation in hl_edge_datas:
        relation_key = (relation["src_id"], relation["tgt_id"])
        if relation_key in merged_relations:
            continue
        else:
            merged_relations[relation_key] = relation

    entities = list(merged_entities.values())
    relations = list(merged_relations.values())

    return {"entities": entities, "relations": relations}


async def process_low_level_keywords(
    ll_keywords,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    query_param: QueryParam,
):
    """Process low-level keywords."""
    ll_result = await entities_vdb.query(ll_keywords, query_param.top_k)
    if not len(ll_result):
        logger.warning("No entities found for low-level keywords")
        ll_result = []

    unique_ll_result = {r["entity_name"]: r for r in ll_result}

    ll_node_datas = (
        await asyncio.gather(
            *[
                knowledge_graph_inst.get_node(r["entity_name"])
                for r in unique_ll_result.values()
            ]
        )
        if unique_ll_result
        else []
    )

    if ll_node_datas and not all([n is not None for n in ll_node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")

    ll_node_datas = (
        [
            {**n, "entity_name": k["entity_name"]}
            for n, k in zip(ll_node_datas, unique_ll_result.values())
            if n is not None
        ]
        if unique_ll_result
        else []
    )

    ll_relations = (
        await _find_most_related_edges_from_entities(
            ll_node_datas, query_param, knowledge_graph_inst
        )
        if ll_node_datas
        else []
    )

    return ll_node_datas, ll_relations


async def process_high_level_keywords(
    hl_keywords,
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    query_param: QueryParam,
):
    """Process high-level keywords."""
    hl_result = []
    if hl_keywords:
        hl_result = await relationships_vdb.query(hl_keywords, top_k=query_param.top_k)
        if not len(hl_result):
            hl_result = []
            logger.warning("No relationships found for high-level keywords")

    seen_relationships = set()
    unique_hl_results = []
    for r in hl_result:
        relationship_key = (r["src_id"], r["tgt_id"])
        if relationship_key not in seen_relationships:
            unique_hl_results.append(r)
            seen_relationships.add(relationship_key)

    hl_edge_datas = (
        await asyncio.gather(
            *[
                knowledge_graph_inst.get_edge(r["src_id"], r["tgt_id"])
                for r in unique_hl_results
            ]
        )
        if unique_hl_results
        else []
    )

    if hl_edge_datas and not all([n is not None for n in hl_edge_datas]):
        logger.warning("Some edges are missing, maybe the storage is damaged")

    hl_edge_datas = (
        [
            {"src_id": k["src_id"], "tgt_id": k["tgt_id"], **v}
            for k, v in zip(unique_hl_results, hl_edge_datas)
            if v is not None
        ]
        if unique_hl_results
        else []
    )

    hl_edge_datas = (
        await truncate_list_by_token_size_async(
            hl_edge_datas,
            key=lambda x: x["description"],
            max_token_size=query_param.max_token_for_global_context,
        )
        if hl_edge_datas
        else []
    )

    hl_entities = (
        await _find_most_related_entities_from_relationships(
            hl_edge_datas, query_param, knowledge_graph_inst
        )
        if hl_edge_datas
        else []
    )

    return hl_entities, hl_edge_datas


async def find_one_hop_entity_relationship(
    candidate_entities,
    entity_set,  # visited nodes so far
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    knowledge_graph_inst: BaseGraphStorage,
) -> dict:
    result_dict = {
        "paths": [],
        "origin_candidate_entities": entity_set.copy(),
        "new_candidate_entities": set(),
    }

    paths = []
    new_build_candidates = set()
    for entity in candidate_entities:
        relationships = await knowledge_graph_inst.get_node_edges(entity)
        if relationships is None:
            relationships = []
        filtered_relationships = [
            relationship
            for relationship in relationships
            if relationship[1] not in entity_set
        ]

        if filtered_relationships:
            relationship_pack = await asyncio.gather(
                *[
                    knowledge_graph_inst.get_edge(rel[0], rel[1])
                    for rel in filtered_relationships
                ]
            )
        else:
            relationship_pack = []
        for relationship, relationship_detail in zip(
            filtered_relationships, relationship_pack
        ):
            source_entity, target_entity = await asyncio.gather(
                knowledge_graph_inst.get_node(relationship[0]),
                knowledge_graph_inst.get_node(relationship[1]),
            )

            path = {
                "source_entity_name": relationship[0],
                "relationship_description": relationship_detail["description"],
                "target_entity_name": relationship[1],
                "target_entity_description": target_entity["description"],
            }
            new_build_candidates.add(relationship[1])
            paths.append(path)
    result_dict["new_candidate_entities"] = new_build_candidates
    result_dict["paths"] = paths

    return result_dict


async def get_one_hop_path(entity_list, knowledge_graph_inst: BaseGraphStorage):
    """Find one-hop paths from the given entity list."""
    paths = []
    node_edges_list = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(entity) for entity in entity_list]
    )

    all_rel_pairs = []
    for rel_result in node_edges_list:
        if rel_result:
            all_rel_pairs.extend(rel_result)

    if not all_rel_pairs:
        return paths

    relationships = await knowledge_graph_inst.get_edges(all_rel_pairs)
    tgt_ids = [src_tgt[1] for src_tgt in all_rel_pairs]
    tgt_entities = await knowledge_graph_inst.get_nodes(tgt_ids)

    for (src, tgt), relationship, tgt_entity in zip(
        all_rel_pairs, relationships, tgt_entities
    ):
        if not relationship or not tgt_entity:
            continue
        rel_chunk_id = split_chunk_ids(relationship.get("source_id", ""))
        ent_chunk_id = split_chunk_ids(tgt_entity.get("source_id", ""))
        paths.append(
            {
                "id": src + tgt,
                "src_name": src,
                "tgt_name": tgt,
                "keywords": relationship.get("keywords", ""),
                "description": relationship.get("description", ""),
                "tgt_description": tgt_entity.get("description", ""),
                "rel_chunk_id": rel_chunk_id,
                "ent_chunk_id": ent_chunk_id,
                "weighted_score": 0.0,
            }
        )

    return paths
