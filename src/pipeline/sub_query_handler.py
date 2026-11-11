"""Sub-query handling for independent and dependent queries."""

from src.utils import parse_json_response
from src.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
)
from src.prompt import PROMPTS
from src.retrieval.fusion import fuse_kg_text
from src.retrieval.beam_search import beam_search, process_beam_output


async def beam_indep_query(
    sub_query,
    knowledge_graph_inst: BaseGraphStorage,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    hyperparams,
    use_model_func,
    rerank_texts_func,
) -> dict:
    # Fuse retrieval across KG and text
    (
        ranked_chunks,
        ranked_entities,
        ranked_relationships,
        chunks,
        entities,
        relationships,
    ) = await fuse_kg_text(
        sub_query,
        chunks_vdb,
        text_chunks_db,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        use_model_func,
        hyperparams=hyperparams,
    )

    max_hops = hyperparams.get("beam_search_max_hops", 2)
    beam_width = hyperparams.get("beam_search_beam_width", 15)
    alpha = hyperparams.get("beam_search_alpha", 1.0)
    beta = hyperparams.get("beam_search_beta", 1.0)
    decay = hyperparams.get("beam_search_decay", 0.9)

    # Beam search from fused retrieval results
    beam = await beam_search(
        sub_query,
        ranked_entities,
        ranked_chunks,
        knowledge_graph_inst,
        max_hops=max_hops,
        beam_width=beam_width,
        alpha=alpha,
        beta=beta,
        decay=decay,
        rerank_texts_func=rerank_texts_func,
    )

    # Process beam search results
    max_token_size = hyperparams.get("max_token_size", 4000)
    _, _, _, context = await process_beam_output(
        beam,
        ranked_entities,
        ranked_relationships,
        ranked_chunks,
        max_hops,
        max_entities_tokens=3000,
        max_relationships_tokens=3000,
        max_chunks_tokens=max_token_size,
    )

    # Build prompt
    indep_query_prompt = PROMPTS["answer_sub_query"].format(
        sub_query=sub_query, context=context
    )

    # Generate answer
    llm_result = await use_model_func(prompt=indep_query_prompt)
    llm_result = await parse_json_response(llm_result, use_model_func)

    return llm_result


async def beam_dep_query(
    query,
    sub_queries,
    dep_sub_query,
    knowledge_graph_inst: BaseGraphStorage,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    hyperparams,
    use_model_func,
    rerank_texts_func,
):
    # Fill in dependencies from previous sub-query answers
    dep_sub_query_content = dep_sub_query["query"]
    depend_on = dep_sub_query.get("depends_on", [])
    query_for_retrieval = ""
    query_for_answer = ""
    previous_info = ""
    has_unknow_answer = False
    has_not_insert = False
    completed_dep_sub_query = dep_sub_query_content

    for depend_on_idx in depend_on:
        try:
            answer_key = f"{{answer_{depend_on_idx}}}"
            previous_answer = sub_queries[depend_on_idx - 1]["answer"]
            previous_rationale = sub_queries[depend_on_idx - 1]["rationale"]
            previous_query = sub_queries[depend_on_idx - 1]["query"]
            if previous_answer == "I don't know.":
                has_unknow_answer = True
                previous_answer = " "
            if answer_key not in completed_dep_sub_query:
                has_not_insert = True
            completed_dep_sub_query = completed_dep_sub_query.replace(
                answer_key, previous_answer
            )

            previous_info += f"""Relevant question: {previous_query}
                Answer: {previous_answer}
                Rationale: {previous_rationale}"""
        except Exception as e:
            print(f"Error processing sub_queries[{depend_on_idx - 1}]: {e}")
            print(f"sub_queries content: {sub_queries}")

    dep_sub_query["query"] = completed_dep_sub_query

    # Handle case where dependency answers may be unknown or not inserted
    merge_query = query + "\n" + completed_dep_sub_query
    if has_unknow_answer:
        query_for_retrieval = merge_query
        query_for_answer = merge_query
    elif has_not_insert:
        query_for_retrieval = merge_query
        query_for_answer = f"""The questions and answers related to this issue are as follows: {previous_info}
        The question to be answered is as follows: {completed_dep_sub_query}"""
    else:
        query_for_retrieval = completed_dep_sub_query
        query_for_answer = completed_dep_sub_query

    # Fuse retrieval across KG and text
    (
        ranked_chunks,
        ranked_entities,
        ranked_relationships,
        chunks,
        entities,
        relationships,
    ) = await fuse_kg_text(
        query_for_retrieval,
        chunks_vdb,
        text_chunks_db,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        use_model_func,
        hyperparams=hyperparams,
    )

    max_hops = hyperparams.get("beam_search_max_hops", 2)
    beam_width = hyperparams.get("beam_search_beam_width", 15)
    alpha = hyperparams.get("beam_search_alpha", 1.0)
    beta = hyperparams.get("beam_search_beta", 1.0)
    decay = hyperparams.get("beam_search_decay", 0.9)

    # Beam search from fused retrieval results
    beam = await beam_search(
        query_for_retrieval,
        ranked_entities,
        ranked_chunks,
        knowledge_graph_inst,
        max_hops=max_hops,
        beam_width=beam_width,
        alpha=alpha,
        beta=beta,
        decay=decay,
        rerank_texts_func=rerank_texts_func,
    )

    # Process beam search results
    max_token_size = hyperparams.get("max_token_size", 4000)
    _, _, _, context = await process_beam_output(
        beam,
        ranked_entities,
        ranked_relationships,
        ranked_chunks,
        max_hops,
        max_entities_tokens=3000,
        max_relationships_tokens=3000,
        max_chunks_tokens=max_token_size,
    )

    # Build prompt
    indep_query_prompt = PROMPTS["answer_sub_query"].format(
        sub_query=query_for_answer, context=context
    )

    # Generate answer
    llm_result = await use_model_func(prompt=indep_query_prompt)
    llm_result = await parse_json_response(llm_result, use_model_func)

    return llm_result
