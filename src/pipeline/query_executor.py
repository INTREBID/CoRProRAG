"""Query execution entry points."""

import copy

from src.utils import (
    parse_json_response,
    truncate_by_token_size_async,
)
from src.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
)
from src.prompt import PROMPTS
from src.retrieval.fusion import fuse_kg_text
from src.retrieval.beam_search import beam_search, process_beam_output
from src.pipeline.sub_query_handler import beam_indep_query, beam_dep_query


async def corprorag_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    global_config: dict,
) -> dict:
    async def decompose_question(query: str, use_model_func) -> (list, str):
        """Decompose a question into sub-queries and a reasoning plan via LLM."""
        try:
            decompose_prompt = PROMPTS["question_decomposition"].format(query=query)
            decompose_response = await use_model_func(prompt=decompose_prompt)
            decompose_result = await parse_json_response(
                decompose_response, use_model_func
            )
            sub_queries = decompose_result.get("sub_queries", [])
            reasoning_plan = decompose_result.get("reasoning_plan", "")
            return sub_queries, reasoning_plan
        except Exception as e:
            print(f"Error occurred during decomposition: {e}")
            raise Exception(f"Decomposition failed, response: {decompose_response}")

    def classify_sub_queries(sub_queries: list) -> (list, list):
        """Split sub-queries into independent and dependent lists, adding indices."""
        dep_sub_queries = []
        indep_sub_queries = []
        for index, sub_query in enumerate(sub_queries):
            sub_query["index"] = index
            if sub_query.get("has_dependency", False):
                dep_sub_queries.append(sub_query)
            else:
                indep_sub_queries.append(sub_query)
        return indep_sub_queries, dep_sub_queries

    async def process_independent_subqueries(
        indep_sub_queries: list, depth: int = 0
    ) -> None:
        for indep_sub_query in indep_sub_queries:
            sub_query_content = indep_sub_query["query"]
            initial_sub_result = await beam_indep_query(
                sub_query_content,
                knowledge_graph_inst,
                chunks_vdb,
                text_chunks_db,
                entities_vdb,
                relationships_vdb,
                hyperparams,
                use_model_func,
                rerank_texts_func=rerank_texts_func,
            )
            sub_result = initial_sub_result
            if sub_result["answer"] == "I don't know.":
                sub_result = await process_unknown_subquery(
                    original_query=query,
                    sub_query=indep_sub_query,
                    is_dependency=False,
                    depth=depth,
                    result=sub_result,
                )
                # Revert to initial result if still unknown
                if sub_result["answer"] == "I don't know.":
                    sub_result = initial_sub_result
            if sub_result["rationale"] == "":
                indep_sub_query["rationale"] = ""
                indep_sub_query["answer"] = ""
            else:
                indep_sub_query["rationale"] = sub_result["rationale"]
                indep_sub_query["answer"] = sub_result["answer"]

    async def process_dependent_subqueries(
        query, dep_sub_queries: list, sub_queries: list, depth: int = 0
    ) -> None:
        for dep_sub_query in dep_sub_queries:
            initial_sub_result = await beam_dep_query(
                query,
                sub_queries,
                dep_sub_query,
                knowledge_graph_inst,
                chunks_vdb,
                text_chunks_db,
                entities_vdb,
                relationships_vdb,
                hyperparams,
                use_model_func,
                rerank_texts_func=rerank_texts_func,
            )
            sub_result = initial_sub_result
            if sub_result["answer"] == "I don't know.":
                sub_result = await process_unknown_subquery(
                    original_query=query,
                    sub_query=dep_sub_query,
                    is_dependency=True,
                    depth=depth,
                    result=sub_result,
                )
                # Revert to initial result if still unknown
                if (
                    "answer" not in sub_result
                    or "answer" not in sub_result["answer"]
                    or sub_result["answer"] == "I don't know."
                ):
                    sub_result = initial_sub_result

            if sub_result["rationale"] == "":
                dep_sub_query["rationale"] = ""
                dep_sub_query["answer"] = ""
            else:
                dep_sub_query["rationale"] = sub_result["rationale"]
                dep_sub_query["answer"] = sub_result["answer"]

    async def process_unknown_subquery(
        original_query: str,
        sub_query: dict,
        is_dependency: bool,
        depth: int = 0,
        max_depth: int = 1,
        result: dict = None,
    ) -> dict:
        """
        Handle sub-queries that returned "I don't know.":
          1. Check if the sub-query is a single-hop question.
          2. If yes, rephrase it more specifically and retry.
          3. If not, decompose further and recurse.
        """
        if depth > max_depth:
            return result

        the_query = sub_query["query"]

        try:
            # Classify whether sub-query is single-hop
            classify_prompt = PROMPTS["classify_subquery"].format(
                original_query=original_query, sub_query=the_query
            )
            classify_response = await use_model_func(prompt=classify_prompt)
            classify_result = await parse_json_response(
                classify_response, use_model_func
            )
            is_single_hop = classify_result.get("is_single_hop", False)
        except Exception as e:
            print(f"Error during subquery classification: {e}")
            is_single_hop = False

        if is_single_hop:
            # Rephrase as a more specific single-hop query and retry
            new_sub_query = classify_result.get("reconstructed query", "")
            sub_result = await beam_indep_query(
                new_sub_query,
                knowledge_graph_inst,
                chunks_vdb,
                text_chunks_db,
                entities_vdb,
                relationships_vdb,
                hyperparams,
                use_model_func,
                rerank_texts_func=rerank_texts_func,
            )
            return sub_result
        else:
            # Decompose further and recurse
            if "sub_queries" in classify_result:
                the_sub_queries = classify_result["sub_queries"]
            else:
                the_sub_queries, reasoning_plan = await decompose_question(
                    the_query, use_model_func
                )
            indep_sub_queries, dep_sub_queries = classify_sub_queries(the_sub_queries)

            await process_independent_subqueries(indep_sub_queries, depth=depth + 1)
            await process_dependent_subqueries(
                the_query, dep_sub_queries, the_sub_queries, depth=depth + 1
            )

            # Aggregate sub-query answers
            sub_queries_content = []
            for idx, sub_query in enumerate(the_sub_queries):
                if sub_query.get("answer", "") == "I don't know.":
                    continue
                sub_query_question = f"Sub query {idx + 1}: {sub_query['query']}"
                sub_query_answer = (
                    f"Answer for  sub query {idx + 1}: {sub_query['answer']}"
                )
                sub_query_rationale = (
                    f"Rationale for sub query {idx + 1}: {sub_query['rationale']}"
                )
                sub_queries_content.append(sub_query_question)
                sub_queries_content.append(sub_query_answer)
                sub_queries_content.append(sub_query_rationale)
            sub_queries_content = "\n".join(sub_queries_content)

            # Retrieve text chunks for the query
            (
                ranked_chunks,
                ranked_entities,
                ranked_relationships,
                chunks,
                entities,
                relationships,
            ) = await fuse_kg_text(
                the_query,
                chunks_vdb,
                text_chunks_db,
                knowledge_graph_inst,
                entities_vdb,
                relationships_vdb,
                use_model_func,
                hyperparams=hyperparams,
            )

            text_chunks = await truncate_by_token_size_async(
                ranked_chunks, key=lambda x: x["content"], max_token_size=max_token_size
            )
            text_chunks = "\n".join([c["content"] for c in text_chunks])
            entity_str = "\n".join([e["description"] for e in ranked_entities[:20]])
            relationship_str = "\n".join([r["description"] for r in relationships[:20]])
            context = entity_str + "\n" + text_chunks + "\n" + relationship_str

            final_answer_prompt = PROMPTS["final_answer_prompt"].format(
                context=context,
                query=the_query,
                sub_queries_and_answers=sub_queries_content,
            )
            final_answer = await use_model_func(prompt=final_answer_prompt)
            final_output = await parse_json_response(final_answer, use_model_func)

            return final_output

    final_output = {}

    use_model_func = global_config["llm_model_func"]
    rerank_texts_func = global_config["rerank_texts_func"]
    config = global_config["config"]

    top_k = config.kg_top_k
    max_token_size = config.kg_max_token_size
    hyperparams = {
        "max_token_size": config.kg_max_token_size,
        "ent_to_chunk_wt": config.ent_to_chunk_wt,
        "rel_to_chunk_wt": config.rel_to_chunk_wt,
        "chunk_to_ent_wt": config.chunk_to_ent_wt,
        "chunk_to_rel_wt": config.chunk_to_rel_wt,
        "etop_k": config.kg_etop_k,
        "rtop_k": config.kg_rtop_k,
        "top_k": config.kg_top_k,
        "beam_search_max_hops": config.beam_search_max_hops,
        "beam_search_beam_width": config.beam_search_beam_width,
        "beam_search_alpha": config.beam_search_alpha,
        "beam_search_beta": config.beam_search_beta,
        "beam_search_decay": config.beam_search_decay,
    }

    # Step 1: Decompose the query
    try:
        sub_queries, reasoning_plan = await decompose_question(query, use_model_func)
    except Exception as decompose_err:
        return {"error": str(decompose_err)}
    sub_query_copy = copy.deepcopy(sub_queries)

    # Step 2: Classify sub-queries into independent and dependent
    independent_sub_queries, dependent_sub_queries = classify_sub_queries(
        sub_query_copy
    )

    # Step 3: Process independent first, then dependent (which need independent results)
    await process_independent_subqueries(independent_sub_queries)

    await process_dependent_subqueries(query, dependent_sub_queries, sub_query_copy)

    # Step 4: Aggregate sub-query answers
    sub_queries_content = []
    for idx, sub_query in enumerate(sub_query_copy):
        if sub_query.get("answer", "") == "I don't know.":
            continue
        sub_query_question = f"Sub query {idx + 1}: {sub_query['query']}"
        sub_query_answer = f"Answer for  sub query {idx + 1}: {sub_query['answer']}"
        sub_query_rationale = (
            f"Rationale for sub query {idx + 1}: {sub_query['rationale']}"
        )
        sub_queries_content.append(sub_query_question)
        sub_queries_content.append(sub_query_answer)
        sub_queries_content.append(sub_query_rationale)
    sub_queries_content = "\n".join(sub_queries_content)

    # Step 5: Retrieve text chunks and run beam search
    (
        ranked_chunks,
        ranked_entities,
        ranked_relationships,
        chunks,
        entities,
        relationships,
    ) = await fuse_kg_text(
        query,
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

    # Multi-hop beam search exploration
    beam = await beam_search(
        query,
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

    # Process beam search output into formatted context
    _, _, _, context = await process_beam_output(
        beam,
        ranked_entities,
        ranked_relationships,
        ranked_chunks,
        max_hops,
        max_entities_tokens=config.kg_max_token_size_for_beam_search,
        max_relationships_tokens=config.kg_max_token_size_for_beam_search,
        max_chunks_tokens=config.kg_max_token_size_for_beam_search,
    )

    # Step 6: Generate final answer
    final_answer_prompt = PROMPTS["final_answer_prompt"].format(
        context=context, query=query, sub_queries_and_answers=sub_queries_content
    )
    final_answer = await use_model_func(prompt=final_answer_prompt)
    final_output = await parse_json_response(final_answer, use_model_func)

    return final_output


async def SGN_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    global_config: dict,
):
    use_model_func = global_config["llm_model_func"]
    rerank_texts_func = global_config["rerank_texts_func"]
    config = global_config["config"]

    hyperparams = {
        "max_token_size": config.kg_max_token_size_for_beam_search,
        "ent_to_chunk_wt": config.ent_to_chunk_wt,
        "rel_to_chunk_wt": config.rel_to_chunk_wt,
        "chunk_to_ent_wt": config.chunk_to_ent_wt,
        "chunk_to_rel_wt": config.chunk_to_rel_wt,
        "etop_k": config.kg_etop_k,
        "rtop_k": config.kg_rtop_k,
        "top_k": config.kg_top_k,
        "beam_search_max_hops": config.beam_search_max_hops,
        "beam_search_beam_width": config.beam_search_beam_width,
        "beam_search_alpha": config.beam_search_alpha,
        "beam_search_beta": config.beam_search_beta,
        "beam_search_decay": config.beam_search_decay,
    }
    # Fuse retrieval for entities, relationships, and text chunks
    (
        ranked_chunks,
        ranked_entities,
        ranked_relationships,
        chunks,
        entities,
        relationships,
    ) = await fuse_kg_text(
        query,
        chunks_vdb,
        text_chunks_db,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        use_model_func,
        hyperparams=hyperparams,
    )

    max_hops = config.beam_search_max_hops
    beam_width = config.beam_search_beam_width
    alpha = config.beam_search_alpha
    beta = config.beam_search_beta
    decay = config.beam_search_decay
    beam = await beam_search(
        query,
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
    # Process beam search output
    (
        final_entities,
        final_relationships,
        final_chunks,
        context,
    ) = await process_beam_output(
        beam=beam,
        entities=ranked_entities,
        relationships=ranked_relationships,
        text_chunks=ranked_chunks,
        max_hops=max_hops,
        max_entities_tokens=config.kg_max_token_size_for_beam_search,
        max_relationships_tokens=config.kg_max_token_size_for_beam_search,
        max_chunks_tokens=config.kg_max_token_size_for_beam_search,
    )
    rag_prompt = PROMPTS["graphnavrag_prompt"].format(query=query, context=context)
    result = await use_model_func(prompt=rag_prompt)
    result = await parse_json_response(result, use_model_func)
    return result
