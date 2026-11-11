import os
import asyncio
import itertools
import numpy as np
from typing import List
import httpx
from openai import AsyncOpenAI, APIConnectionError, RateLimitError, Timeout
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.clients import httpx_client, openai_client, openai_embedding_client
from src.config import settings
from src.base import BaseKVStorage
from src.utils import compute_args_hash, wrap_embedding_func_with_attrs

os.environ["TOKENIZERS_PARALLELISM"] = "false"

@retry(
    stop=stop_after_attempt(12),
    wait=wait_exponential(multiplier=2, min=8, max=60),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def openai_complete_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    base_url=None,
    api_key=None,
    temperature=0.7,
    max_tokens=None,
    top_p=None,
    presence_penalty=None,
    extra_body=None,
    openai_client: AsyncOpenAI = openai_client,
    **kwargs,
) -> str:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    api_params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    # Add optional params
    if max_tokens is not None:
        api_params["max_tokens"] = max_tokens
    if top_p is not None:
        api_params["top_p"] = top_p
    if presence_penalty is not None:
        api_params["presence_penalty"] = presence_penalty
    if extra_body is not None:
        api_params["extra_body"] = extra_body

    api_params.update(kwargs)

    # Check cache
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await openai_client.chat.completions.create(**api_params)

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": model}}
        )
    return response.choices[0].message.content


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(12),
    wait=wait_exponential(multiplier=2, min=8, max=60),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, Timeout)
    ),
)
async def openai_embedding(
    texts: list[str],
    model: str = "text-embedding-3-small",
    base_url: str = None,
    api_key: str = None,
    client: AsyncOpenAI = openai_embedding_client,
    **kwargs,
) -> np.ndarray:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    hashing_kv: BaseKVStorage = kwargs.pop("embedding_cache", None)
    # Check cache
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, texts)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return np.array(if_cache_return["return"])

    response = await client.embeddings.create(
        model=model, input=texts, encoding_format="float"
    )

    result = np.array([dp.embedding for dp in response.data])

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": result.tolist(), "model": model}}
        )
    return result


async def rerank_texts(
    query: str,
    texts: List[str],
    client: httpx.AsyncClient = httpx_client,
    model: str = settings.rerank_model_name,
    base_url: str = settings.rerank_base_url,
    max_retries: int = settings.rerank_max_retries,
    max_chunks_per_doc: int = settings.rerank_max_chunks_per_doc,
    overlap_tokens: int = settings.rerank_overlap_tokens,
    **kwargs,
) -> List[dict]:
    """Rerank texts using a rerank model. Returns sorted results with relevance_score and index."""

    # Check cache
    rerank_cache: BaseKVStorage = kwargs.pop("rerank_cache", None)
    if rerank_cache is not None:
        cache_key_data = {
            "query": query,
            "texts": texts,
            "model": model,
            "max_chunks_per_doc": max_chunks_per_doc,
            "overlap_tokens": overlap_tokens,
        }
        args_hash = compute_args_hash("rerank", cache_key_data)
        if_cache_return = await rerank_cache.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    payload = {
        "model": model,
        "query": query,
        "documents": texts,
        "top_n": len(texts),
        "return_documents": False,
        "max_chunks_per_doc": max_chunks_per_doc,
        "overlap_tokens": overlap_tokens,
    }

    for attempt in range(max_retries):
        try:
            headers = {
                "Authorization": f"Bearer {settings.rerank_api_key}",
                "Content-Type": "application/json",
            }
            response = await client.post(base_url, json=payload, headers=headers)
            response.raise_for_status()

            result = response.json()
            if "results" not in result:
                raise ValueError("API response missing 'results' field")

            result = sorted(result["results"], key=lambda x: x["index"])

            # Store to cache
            if rerank_cache is not None:
                await rerank_cache.upsert(
                    {args_hash: {"return": result, "model": model}}
                )

            return result

        except Exception as e:
            # Exponential backoff, max 60s
            wait_time = min(2**attempt, 60)
            # Fixed wait for 429 rate limit
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 429:
                wait_time = 45
                print(
                    f"[Retry {attempt + 1}/{max_retries}] 429 Too Many Requests, waiting {wait_time}s..."
                )
            else:
                print(
                    f"[Retry {attempt + 1}/{max_retries}] {type(e).__name__}: {str(e)}, waiting {wait_time}s..."
                )
                print(f"Query: {query}")

            if attempt == max_retries - 1:
                print("Max retries reached, returning default scores.")
                default_result = [
                    {"relevance_score": 0.5, "index": i} for i in range(len(texts))
                ]
                # Cache default result
                if rerank_cache is not None:
                    await rerank_cache.upsert(
                        {args_hash: {"return": default_result, "model": model}}
                    )
                return default_result

            await asyncio.sleep(wait_time)
