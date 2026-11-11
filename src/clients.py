import httpx
from httpx import Limits
from openai import AsyncOpenAI
from src.config import settings


def create_openai_client(cfg=settings):
    return AsyncOpenAI(
        base_url=cfg.llm_base_url,
        api_key=cfg.llm_api_key,
        timeout=cfg.llm_timeout,
    )

def create_embedding_client(cfg=settings):
    return AsyncOpenAI(
        base_url=cfg.effective_embedding_base_url,
        api_key=cfg.effective_embedding_api_key,
        timeout=cfg.embedding_timeout,
    )

def create_httpx_client(cfg=settings):
    return httpx.AsyncClient(
        timeout=httpx.Timeout(cfg.httpx_timeout),
        limits=httpx.Limits(
            max_connections=cfg.httpx_max_connections, 
            max_keepalive_connections=cfg.httpx_max_keepalive_connections
        ),
    )

openai_client = create_openai_client()
openai_embedding_client = create_embedding_client()
httpx_client = create_httpx_client()

