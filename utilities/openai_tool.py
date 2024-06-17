import os
from typing import AsyncGenerator, Dict
from urllib.parse import urljoin
import httpx
from loguru import logger
from urllib.parse import urlparse
from openai import AsyncOpenAI

API_KEY = os.environ.get("OPENAI_API_KEY")
# 解析出host
parsed_url = urlparse(os.environ['OPENAI_BASE_URL'])
BASE_URL = f"{parsed_url.scheme}://{parsed_url.netloc}"

TIMEOUT = 30

client: None | AsyncOpenAI = None


async def openai_stream(data: Dict, method: str = "POST", path: str = "", channel: str = "openai"):
    """根据是否流式选择"""
    if not data.get("stream"):
        async for chat_completion in _openai_stream(data, method, path, channel):
            return chat_completion.to_dict()
    else:
        return _openai_stream(data, method, path, channel)


async def _openai_stream(data: Dict, method: str = "POST", path: str = "", channel: str = "openai") \
        -> AsyncGenerator[str, None]:
    global client

    if method != "POST":
        raise NotImplementedError
    if channel == "httpx":
        async with httpx.AsyncClient() as client:
            async with client.stream(
                    method,
                    urljoin(BASE_URL, path),
                    timeout=httpx.Timeout(TIMEOUT),
                    headers={
                        "Authorization": f"Bearer {API_KEY}",
                    },
                    json=data,
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_text():
                    c_cache = ''
                    for c in chunk:
                        c_cache += c
                        if c_cache.endswith("\n\n"):
                            logger.debug(f"{c_cache=}")
                            yield c_cache
                            c_cache = ''
                    else:
                        if c_cache:
                            yield c_cache
    elif channel == "openai" and path == "/v1/chat/completions":
        client = client or AsyncOpenAI()

        if not data.get("stream"):
            yield await client.chat.completions.create(**data)
            return

        stream = await client.chat.completions.create(**data)
        async for chunk in stream:
            chunk_s = "data: " + chunk.to_json(indent=0).replace("\n", "") + "\n\n"
            logger.debug(f"{chunk_s=}")
            yield chunk_s
    else:
        raise NotImplementedError
