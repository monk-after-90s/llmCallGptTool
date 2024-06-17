import os
from typing import AsyncGenerator, Dict
from urllib.parse import urljoin
import httpx
from loguru import logger
from urllib.parse import urlparse

API_KEY = os.environ.get("OPENAI_API_KEY")
# 解析出host
parsed_url = urlparse(os.environ['OPENAI_BASE_URL'])
BASE_URL = f"{parsed_url.scheme}://{parsed_url.netloc}"

TIMEOUT = 30


async def openai_stream(data: Dict, method: str = "POST", path: str = "") -> AsyncGenerator[str, None]:
    if method != "POST":
        raise NotImplementedError

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
                logger.debug(f"received chunk: {chunk}")
                yield chunk
