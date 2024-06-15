import os
from typing import AsyncGenerator
from urllib.parse import urljoin
import httpx
from loguru import logger
from openai import AsyncOpenAI

API_KEY = os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ["OPENAI_BASE_URL"]
TIMEOUT = 30


async def completions_stream(
        data
) -> AsyncGenerator[str, None]:
    async with httpx.AsyncClient() as client:
        async with client.stream(
                "POST",
                urljoin(BASE_URL + "/", "chat/completions"),
                timeout=httpx.Timeout(TIMEOUT),
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                },
                json=data,
        ) as response:
            print(f"received response status_code={response.status_code}")
            response.raise_for_status()
            async for chunk in response.aiter_text():
                logger.debug(f"received chunk: {chunk}")
                yield chunk


async def completions_stream1(data: dict):
    client = AsyncOpenAI()

    try:
        stream = await client.chat.completions.create(**data)
        async for chunk in stream:
            logger.debug(f"received chunk: {chunk}")
            # print(chunk.choices[0].delta.content or "", end="")
            yield chunk.json()
    finally:
        await client.close()
