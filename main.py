from fastapi import FastAPI, Request, logger
from fastapi.responses import StreamingResponse
import httpx
from loguru import logger
from urllib.parse import urlparse

app = FastAPI()

TARGET_URL = "https://qwen2-72b-instruct-gptq-int4.excn.top"


@app.middleware("http")
async def proxy_middleware(request: Request, call_next):
    logger.debug(f"request: {request.url}")
    async with httpx.AsyncClient() as client:
        # 构建目标URL
        url = f"{TARGET_URL}{request.url.path}"

        # 获取请求方法
        method = request.method

        # 获取请求头
        headers = dict(request.headers)
        parsed_url = urlparse(url)
        host = parsed_url.netloc
        headers["host"] = headers["x-forwarded-host"] = host
        # 获取请求体
        body = await request.body()

        # 发送请求到目标服务
        response = await client.request(
            method=method,
            url=url,
            headers=headers,
            content=body,
            params=request.query_params
        )

        # 构建响应
        return StreamingResponse(response.aiter_bytes(), status_code=response.status_code, headers=response.headers)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
