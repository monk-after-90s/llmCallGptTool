import os
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import httpx
from loguru import logger
from urllib.parse import urlparse
from fastapi.middleware.cors import CORSMiddleware
from utilities.openai_tool import completions_stream

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有HTTP头
)

TARGET_URL = os.environ['OPENAI_BASE_URL']

# 全局唯一的 httpx.AsyncClient 实例
client: None | httpx.AsyncClient = None


@app.on_event("startup")
async def startup_event():
    global client
    client = httpx.AsyncClient()
    logger.info("HTTP client initialized")


@app.on_event("shutdown")
async def shutdown_event():
    global client
    await client.aclose()
    logger.info("HTTP client closed")


@app.middleware("http")
async def proxy_middleware(request: Request, call_next):
    # 构建目标URL
    url = f"{TARGET_URL}{request.url.path}"
    logger.debug(f"target url: {url}")
    # 获取请求方法
    method = request.method

    # 获取请求头
    headers = dict(request.headers)
    parsed_url = urlparse(url)
    host = parsed_url.netloc
    headers["host"] = headers["x-forwarded-host"] = host

    if '/v1/chat/completions' == request.url.path and method == "POST":
        data = await request.json()
        logger.debug(f"{data=}")

        resp = StreamingResponse(
            completions_stream(data=data),
            media_type="text/event-stream")
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp
    else:
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
