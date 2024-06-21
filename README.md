# llmCallGptTool: Enable any LLM to call tool in gpt format.

Make any LLM call gpt tool, which means the
openai [function calling doc](https://platform.openai.com/docs/guides/function-calling)
can be applied to any LLM through the api served by this project.

This project was initially created for Qwen2. But it might be applied to any LLM in theory. Welcome to contribute.

## Install

Python 3.11

```bash
pip install -r requirements.txt
```

## Run

```bash
OPENAI_API_KEY={OPENAI_API_KEY} OPENAI_BASE_URL={OPENAI_BASE_URL} uvicorn main:app [--workers {worker_num}] [--port {port}]
```

The `OPENAI_API_KEY` and `OPENAI_BASE_URL` are the environment variables for the OpenAI compatible API key and base URL(
end with `/v1`)
of the LLM. worker_num is the number of workers, port is the port of the server. worker_num and port are optional.

The API served by this project is compatible with the OpenAI API.

## Citation
```citation
@article{llmCallGptTool,
title={llmCallGptTool: Enable any LLM to call tool in gpt format},
author={monk-after-90s },
journal={},
year={2024}}
```