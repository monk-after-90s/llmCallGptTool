"""
qwen2的agent版，支持高级功能比如tool calling
"""
import copy
import json
from typing import List, Dict, Iterator, Union, Optional, Literal

from qwen_agent.llm.base import _truncate_input_messages_roughly, retry_model_service_iterator, retry_model_service
from qwen_agent.log import logger
from openai import OpenAIError
from qwen_agent.llm import get_chat_model, BaseChatModel, TextChatAtOAI, ModelServiceError
from qwen_agent.llm.schema import Message, SYSTEM, DEFAULT_SYSTEM_MESSAGE
from pprint import pformat

from qwen_agent.settings import DEFAULT_MAX_INPUT_TOKENS
from qwen_agent.utils.utils import merge_generate_cfgs, has_chinese_messages


class TextChatChunkEnabled(TextChatAtOAI):
    """流式响应OpenAI api chunk的BaseTextChatModel继承类"""

    def _chat_stream(
            self,
            messages: List[Message],
            delta_stream: bool,
            generate_cfg: dict,
    ) -> Iterator[List[Message]]:
        messages = [msg.model_dump() for msg in messages]
        logger.debug(f'*{pformat(messages, indent=2)}*')
        try:
            response = self._chat_complete_create(model=self.model, messages=messages, stream=True, **generate_cfg)
            for chunk in response:
                if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                    yield chunk.to_dict()
        except OpenAIError as ex:
            raise ModelServiceError(exception=ex)

    def _postprocess_messages_iterator(
            self,
            messages: Iterator[List[Message]],
            fncall_mode: bool,
            generate_cfg: dict,
    ) -> Iterator[List[Message]]:
        for m in messages:
            if m:
                yield m

    # def _convert_messages_iterator_to_target_type(
    #         self, messages_iter: Iterator[List[Message]],
    #         target_type: str) -> Union[Iterator[List[Message]], Iterator[List[Dict]]]:
    #     for messages in messages_iter:
    #         yield messages

    def _chat_no_stream(
            self,
            messages: List[Message],
            generate_cfg: dict,
    ) -> List[Message]:
        messages = [msg.model_dump() for msg in messages]
        logger.debug(f'*{pformat(messages, indent=2)}*')
        try:
            response = self._chat_complete_create(model=self.model, messages=messages, stream=False, **generate_cfg)
            return response.to_dict()
        except OpenAIError as ex:
            raise ModelServiceError(exception=ex)

    def chat(
            self,
            messages: List[Union[Message, Dict]],
            functions: Optional[List[Dict]] = None,
            stream: bool = True,
            delta_stream: bool = False,
            extra_generate_cfg: Optional[Dict] = None,
    ) -> Union[Dict, Iterator[List[Message]], Iterator[List[Dict]]]:
        """LLM chat interface.

        Args:
            messages: Inputted messages.
            functions: Inputted functions for function calling. OpenAI format supported.
            stream: Whether to use streaming generation.
            delta_stream: Whether to stream the response incrementally.
              (1) When False (recommended): Stream the full response every iteration.
              (2) When True: Stream the chunked response, i.e, delta responses.
            extra_generate_cfg: Extra LLM generation hyper-paramters.

        Returns:
            the generated message list response by llm.
        """

        generate_cfg = merge_generate_cfgs(base_generate_cfg=self.generate_cfg, new_generate_cfg=extra_generate_cfg)
        if 'lang' in generate_cfg:
            lang: Literal['en', 'zh'] = generate_cfg.pop('lang')
        else:
            lang: Literal['en', 'zh'] = 'zh' if has_chinese_messages(messages) else 'en'

        messages = copy.deepcopy(messages)

        _return_message_type = 'dict'
        new_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                new_messages.append(Message(**msg))
            else:
                new_messages.append(msg)
                _return_message_type = 'message'
        messages = new_messages

        if messages[0].role != SYSTEM:
            messages = [Message(role=SYSTEM, content=DEFAULT_SYSTEM_MESSAGE)] + messages

        # Not precise. It's hard to estimate tokens related with function calling and multimodal items.
        messages = _truncate_input_messages_roughly(
            messages=messages,
            max_tokens=generate_cfg.pop('max_input_tokens', DEFAULT_MAX_INPUT_TOKENS),
        )

        messages = self._preprocess_messages(messages, lang=lang)

        if functions:
            fncall_mode = True
        else:
            fncall_mode = False

        def _call_model_service():
            if fncall_mode:
                return self._chat_with_functions(
                    messages=messages,
                    functions=functions,
                    stream=stream,
                    delta_stream=delta_stream,
                    generate_cfg=generate_cfg,
                    lang=lang,
                )
            else:
                return self._chat(
                    messages,
                    stream=stream,
                    delta_stream=delta_stream,
                    generate_cfg=generate_cfg,
                )

        if stream and delta_stream:
            # No retry for delta streaming
            output = _call_model_service()
        elif stream and (not delta_stream):
            output = retry_model_service_iterator(_call_model_service, max_retries=self.max_retries)
        else:
            output = retry_model_service(_call_model_service, max_retries=self.max_retries)

        if isinstance(output, Dict):
            # 非流式响应字典
            return output
        else:
            # 流式响应chunk流
            output = self._postprocess_messages_iterator(output, fncall_mode=fncall_mode, generate_cfg=generate_cfg)
            return output


def qwen2_call_tool(llm_cfg: Dict, messages: List[Dict], functions: List[Dict], stream: bool = True):
    """
    让qwen决定是否调用tool

    functions: 对应OpenAI API中的tools。目前只支持function，例如：
    [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    """
    # messages格式不用转换
    messages = messages
    # functions格式需要转换
    functions = [f['function'] for f in functions if f['type'] == 'function']

    llm = TextChatChunkEnabled(llm_cfg)
    function_choices = llm.chat(messages=messages,
                                functions=functions,
                                stream=stream)
    if not stream:
        return function_choices


def test():
    llm = get_chat_model({
        'model': 'Qwen/Qwen2-72B-Instruct-GPTQ-Int4',
        'model_server': 'https://oneapi.excn.top/v1',  # api_base
        'api_key': 'sk-BdEz66b0Aj1Qfm0sA45c0e5097F8453dA9B6696918D54702',
    })

    # Step 1: send the conversation and available functions to the model
    messages = [{
        'role': 'user',
        'content': "What's the weather like in San Francisco?"
    }]
    functions = [{
        'name': 'get_current_weather',
        'description': 'Get the current weather in a given location',
        'parameters': {
            'type': 'object',
            'properties': {
                'location': {
                    'type': 'string',
                    'description':
                        'The city and state, e.g. San Francisco, CA',
                },
                'unit': {
                    'type': 'string',
                    'enum': ['celsius', 'fahrenheit']
                },
            },
            'required': ['location'],
        },
    }]

    print('# Assistant Response 1:')
    responses = []
    for responses in llm.chat(messages=messages,
                              functions=functions,
                              stream=True):
        print(responses)

    messages.extend(responses)  # extend conversation with assistant's reply

    # Step 2: check if the model wanted to call a function
    last_response = messages[-1]
    if last_response.get('function_call', None):

        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            'get_current_weather': get_current_weather,
        }  # only one function in this example, but you can have multiple
        function_name = last_response['function_call']['name']
        function_to_call = available_functions[function_name]
        function_args = json.loads(last_response['function_call']['arguments'])
        function_response = function_to_call(
            location=function_args.get('location'),
            unit=function_args.get('unit'),
        )
        print('# Function Response:')
        print(function_response)

        # Step 4: send the info for each function call and function response to the model
        messages.append({
            'role': 'function',
            'name': function_name,
            'content': function_response,
        })  # extend conversation with function response

        print('# Assistant Response 2:')
        for responses in llm.chat(
                messages=messages,
                functions=functions,
                stream=True,
        ):  # get a new response from the model where it can see the function response
            print(responses)


if __name__ == '__main__':
    test()
