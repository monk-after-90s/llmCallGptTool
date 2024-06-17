"""
qwen2的agent版，支持高级功能比如tool calling
"""
import json
from typing import List, Dict

from qwen_agent.llm import get_chat_model, BaseChatModel


# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit='fahrenheit'):
    """Get the current weather in a given location"""
    if 'tokyo' in location.lower():
        return json.dumps({
            'location': 'Tokyo',
            'temperature': '10',
            'unit': 'celsius'
        })
    elif 'san francisco' in location.lower():
        return json.dumps({
            'location': 'San Francisco',
            'temperature': '72',
            'unit': 'fahrenheit'
        })
    elif 'paris' in location.lower():
        return json.dumps({
            'location': 'Paris',
            'temperature': '22',
            'unit': 'celsius'
        })
    else:
        return json.dumps({'location': location, 'temperature': 'unknown'})


def qwen2_call_tool(llm: BaseChatModel, messages: List[Dict], functions: List[Dict], stream: bool = True):
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

    function_choices = llm.chat(messages=messages,
                                functions=functions,
                                stream=stream)
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
