import os
import sys
import time
import json
import numpy as np
from openai import OpenAI, AsyncOpenAI, APIError, APIConnectionError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from locomo.config import config

def get_openai_client():
    client = OpenAI(
        api_key=config.openai_api_key, 
        base_url=config.openai_api_base,
    )
    return client

def get_async_openai_client():
    client = AsyncOpenAI(
        api_key=config.openai_api_key, 
        base_url=config.openai_api_base,
    )
    return client

# Synchronous version (keeping for backward compatibility if needed, though we should migrate away)
def run_chatgpt(query, num_gen=1, num_tokens_request=1000, temperature=0, wait_time=1, response_format=None, model=None):
    if model is None:
        assert config.model_name is not None, "model_name must be set in config before calling run_chatgpt"
        model = config.model_name
    
    client = get_openai_client()
    
    messages = [{"role": "user", "content": query}]
    
    completion = None
    while completion is None:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=num_tokens_request,
                n=num_gen,
                response_format=response_format
            )
        except RateLimitError as e:
            print(f"OpenAI API request exceeded rate limit: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
            wait_time *= 2
        except APIConnectionError as e:
            print(f"Failed to connect to OpenAI API: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
            wait_time *= 2
        except APIError as e:
            print(f"OpenAI API returned an API Error: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
            wait_time *= 2
        except Exception as e:
            print(f"An unexpected error occurred: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
            wait_time *= 2

    if num_gen > 1:
        return [choice.message.content for choice in completion.choices]
    else:
        content = completion.choices[0].message.content
        return content if content is not None else ""

@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIError)),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(10)
)
async def run_chatgpt_async(query, num_gen=1, num_tokens_request=1000, temperature=0, response_format=None, model=None):
    if model is None:
        assert config.model_name is not None, "model_name must be set in config before calling run_chatgpt_async"
        model = config.model_name
    
    client = get_async_openai_client()
    messages = [{"role": "user", "content": query}]
    
    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=num_tokens_request,
            n=num_gen,
            response_format=response_format
        )
    except Exception as e:
        # print(f"Async Request Error: {e}") # Tenacity will handle logging if configured, or we can just let it bubble up to retry
        raise e

    if num_gen > 1:
        return [choice.message.content for choice in completion.choices]
    else:
        content = completion.choices[0].message.content
        return content if content is not None else ""
