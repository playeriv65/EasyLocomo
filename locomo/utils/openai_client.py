import os
import sys
import time
import json
import numpy as np
from openai import OpenAI, APIError, APIConnectionError, RateLimitError

def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_API_BASE")
    return OpenAI(api_key=api_key, base_url=base_url)

def set_openai_key():
    # In v1, we don't set global keys, but we can check if env vars are set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set in environment variables.")

def run_chatgpt(query, num_gen=1, num_tokens_request=1000, 
                model="gpt-3.5-turbo", use_16k=False, temperature=1.0, wait_time=1):
    
    client = get_openai_client()
    
    # Map legacy model names if necessary, or just use what's passed
    if model == "chatgpt":
        model = "gpt-3.5-turbo"
    
    messages = [{"role": "user", "content": query}]
    
    completion = None
    while completion is None:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=num_tokens_request,
                n=num_gen
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
        return completion.choices[0].message.content

def run_chatgpt_with_examples(query, examples, input_text, num_gen=1, num_tokens_request=1000, use_16k=False, wait_time=1, temperature=1.0):
    client = get_openai_client()
    
    messages = [{"role": "system", "content": query}]
    for inp, out in examples:
        messages.append({"role": "user", "content": inp})
        messages.append({"role": "assistant", "content": out}) # Changed system to assistant for examples
    messages.append({"role": "user", "content": input_text})

    completion = None
    while completion is None:
        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo" if not use_16k else "gpt-3.5-turbo-16k",
                temperature=temperature,
                max_tokens=num_tokens_request,
                n=num_gen,
                messages=messages
            )
        except Exception as e:
            print(f"Error: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
            wait_time *= 2
            
    return completion.choices[0].message.content

def run_json_trials(query, num_gen=1, num_tokens_request=1000, 
                model="gpt-3.5-turbo", use_16k=False, temperature=1.0, wait_time=1, examples=None, input=None):

    run_loop = True
    counter = 0
    while run_loop:
        try:
            if examples is not None and input is not None:
                output = run_chatgpt_with_examples(query, examples, input, num_gen=num_gen, wait_time=wait_time,
                                                   num_tokens_request=num_tokens_request, use_16k=use_16k, temperature=temperature).strip()
            else:
                output = run_chatgpt(query, num_gen=num_gen, wait_time=wait_time, model=model,
                                                   num_tokens_request=num_tokens_request, use_16k=use_16k, temperature=temperature)
            
            # Clean up potential markdown code blocks
            if "```" in output:
                import re
                json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", output, re.DOTALL)
                if json_match:
                    output = json_match.group(1)
                else:
                    output = output.replace("```json", "").replace("```", "")

            output = output.replace("json", "") 
            facts = json.loads(output.strip())
            run_loop = False
        except json.decoder.JSONDecodeError:
            counter += 1
            time.sleep(1)
            print("Retrying to avoid JsonDecodeError, trial %s ..." % counter)
            # print(output)
            if counter == 10:
                print("Exiting after 10 trials")
                sys.exit()
            continue
    return facts
