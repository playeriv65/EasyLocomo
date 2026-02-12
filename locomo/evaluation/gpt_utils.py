import random
import os, json, hashlib
import asyncio
from tqdm.asyncio import tqdm_asyncio
import time
from locomo.utils.openai_client import run_chatgpt, run_chatgpt_async
import tiktoken

PER_QA_TOKEN_BUDGET = 50

QA_PROMPT = """
Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {} Short answer:
"""

QA_PROMPT_CAT_5 = """
Based on the above context, answer the following question.

Question: {} Short answer:
"""

# CONV_START_PROMPT = "Below is a conversation between two people: {} and {}. The conversation takes place over multiple days and the date of each conversation is wriiten at the beginning of the conversation.\n\n"
CONV_START_PROMPT = "Below is a conversation between two people: {} and {}. The conversation takes place over multiple days, presented in chronological order.\n\n"


def get_cat_5_answer(model_prediction, answer_key):
    model_prediction = model_prediction.strip().lower()
    if len(model_prediction) == 1:
        if 'a' in model_prediction:
            return answer_key['a']
        else:
            return answer_key['b']
    elif len(model_prediction) == 3:
        if '(a)' in model_prediction:
            return answer_key['a']
        else:
            return answer_key['b']
    else:
        return model_prediction


def get_input_context(data, num_question_tokens, encoding, args):
    """
    Constructs the conversation context in CHRONOLOGICAL order.
    Most recent sessions are appended to the end.
    """
    query_conv = ''
    
    # 1. Identify and sort sessions chronologically (Session 1 -> Session N)
    session_nums = [int(k.split('_')[-1]) for k in data.keys() if 'session' in k and 'date_time' not in k]
    sorted_sessions = sorted(session_nums) # Default ascending order: 1, 2, 3...
    
    # We build the full context first, then truncate from the beginning if it helps fit the window? 
    # Or should we just take the last N sessions that fit?
    # Standard practice for long context is usually to keep the most recent history that fits.
    
    full_conv_text = ""
    
    for i in sorted_sessions:
        if 'session_%s' % i in data:
            session_text = ""
            session_date = data.get('session_%s_date_time' % i, "Unknown Date")
            session_text += f"DATE: {session_date}\nCONVERSATION:\n"
            
            # Dialogs in json are usually list of turns. We assume they are chronological in the list?
            # The original code did `for dialog in data['session_%s' % i][::-1]:` which implies the list was newest-first?
            # Let's inspect a sample if possible, but standard is list order = time order.
            # If the original code reversed it to prepend, then the original list might have been chronological.
            # Let's assume the list in JSON is chronological (Turn 1, Turn 2...).
            
            for dialog in data['session_%s' % i]:
                turn = dialog['speaker'] + ' said, \"' + dialog['text'] + '\"' + '\n'
                if "blip_caption" in dialog:
                    turn += ' and shared %s.' % dialog["blip_caption"]
                turn += '\n'
                session_text += turn
            
            session_text += "\n"
            full_conv_text += session_text

    # Check token length
    full_tokens = encoding.encode(full_conv_text)
    max_avail = args.max_context - num_question_tokens - 100 # Buffer
    
    if len(full_tokens) <= max_avail:
        return full_conv_text
    else:
        # Truncate from the beginning (keep recent history)
        # Decode properly to avoid splitting characters
        truncated_tokens = full_tokens[-max_avail:]
        return encoding.decode(truncated_tokens)


# Standardize question token budget to ensure identical context across prompts for caching.
# 1024 is a safe upper bound for most QA prompts in LoCoMo.
RESERVED_QA_TOKENS = 1024

async def process_single_question(qa, include_idx, in_data, start_prompt, start_tokens, encoding, args, prediction_key):
    """
    Async worker for a single question.
    """
    
    # 1. Prepare Question
    question_text = qa['question']
    is_cat_5 = False
    cat_5_options = None
    
    if qa.get('category') == 2:
        question_text += ' Use DATE of CONVERSATION to answer with an approximate date.'
    elif qa.get('category') == 5:
        is_cat_5 = True
        ans_text = qa.get('answer', qa.get('adversarial_answer', ''))
        q_template = qa['question'] + " Select the correct answer: (a) {} (b) {}. "
        
        # Deterministic choice based on sample_id and question text
        # This ensures the same question in the same sample always has the same options order
        stable_hash = hashlib.sha256(f"{in_data.get('sample_id', '')}_{qa['question']}".encode()).hexdigest()
        is_swapped = int(stable_hash[0], 16) % 2 == 0
        
        if is_swapped:
            question_text = q_template.format('Not mentioned in the conversation', ans_text)
            cat_5_options = {'a': 'Not mentioned in the conversation', 'b': ans_text}
        else:
            question_text = q_template.format(ans_text, 'Not mentioned in the conversation')
            cat_5_options = {'b': 'Not mentioned in the conversation', 'a': ans_text}
            
    # 2. Build Prompt Parts
    final_prompt_template = QA_PROMPT if not is_cat_5 else QA_PROMPT_CAT_5
    prompt_suffix = final_prompt_template.format(question_text)
    
    # 3. Get Context (Chronological & Deterministic Prefix)
    # We use a FIXED RESERVED_QA_TOKENS instead of dynamic num_q_tokens.
    # This ensures that for all questions in the same sample, the 'context' text is identical,
    # enabling bit-for-bit identical prefixes for efficient provider-side prompt caching.
    context = get_input_context(in_data['conversation'], RESERVED_QA_TOKENS + start_tokens, encoding, args)
    full_query = start_prompt + context + "\n\n" + prompt_suffix
    
    # 4. Call LLM (Async)
    try:
        # Use a reasonable token limit for a single answer
        answer = await run_chatgpt_async(
            full_query, 
            num_gen=1, 
            num_tokens_request=128, 
            temperature=0
        )
        
        result = answer.strip()
        
        if is_cat_5:
            result = get_cat_5_answer(result, cat_5_options)
        
        return include_idx, result, None
        
    except Exception as e:
        return include_idx, None, str(e)


async def get_gpt_answers_async(in_data, out_data, prediction_key, args, out_samples=None, out_file=None):
    try:
        encoding = tiktoken.encoding_for_model(args.model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")
        
    start_prompt = CONV_START_PROMPT.format(in_data['person1'], in_data['person2'])
    start_tokens = len(encoding.encode(start_prompt))

    qa_list = in_data['qa']
    if args.max_questions:
        qa_list = qa_list[:args.max_questions]
        
    tasks = []
    
    # Semaphore to control concurrency
    # batch_size now strictly controls the number of concurrent requests.
    # If batch_size=1, it will be serial. If batch_size=20, up to 20 requests in parallel.
    sem = asyncio.Semaphore(args.batch_size)

    async def sem_task(task_func):
        async with sem:
            return await task_func

    for i, qa in enumerate(qa_list):
        # Filter by category
        if getattr(args, 'category', None) is not None and qa.get('category') != args.category:
            continue

        if prediction_key not in out_data['qa'][i] or args.overwrite:
            task = process_single_question(qa, i, in_data, start_prompt, start_tokens, encoding, args, prediction_key)
            tasks.append(sem_task(task))
            
    if not tasks:
        return out_data

    # Run all tasks with progress bar
    results = await tqdm_asyncio.gather(*tasks, desc=f"Processing {len(tasks)} questions")
    
    # Collect results
    for idx, answer, error in results:
        if error:
             out_data['qa'][idx][prediction_key] = f"Error: {error}"
        else:
             out_data['qa'][idx][prediction_key] = answer
             
    # Save once at the end of sample
    if out_samples is not None and out_file is not None:
        out_samples[in_data.get('sample_id', 'unknown')] = out_data
        with open(out_file, "w") as f:
            json.dump(list(out_samples.values()), f, indent=2)
            
    return out_data

def get_gpt_answers(in_data, out_data, prediction_key, args, out_samples=None, out_file=None):
    """
    Synchronous wrapper for the async function.
    """
    return asyncio.run(get_gpt_answers_async(in_data, out_data, prediction_key, args, out_samples, out_file))
