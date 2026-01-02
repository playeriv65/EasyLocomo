import random
import os, json
from tqdm import tqdm
import time
from locomo.utils.openai_client import run_chatgpt
import tiktoken

MAX_LENGTH={'gpt-4-turbo': 128000,
            'gpt-4': 4096,
            'gpt-3.5-turbo-16k': 16000,
            'gpt-3.5-turbo-12k': 12000,
            'gpt-3.5-turbo-8k': 8000,
            'gpt-3.5-turbo-4k': 4000,
            'gpt-3.5-turbo': 4096,
            'gpt-4-32k': 320000}
PER_QA_TOKEN_BUDGET = 50

QA_PROMPT = """
Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {} Short answer:
"""

QA_PROMPT_CAT_5 = """
Based on the above context, answer the following question.

Question: {} Short answer:
"""

# QA_PROMPT_BATCH = """
# Based on the above conversations, answer the following questions in a few words. Write the answers as a list of strings in the json format. Start and end with a square bracket.

# """

QA_PROMPT_BATCH = """
Based on the above conversations, write short answers for each of the following questions in a few words. 
Write the answers in the form of a json dictionary where each entry contains the question number as "key" and the short answer as "value". 
Use single-quote characters for named entities and double-quote characters for enclosing json elements. Answer with exact words from the conversations whenever possible.

"""

# If no information is available to answer the question, write 'No information available'.

CONV_START_PROMPT = "Below is a conversation between two people: {} and {}. The conversation takes place over multiple days and the date of each conversation is wriiten at the beginning of the conversation.\n\n"


def process_ouput(text):
    # Try to find a JSON-like structure in the text
    text = text.strip()
    if not text:
        raise ValueError("Empty response from model")
    
    # Remove markdown code blocks if present
    if "```" in text:
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        else:
            # Fallback: just strip the backticks and language identifier
            text = re.sub(r'```(?:json)?', '', text).replace('```', '').strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try the original heuristic for single/double quotes
        single_quote_count = text.count("'")
        double_quote_count = text.count('"')
        if single_quote_count > double_quote_count:
            temp_text = text.replace('"', "")
            temp_text = temp_text.replace("'", '"')
            try:
                return json.loads(temp_text)
            except:
                pass
        
        # Last resort: try to find anything between { and }
        import re
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
        
        raise


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

    query_conv = ''
    min_session = -1
    stop = False
    session_nums = [int(k.split('_')[-1]) for k in data.keys() if 'session' in k and 'date_time' not in k]
    for i in range(min(session_nums), max(session_nums) + 1):
        if 'session_%s' % i in data:
            query_conv += "\n\n"
            for dialog in data['session_%s' % i][::-1]:
                turn = ''
                turn = dialog['speaker'] + ' said, \"' + dialog['text'] + '\"' + '\n'
                if "blip_caption" in dialog:
                    turn += ' and shared %s.' % dialog["blip_caption"]
                turn += '\n'
        
                num_tokens = len(encoding.encode('DATE: ' + data['session_%s_date_time' % i] + '\n' + 'CONVERSATION:\n' + turn))
                model_max_length = MAX_LENGTH.get(args.model, 4096) # Default to 4k if unknown
                if (num_tokens + len(encoding.encode(query_conv)) + num_question_tokens) < (model_max_length-(PER_QA_TOKEN_BUDGET*(args.batch_size))): # 20 tokens assigned for answers
                    query_conv = turn + query_conv
                else:
                    min_session = i
                    stop = True
                    break
            query_conv = 'DATE: ' + data['session_%s_date_time' % i] + '\n' + 'CONVERSATION:\n' + query_conv
        if stop:
            break
        
        # if min_session == -1:
        #     print("Saved %s tokens in query conversation from full conversation" % len(encoding.encode(query_conv)))
        # else:
        #     print("Saved %s conv. tokens + %s question tokens in query from %s out of %s sessions" % (len(encoding.encode(query_conv)), num_question_tokens, max_session-min_session, max_session))

    return query_conv


def get_gpt_answers(in_data, out_data, prediction_key, args):
    encoding = tiktoken.get_encoding("cl100k_base")
    start_prompt = CONV_START_PROMPT.format(in_data['person1'], in_data['person2'])
    start_tokens = len(encoding.encode(start_prompt))

    for batch_start_idx in tqdm(range(0, len(in_data['qa']), args.batch_size), desc='Generating answers'):

        questions = []
        include_idxs = []
        cat_5_idxs = []
        cat_5_answers = []
        for i in range(batch_start_idx, batch_start_idx + args.batch_size):

            if i>=len(in_data['qa']):
                break

            qa = in_data['qa'][i]
            
            if prediction_key not in out_data['qa'][i] or args.overwrite:
                include_idxs.append(i)
            else:
                continue

            if qa['category'] == 2:
                questions.append(qa['question'] + ' Use DATE of CONVERSATION to answer with an approximate date.')
            elif qa['category'] == 5:
                distractor = qa['adversarial_answer']
                question = qa['question'] + " (a) {} (b) {}"
                question = question.format(distractor, 'Not mentioned in the conversation')
                answer = {'b': 'Not mentioned in the conversation', 'a': distractor}

                cat_5_idxs.append(len(questions))
                questions.append(question)
                cat_5_answers.append(answer)
                # questions.append(qa['question'] + "Write NOT ANSWERABLE if the question cannot be answered")
            else:
                questions.append(qa['question'])


        if questions == []:
            continue


        question_prompt =  QA_PROMPT_BATCH + "\n".join(["%s: %s" % (k, q) for k, q in enumerate(questions)])
        num_question_tokens = len(encoding.encode(question_prompt))
        query_conv = get_input_context(in_data['conversation'], num_question_tokens + start_tokens, encoding, args)
        query_conv = start_prompt + query_conv
        
        # print("%s tokens in query" % len(encoding.encode(query_conv)))

        if args.batch_size == 1:

            query = query_conv + '\n\n' + QA_PROMPT.format(questions[0]) if len(cat_5_idxs) == 0 else query_conv + '\n\n' + QA_PROMPT_CAT_5.format(questions[0])
            answer = run_chatgpt(query, num_gen=1, num_tokens_request=32, 
                    model='chatgpt' if 'gpt-3.5' in args.model else args.model, 
                    use_16k=True if any([k in args.model for k in ['16k', '12k', '8k', '4k']]) else False, 
                    temperature=0, wait_time=2)
            
            if len(cat_5_idxs) > 0:
                answer = get_cat_5_answer(answer, cat_5_answers[0])

            out_data['qa'][include_idxs[0]][prediction_key] = answer.strip()

        else:
            question_prompt =  QA_PROMPT_BATCH + "\n".join(["%s: %s" % (k, q) for k, q in enumerate(questions)])
            num_question_tokens = len(encoding.encode(question_prompt))
            query = query_conv + '\n\n' + question_prompt
            for trials in range(1, 4):
                try:
                    tqdm.write("Trial %s/3" % trials)
                    # print("Sending query of %s tokens" % len(encoding.encode(query)))
                    # print("Trying with answer token budget = %s per question" % PER_QA_TOKEN_BUDGET)
                    answer = run_chatgpt(query, num_gen=1, num_tokens_request=args.batch_size*PER_QA_TOKEN_BUDGET, 
                            model='chatgpt' if 'gpt-3.5' in args.model else args.model, 
                            use_16k=True if any([k in args.model for k in ['16k', '12k', '8k', '4k']]) else False, 
                            temperature=0, wait_time=2)
                    if not answer:
                        tqdm.write(f"Warning: Empty answer received on trial {trials}")
                        if trials == 3:
                            answer = "" # Ensure answer is defined
                        else:
                            continue
                    answer = answer.replace('\\"', "'").replace('json','').replace('`','').strip().replace("\\'", "")
                    # answers = process_ouput(answer.strip())
                    break

                except Exception as e:
                    tqdm.write(f"Error at trial {trials}/3: {e}")
                    if trials == 3:
                        output_str = str(answer) if 'answer' in locals() else "None"
                        tqdm.write(f"Failed after 3 trials. Model output was: {output_str}")
                        raise ValueError
                    time.sleep(2)
                    continue
            
            for k, idx in enumerate(include_idxs):
                parsed_answer = None
                
                # Attempt 1: process_ouput (handles markdown, regex for {}, etc.)
                try:
                    answers = process_ouput(answer.strip())
                    if isinstance(answers, dict):
                        parsed_answer = answers.get(str(k), answers.get(k))
                    elif isinstance(answers, list) and k < len(answers):
                        parsed_answer = answers[k]
                except:
                    pass
                
                # Attempt 2: direct json.loads
                if parsed_answer is None:
                    try:
                        answers = json.loads(answer.strip())
                        if isinstance(answers, dict):
                            parsed_answer = answers.get(str(k), answers.get(k))
                        elif isinstance(answers, list) and k < len(answers):
                            parsed_answer = answers[k]
                    except:
                        pass
                
                # Final assignment
                if parsed_answer is not None:
                    if k in cat_5_idxs:
                        out_data['qa'][idx][prediction_key] = get_cat_5_answer(str(parsed_answer), cat_5_answers[cat_5_idxs.index(k)])
                    else:
                        out_data['qa'][idx][prediction_key] = str(parsed_answer).replace('(a)', '').replace('(b)', '').strip()
                else:
                    out_data['qa'][idx][prediction_key] = "Error: Could not parse answer"

    return out_data
