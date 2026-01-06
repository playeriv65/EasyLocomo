import os, json
import math
from tqdm import tqdm
from collections import defaultdict


def get_conversation_lengths(data, encoder=None):

    total_conv_length = 0
    id2length = {}
    for sess_num in range(1, 50):
        if 'session_%s' % sess_num not in data:
            continue
        if data['session_%s' % sess_num] == []:
            continue

        for dialog in data['session_%s' % sess_num]:
            dialog_tokens = dialog['speaker'] + ': ' + dialog['text'] + '\n'
            if "img_file" in dialog and len(dialog["img_file"]) > 0:
                dialog_tokens += '[shares %s]\n' % dialog["blip_caption"]
            if encoder is not None:
                dialog_length = len(encoder.encode(dialog_tokens))
            else:
                # dialog_length = len(dialog_tokens.split())
                dialog_length = len(dialog_tokens)
            id2length[dialog["dia_id"]] = total_conv_length + dialog_length
            total_conv_length += dialog_length
    return id2length


def analyze_aggr_acc(ann_file, in_file, out_file, model_name, metric_key, encoder=None):

    total_counts = defaultdict(lambda: 0)
    acc_counts = defaultdict(lambda: 0)
    memory_counts = defaultdict(lambda: defaultdict(lambda: 0))
    memory_counts_og = defaultdict(lambda: defaultdict(lambda: 0))
    context_len_counts = defaultdict(lambda: 0)
    context_len_og = defaultdict(lambda: 0)
    recall_by_category = defaultdict(lambda: 0)

    outputs = {d['sample_id']: d for d in json.load(open(in_file))}
    data = {d['sample_id']: d for d in json.load(open(ann_file))}
    sample_ids = outputs.keys()
    
    for sample_id in sample_ids:
        output = outputs[sample_id]
        ann = data[sample_id]

        id2length = get_conversation_lengths(ann['conversation'], encoder)
        # print(id2length)

        for i, qa in tqdm(enumerate(output['qa'])):
            # if qa['category'] in [4, 5]:
            #     continue
            total_counts[qa['category']] += 1
            if metric_key in qa:
                
                acc_counts[qa['category']] += qa[metric_key]
                qa['evidence'] = [q.replace('(', '').replace(')', '') for q in qa["evidence"]]
                if len(qa['evidence']) > 0:
                    # farthest_session = min([int(e.split(':')[0][1:]) for e in qa['evidence'] if e != ""])
                    # memory_counts_og[farthest_session] += 1
                    # if qa[metric_key]:
                    #     memory_counts[farthest_session] += qa[metric_key]

                    try:
                        farthest_session = min([int(e.split(':')[0][1:]) for e in qa['evidence'] if e != ""])
                        farthest_dialog = min([int(e.split(':')[-1]) for e in qa['evidence'] if e != "" and int(e.split(':')[0][1:]) == farthest_session])

                        farthest_length = id2length['D' + str(farthest_session) + ':' + str(farthest_dialog)]


                        memory_counts_og[qa['category']][math.ceil(farthest_length/1000)] += 1
                        memory_counts[qa['category']][math.ceil(farthest_length/1000)] += qa[metric_key]

                        if qa['category'] == 1:
                            latest_session = max([int(e.split(':')[0][1:]) for e in qa['evidence'] if e != ""])
                            latest_dialog = max([int(e.split(':')[-1]) for e in qa['evidence'] if e != "" and int(e.split(':')[0][1:]) == latest_session])

                            latest_length = id2length['D' + str(latest_session) + ':' + str(latest_dialog)]
                            context_length = latest_length-farthest_length
                            context_len_og[math.ceil(context_length/1000)] += 1
                            context_len_counts[math.ceil(context_length/1000)] += qa[metric_key]
                    except:
                        continue
            else:
                pass

    total_k = 0
    total_v = 0
    keys = [4, 1, 2, 3, 5]
    for k in keys:
        v = total_counts[k]
        total_v += acc_counts[k]
        total_k += v

    def default_to_regular(d):
        if isinstance(d, defaultdict):
            d = {k: default_to_regular(v) for k, v in d.items()}
        return d
    
    if os.path.exists(out_file):
        try:
            results_dict = json.load(open(out_file))
        except:
            results_dict = {}
    else:
        results_dict = {}

    results_dict[model_name] = {
        'category_counts': dict(total_counts),
        'cum_accuracy_by_category': dict(acc_counts),
        'category_counts_by_memory': default_to_regular(memory_counts_og),
        'cum_accuracy_by_category_by_memory': default_to_regular(memory_counts),
        'context_length_counts': dict(context_len_og),
        'cum_accuracy_by_context_length': dict(context_len_counts)
    }

    with open(out_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    return results_dict[model_name]


if __name__ == "__main__":

    # analyze_acc('./data/multimodal_dialog/completed_annotations/3_out_gpt3.5_summary.json', 'gpt3.5-16k')
    
    # analyze_aggr_acc('./data/multimodal_dialog/quest_data_final/with_qa',
    #                  './data/multimodal_dialog/quest_data_final/qa_outputs', 
    #                  './data/multimodal_dialog/quest_data_final/qa_outputs/all_results.json',
    #                  'gpt-3.5-turbo',
    #                  'gpt-3.5-turbo_f1'
    #                  )
    
    # analyze_aggr_acc('./data/multimodal_dialog/quest_data_final/with_qa',
    #                 './data/multimodal_dialog/quest_data_final/qa_outputs', 
    #                 './data/multimodal_dialog/quest_data_final/qa_outputs/all_results.json',
    #                 'gpt-3.5-turbo-16k',
    #                 'gpt-3.5-turbo-16k_f1'
    #                 )

    # analyze_aggr_acc('./data/multimodal_dialog/final',
    #             './outputs/all', 
    #             './outputs/all_results.json',
    #             'gemini-pro-1.0',
    #             'gemini-pro-1.0_f1'
    #             )

    # analyze_aggr_acc('./data/multimodal_dialog/final',
    #             './outputs/all', 
    #             './outputs/all_results.json',
    #             'llama3-chat-70b',
    #             'llama3-chat-70b_rouge'
    #             )

    # analyze_aggr_acc('./data/multimodal_dialog/final',
    #             './outputs/all', 
    #             './outputs/all_results.json',
    #             'gpt-3.5-turbo_summary_top_10',
    #             'gpt-3.5-turbo_summary_top_10_f1'
    #             )

    analyze_aggr_acc('./data/locomo10.json', './data/locomo10_qa.json',
            './data/locomo10_qa_scores.json',
            'gemini-pro-1.0',
            'gemini-pro-1.0_f1'
            )