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
            if 'img_file' in dialog and len(dialog['img_file']) > 0:
                dialog_tokens += '[shares %s]\n' % dialog['blip_caption']
            if encoder is not None:
                dialog_length = len(encoder.encode(dialog_tokens))
            else:
                dialog_length = len(dialog_tokens)
            id2length[dialog['dia_id']] = total_conv_length + dialog_length
            total_conv_length += dialog_length
    return id2length


def analyze_aggr_acc(ann_file, in_file, out_file, model_name, metric_keys, encoder=None):
    if isinstance(metric_keys, str):
        metric_keys = [metric_keys]

    total_counts = defaultdict(lambda: 0)
    
    # Store accumulators for each metric
    # metrics_data[metric][key]
    metrics_data = defaultdict(lambda: {
        'acc_counts': defaultdict(lambda: 0),
        'memory_counts': defaultdict(lambda: defaultdict(lambda: 0)),
        'context_len_counts': defaultdict(lambda: 0)
    })
    
    memory_counts_og = defaultdict(lambda: defaultdict(lambda: 0))
    context_len_og = defaultdict(lambda: 0)

    outputs = {d['sample_id']: d for d in json.load(open(in_file))}
    data = {d['sample_id']: d for d in json.load(open(ann_file))}
    sample_ids = sorted(outputs.keys())
    
    print(f'Analyzing metrics: {metric_keys}')
    
    for sample_id in sample_ids:
        output = outputs[sample_id]
        ann = data[sample_id]

        id2length = get_conversation_lengths(ann['conversation'], encoder)

        for i, qa in enumerate(output['qa']):
            total_counts[qa['category']] += 1
            
            qa_evidence_clean = []
            farthest_length = -1
            latest_length = -1
            has_valid_evidence = False
            
            if 'evidence' in qa and qa['evidence']:
                qa_evidence_clean = [q.replace('(', '').replace(')', '') for q in qa['evidence']]
                if len(qa_evidence_clean) > 0:
                    try:
                        farthest_session = min([int(e.split(':')[0][1:]) for e in qa_evidence_clean if e != ''])
                        matching_dialogs_f = [int(e.split(':')[-1]) for e in qa_evidence_clean if e != '' and int(e.split(':')[0][1:]) == farthest_session]
                        if matching_dialogs_f:
                            farthest_dialog = min(matching_dialogs_f)
                            farthest_length = id2length['D' + str(farthest_session) + ':' + str(farthest_dialog)]
                            has_valid_evidence = True

                        if qa['category'] == 1:
                            latest_session = max([int(e.split(':')[0][1:]) for e in qa_evidence_clean if e != ''])
                            matching_dialogs_l = [int(e.split(':')[-1]) for e in qa_evidence_clean if e != '' and int(e.split(':')[0][1:]) == latest_session]
                            if matching_dialogs_l:
                                latest_dialog = max(matching_dialogs_l)
                                latest_length = id2length['D' + str(latest_session) + ':' + str(latest_dialog)]
                    except Exception as e:
                        pass

            if has_valid_evidence:
                memory_counts_og[qa['category']][math.ceil(farthest_length/1000)] += 1
                if qa['category'] == 1 and latest_length != -1:
                    context_length = latest_length - farthest_length
                    context_len_og[math.ceil(context_length/1000)] += 1

            for m_key in metric_keys:
                if m_key in qa:
                    score = qa[m_key]
                    metrics_data[m_key]['acc_counts'][qa['category']] += score
                    
                    if has_valid_evidence:
                        metrics_data[m_key]['memory_counts'][qa['category']][math.ceil(farthest_length/1000)] += score
                        if qa['category'] == 1 and latest_length != -1:
                            context_length = latest_length - farthest_length
                            metrics_data[m_key]['context_len_counts'][math.ceil(context_length/1000)] += score


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

    metrics_output = {}
    for m_key, data in metrics_data.items():
        metrics_output[m_key] = {
            'cum_accuracy_by_category': dict(data['acc_counts']),
            'cum_accuracy_by_category_by_memory': default_to_regular(data['memory_counts']),
            'cum_accuracy_by_context_length': dict(data['context_len_counts'])
        }

    results_dict[model_name] = {
        'category_counts': dict(total_counts),
        'category_counts_by_memory': default_to_regular(memory_counts_og),
        'context_length_counts': dict(context_len_og),
        'metrics': metrics_output
    }
    
    if metric_keys and metric_keys[0] in metrics_output:
        base_metric = metrics_output[metric_keys[0]]
        results_dict[model_name].update(base_metric)

    with open(out_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    return results_dict[model_name]
