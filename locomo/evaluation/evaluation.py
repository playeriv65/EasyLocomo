from locomo.evaluation.metrics import f1_score_single, f1_score_multi, calculate_bert_score

def eval_question_answering(qas, eval_key='prediction', run_bert_score=False):
    f1_scores = []
    bert_predictions = []
    bert_references = []
    
    for i, line in enumerate(qas):
        answer = line.get('answer', line.get('adversarial_answer', ''))
        
        if isinstance(line.get(eval_key), list):
             pass
        else:
            answer = str(answer)
            
        if line['category'] == 3:
            answer = answer.split(';')[0].strip()
        
        output = line.get(eval_key, line.get('prediction', ''))
        if output is None:
            output = ''
            
        score = 0
        if line['category'] in [2, 3, 4]:
            score = f1_score_single(output, answer)
        elif line['category'] in [1]:
            score = f1_score_multi(output, answer)
        elif line['category'] in [5]:
            if 'no information available' in output.lower() or 'not mentioned' in output.lower():
                score = 1
            else:
                score = 0
        else:
            score = 0
            
        f1_scores.append(score)
        bert_predictions.append(output)
        bert_references.append(answer)

    results = {'f1': f1_scores}

    if run_bert_score:
        print('Calculating BERTScore...')
        bert_scores = calculate_bert_score(bert_predictions, bert_references, verbose=True)
        results['bertscore'] = bert_scores

    print(f'{len(qas)} QA samples evaluated.')
    return results
