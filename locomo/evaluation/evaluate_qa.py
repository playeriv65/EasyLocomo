import argparse
import json
import os
import sys

# Ensure locomo is on path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../../'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from locomo.evaluation.evaluation import eval_question_answering
from locomo.evaluation.evaluation_stats import analyze_aggr_acc
from locomo.evaluation.summary_generator import generate_summary_from_stats

def evaluate_and_report(qa_file, model_name, data_file=None, no_bert=False, metrics="f1,bertscore"):
    if not data_file: 
        if os.path.exists(qa_file):
             data_file = qa_file
    
    print(f"Evaluating {qa_file} for model {model_name}")
    
    with open(qa_file, 'r') as f:
        data = json.load(f)
        
    all_qas = []
    if isinstance(data, list):
        for sample in data:
            if 'qa' in sample:
                all_qas.extend(sample['qa'])
    
    if not all_qas:
        print("No QA items found.")
        return

    pred_key = f"{model_name}_prediction"
    
    requested_metrics = metrics.split(',')
    run_bert = 'bertscore' in requested_metrics and not no_bert
    
    results = eval_question_answering(all_qas, eval_key=pred_key, run_bert_score=run_bert)
    
    # Update data with scores
    q_idx = 0
    for sample in data:
        if 'qa' in sample:
            for qa_item in sample['qa']:
                if 'f1' in results:
                    qa_item[f"{model_name}_f1"] = results['f1'][q_idx]
                if 'bertscore' in results:
                    qa_item[f"{model_name}_bertscore"] = results['bertscore'][q_idx]
                q_idx += 1
            
    with open(qa_file, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"Updated {qa_file} with scores.")
    
    # Run Stats
    stats_file = qa_file.replace(".json", "_stats.json")
    
    metric_keys = []
    if 'f1' in results:
        metric_keys.append(f"{model_name}_f1")
    if 'bertscore' in results:
        metric_keys.append(f"{model_name}_bertscore")
        
    stats = analyze_aggr_acc(data_file, qa_file, stats_file, model_name, metric_keys)
    
    # Run Summary
    summary_file = qa_file.replace(".json", "_summary.json")
    summary = generate_summary_from_stats(stats, model_name)
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"Summary saved to {summary_file}")
    
    for metric, res in summary.items():
        if 'overall' in res:
            print(f"{metric}: {res['overall'].get('accuracy', 0):.2%}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate QA results with F1 and BERTScore")
    parser.add_argument("--qa-file", type=str, required=True, help="Path to the qa.json file")
    parser.add_argument("--model-name", type=str, required=True, help="Model name identifier in the json keys")
    parser.add_argument("--data-file", type=str, help="Original data file", default=None)
    parser.add_argument("--no-bert", action="store_true", help="Skip BERTScore calculation")
    parser.add_argument("--metrics", type=str, default="f1,bertscore", help="Comma separated list of metrics to run")

    args = parser.parse_args()
    
    evaluate_and_report(args.qa_file, args.model_name, args.data_file, args.no_bert, args.metrics)


if __name__ == "__main__":
    main()
