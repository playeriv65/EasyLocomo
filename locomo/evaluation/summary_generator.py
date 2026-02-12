import json
import os
import argparse

def calculate_metric_summary(category_counts, acc_counts):
    summary = {
        "categories": {},
        "overall": {}
    }
    total_correct = 0
    total_questions = 0

    cat_names = {
        "1": "Multi-hop",
        "2": "Temporal",
        "3": "Single-hop",
        "4": "Open-domain",
        "5": "Adversarial"
    }

    for cat_id, count in category_counts.items():
        if count == 0:
            continue
            
        correct = acc_counts.get(cat_id, 0)
        accuracy = round(float(correct) / count, 4) if count > 0 else 0
        
        name = cat_names.get(str(cat_id), f"Category_{cat_id}")
        summary["categories"][name] = {
            "count": count,
            "accuracy": accuracy
        }
        
        total_correct += correct
        total_questions += count

    if total_questions > 0:
        summary["overall"] = {
            "total_questions": total_questions,
            "accuracy": round(float(total_correct) / total_questions, 4)
        }
    else:
        summary["overall"] = {
            "total_questions": 0,
            "accuracy": 0.0
        }
    return summary

def generate_summary_from_stats(stats_data: dict, model_name: str = None) -> dict:
    """
    Processes the raw statistics data and returns a consolidated summary.
    """
    if not stats_data:
        return {"error": "No stats data provided"}

    # If it's the full nested dict (e.g. loaded from file), extract the specific model's data
    if model_name and isinstance(stats_data, dict) and model_name in stats_data:
        data = stats_data[model_name]
    else:
        # Assume stats_data is already the model-specific dict
        data = stats_data

    if not isinstance(data, dict):
        return {"error": f"Invalid data format: expected dict, got {type(data)}"}

    category_counts = data.get('category_counts', {})
    
    # Check if we have multiple metrics
    metrics = data.get('metrics', {})
    
    full_summary = {}
    
    # If no "metrics" key, fallback to old style
    if not metrics:
        acc_counts = data.get('cum_accuracy_by_category', {})
        full_summary['default'] = calculate_metric_summary(category_counts, acc_counts)
    else:
        for metric_name, metric_data in metrics.items():
            acc_counts = metric_data.get('cum_accuracy_by_category', {})
            full_summary[metric_name] = calculate_metric_summary(category_counts, acc_counts)

    return full_summary

def main():
    parser = argparse.ArgumentParser(description="Generate a readable summary from stats file")
    parser.add_argument("--stats-file", type=str, required=True, help="Path to the _stats.json file")
    parser.add_argument("--model", type=str, help="Model name key in the stats file")
    parser.add_argument("--output", type=str, help="Path to save the summary (optional)")
    
    args = parser.parse_args()

    if not os.path.exists(args.stats_file):
        print(f"Error: File {args.stats_file} not found.")
        return

    with open(args.stats_file, 'r') as f:
        full_stats = json.load(f)

    # If model is not provided, take the first one found
    model_key = args.model
    if not model_key:
        model_key = sorted(full_stats.keys())[0]
        print(f"No model specified, using first available key: {model_key}")

    summary = generate_summary_from_stats(full_stats, model_key)
    
    # Print to console
    print("\n" + "="*30)
    print(f"EVALUATION SUMMARY: {model_key}")
    print("="*30)
    
    for metric, res in summary.items():
        print(f"\nMetric: {metric}")
        overall = res.get('overall', {})
        acc = overall.get('accuracy', 0)
        total_q = overall.get('total_questions', 0)
        print(f"Overall Accuracy: {acc:.2%}")
        
        categories = res.get('categories', {})
        for cat, val in categories.items():
            print(f"  {cat}: {val['accuracy']:.2%} ({val['count']})")
            
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
