import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os, json
from tqdm import tqdm
import argparse
from locomo.utils.openai_client import set_openai_key
from locomo.evaluation.evaluation import eval_question_answering
from locomo.evaluation.evaluation_stats import analyze_aggr_acc
from locomo.evaluation.gpt_utils import get_gpt_answers

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-file", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--data-file", type=str, required=True)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    return args

def main():
    # get arguments
    args = parse_args()

    print("******************  Evaluating Model %s ***************" % args.model)

    # Always use OpenAI format
    set_openai_key()

    # load conversations
    samples = json.load(open(args.data_file))
    prediction_key = "%s_prediction" % args.model
    model_key = "%s" % args.model
    
    # load the output file if it exists to check for overwriting
    if os.path.exists(args.out_file):
        out_samples = {d["sample_id"]: d for d in json.load(open(args.out_file))}
    else:
        out_samples = {}

    for data in samples:
        out_data = {"sample_id": data["sample_id"]}
        if data["sample_id"] in out_samples:
            out_data["qa"] = out_samples[data["sample_id"]]["qa"].copy()
        else:
            out_data["qa"] = data["qa"].copy()

        # Flatten conversation data if present
        if 'conversation' in data and isinstance(data['conversation'], dict):
            conv_data = data['conversation']
            for k, v in conv_data.items():
                data[k] = v
            
            # Map speakers to person1/person2 if needed
            if 'speaker_a' in conv_data and 'person1' not in data:
                data['person1'] = conv_data['speaker_a']
            if 'speaker_b' in conv_data and 'person2' not in data:
                data['person2'] = conv_data['speaker_b']

        # Always use get_gpt_answers (OpenAI format)
        answers = get_gpt_answers(data, out_data, prediction_key, args)

        # evaluate individual QA samples and save the score
        exact_matches, lengths, recall = eval_question_answering(answers["qa"], prediction_key)
        for i in range(0, len(answers["qa"])):
            answers["qa"][i][model_key + "_f1"] = round(exact_matches[i], 3)

        out_samples[data["sample_id"]] = answers

    with open(args.out_file, "w") as f:
        json.dump(list(out_samples.values()), f, indent=2)
    
    analyze_aggr_acc(args.data_file, args.out_file, args.out_file.replace(".json", "_stats.json"),
                model_key, model_key + "_f1")

if __name__ == "__main__":
    main()
