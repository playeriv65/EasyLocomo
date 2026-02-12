import os, json
from tqdm import tqdm
import argparse
from locomo.evaluation.gpt_utils import get_gpt_answers
from locomo.evaluation.evaluate_qa import evaluate_and_report
from locomo.config import config

def run_locomo():
    # Ensure configuration is valid
    assert config.out_file_path is not None, "out_file_path must be set before running"
    
    # load conversations
    samples = json.load(open(config.data_path))
    prediction_key = "%s_prediction" % config.model_name
    model_key = "%s" % config.model_name
    
    # load the output file if it exists to check for overwriting
    if config.out_file_path and os.path.exists(config.out_file_path):
        out_samples = {d["sample_id"]: d for d in json.load(open(config.out_file_path))}
    else:
        out_samples = {}

    if config.max_samples:
        samples = samples[:config.max_samples]

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
        answers = get_gpt_answers(data, out_data, prediction_key, config, out_samples, config.out_file_path)

        # Inline evaluation removed. Done in batch at the end.
        out_samples[data["sample_id"]] = answers

        # Real-time saving: update the output file after each sample
        with open(config.out_file_path, "w") as f:
            json.dump(list(out_samples.values()), f, indent=2)

    print("\n" + "="*50)
    print("FINISHED PREDICTIONS. ANALYZING RESULTS...")
    print("="*50)
    
    evaluate_and_report(
        qa_file=config.out_file_path,
        model_name=model_key,
        data_file=config.data_path
    )

