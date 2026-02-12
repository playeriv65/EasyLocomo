import argparse
from locomo.evaluation.runner import run_locomo
from locomo.config import config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--max-context", default=32768, type=int, help="Maximum context length for the model")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--category", type=int, default=None, help="Specific category to evaluate (1-5)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to evaluate")
    parser.add_argument("--max-questions", type=int, default=None, help="Limit number of questions per sample")
    
    args = parser.parse_args()

    config.model_name = args.model
    config.batch_size = args.batch_size
    config.max_context = args.max_context
    config.overwrite = args.overwrite
    config.category = args.category
    config.max_samples = args.limit
    config.max_questions = args.max_questions
    
    config.run_validate()

def run_test():
    parse_args()
    
    print(f"Starting evaluation for {config.model_name}...")
    if config.category is not None:
        print(f"Filtering for category: {config.category}")
    print(f"Output file: {config.out_file_path}")
    
    try:
        run_locomo()
        print("\nEvaluation completed successfully!")
        print(f"Results saved to: {config.out_file_path}")
    except Exception as e:
        print(f"\nAn error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
