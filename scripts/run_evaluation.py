import os
import sys
from pathlib import Path

# Ensure the project root is in sys.path
# Assuming scripts/run_evaluation.py, so parent.parent is the root
sys.path.insert(0, str(Path(__file__).parent.parent))

from locomo.evaluation.runner import main as run_qa_eval

def run_test(model_name="gpt-3.5-turbo", batch_size=1, data_file="data/locomo10.json", api_key=None, base_url=None):
    # Set API configuration if provided
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_API_BASE"] = base_url

    # Resolve data_file relative to project root if it's a relative path
    if not os.path.isabs(data_file):
        data_file = os.path.join(Path(__file__).parent.parent, data_file)

    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        return

    out_dir = os.path.join(Path(__file__).parent.parent, "outputs")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    out_file = os.path.join(out_dir, f"{model_name.replace('/', '_')}_qa.json")
    
    # Mock command line arguments for runner.py
    sys.argv = [
        "runner.py",
        "--data-file", data_file,
        "--out-file", out_file,
        "--model", model_name,
        "--batch-size", str(batch_size)
    ]
    
    print(f"Starting evaluation for {model_name}...")
    print(f"Data file: {data_file}")
    print(f"Output file: {out_file}")
    
    try:
        run_qa_eval()
        print("\nEvaluation completed successfully!")
        print(f"Results saved to: {out_file}")
    except Exception as e:
        print(f"\nAn error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # You can change the model name and API configuration here
    run_test(
        model_name="Qwen/Qwen3-8B-FP8", 
        batch_size=4,
        api_key="YOUR_API_KEY",      # Replace with your key
        base_url="http://100.67.94.49:8000/v1" # Replace with your base URL if needed
    )
