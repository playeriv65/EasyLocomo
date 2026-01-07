import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Ensure the project root is in sys.path
# The script is in the root, so parent is the root
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

from locomo.evaluation.runner import main as run_qa_eval

def run_test(model_name="gpt-4o-mini", batch_size=1, max_context=32768, data_file="data/locomo10.json", api_key=None, base_url=None, category=None, overwrite=False):
    # Set API configuration if provided
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_API_BASE"] = base_url

    # Resolve data_file relative to project root if it's a relative path
    if not os.path.isabs(data_file):
        data_file = os.path.join(root_dir, data_file)

    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        return

    out_dir = os.path.join(root_dir, "outputs")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    out_file = os.path.join(out_dir, f"{model_name.replace('/', '_')}_qa.json")
    
    # Mock command line arguments for runner.py
    cmd_args = [
        "runner.py",
        "--data-file", data_file,
        "--out-file", out_file,
        "--model", model_name,
        "--batch-size", str(batch_size),
        "--max-context", str(max_context)
    ]
    
    if category is not None:
        cmd_args.extend(["--category", str(category)])
    
    if overwrite:
        cmd_args.append("--overwrite")
        
    sys.argv = cmd_args
    
    print(f"Starting evaluation for {model_name}...")
    if category is not None:
        print(f"Filtering for category: {category}")
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
    # Credentials can be set in .env or passed here
    api_key = os.getenv("OPENAI_API_KEY") 
    base_url = os.getenv("OPENAI_API_BASE") 
    
    run_test(
        model_name="gpt-4o-mini", 
        batch_size=15,
        api_key=api_key,
        base_url=base_url,
        max_context=65536,
    )
