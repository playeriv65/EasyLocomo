import argparse
import subprocess
import sys
import os

def run_test(test_type):
    model_name = "gemini-2.5-flash" # Default model for testing
    
    if test_type == "small":
        print("Running SMALL test (limit 1 sample, 5 questions)...")
        command = [
            sys.executable, "run_evaluation.py",
            "--model", model_name,
            "--limit", "1",
            "--max-questions", "5",
            "--batch-size", "5", # Enable concurrency
            "--overwrite"
        ]
    elif test_type == "large":
        print("Running LARGE test (all samples)...")
        command = [
            sys.executable, "run_evaluation.py",
            "--model", model_name,
            "--overwrite"
        ]
    else:
        print(f"Unknown test type: {test_type}")
        return

    try:
        subprocess.run(command, check=True)
        print(f"\n{test_type.upper()} test completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n{test_type.upper()} test failed with error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Locomo automated tests")
    parser.add_argument("test_type", choices=["small", "large"], help="Type of test to run")
    
    args = parser.parse_args()
    
    # Ensure we are in the root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root_dir)
    sys.path.append(root_dir)

    run_test(args.test_type)
