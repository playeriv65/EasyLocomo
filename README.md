# EasyLocomo

**EasyLocomo** is a streamlined, easy-to-use version of the evaluation framework for the **LoCoMo** (Long-term Conversational Memory) benchmark. 

This repository adapts the original logic and data from the paper ["Evaluating Very Long-Term Conversational Memory of LLM Agents"](https://github.com/snap-research/locomo) (ACL 2024), ensuring that the evaluation results are consistent with the original author's repository while providing a much simpler experience for testing any LLM via OpenAI-compatible APIs.

## 🌟 Key Features

- **Result Consistency**: Uses the same data and evaluation logic as the original LoCoMo project. Consistency of results has been verified using GPT-4o-mini. See [release 0.1.0](../../releases/tag/v0.1.0) for details.
- **Cache-Friendly**: Implemented fixed-length context truncation to maximize Gemini implicit prompt caching (approx. 95% hit rate).
- **Fully Deterministic**: Eliminated prompt randomness via hashing and sorted processing, guaranteeing 100% reproducible inputs and outputs.
- **Simplified Setup**: No complex bash scripts or environment setup. Optimized for [uv](https://github.com/astral-sh/uv) and standard Python environments.
- **OpenAI API Compatibility**: Call any LLM that supports the OpenAI API format (e.g., GPT-4o, GPT-4o-mini, Claude via proxy, DeepSeek, or local models via Ollama/vLLM).
- **RAG Demo**: Includes data extraction tools and a vanilla RAG example (`examples/vanilla_rag.py`) to demonstrate external memory integration.
- **No-Context Evaluation**: Supports a `--no-context` mode to evaluate models that rely entirely on external memory or retrieval.
- **Flexible Configuration**: Easily set your API key, base URL, and model name.
- **Breakpoint Resumption**: Automatically saves progress after each sample/batch and skips already predicted samples, allowing for reliable long-running evaluations.
- **JSON Mode & Robust Parsing**: Utilizes OpenAI's JSON mode for structured outputs and includes advanced cleaning logic (removing reasoning thoughts, markdown blocks) to ensure high parsing success rates.
- **Error Logging**: Detailed parsing errors are logged to a separate `*_errors.jsonl` file for easy debugging and model output analysis.
- **Automatic Reporting**: Automatically generates performance statistics (Accuracy, BERTScore, etc.) and summaries of the results.
- **Token Estimation**: Includes a utility script to estimate the token count of the evaluation dataset to help manage costs.

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install the dependencies. We recommend using [uv](https://github.com/astral-sh/uv) for extremely fast setup:

```bash
# Using uv (Recommended)
uv sync

# Or using standard pip
pip install -r requirements.txt
```

### 2. Configuration

You can configure your API credentials by creating a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1
```

Or you can pass them directly in the `run_evaluation.py` script.

### 3. Run Evaluation

Simply run the `run_evaluation.py` script:

```bash
# Using uv
uv run run_evaluation.py

# Or using standard python
python run_evaluation.py
```

### 4. Run Tests

To verify logic and concurrency:

```bash
# Run unit tests
python -m pytest tests/

# Run a quick integration test
python scripts/test_runner.py small
```

By default, this will evaluate the model on the `data/locomo10.json` dataset. Results, including predictions and statistical reports, will be saved in the `outputs/` directory.

---

## 📊 Results and Statistics

After running the evaluation, you will find the following files in the `outputs/` directory:
- `[model_name]_qa.json`: The model's predictions.
- `[model_name]_qa_stats.json`: Detailed accuracy metrics (Overall, Session-level, etc.).
- `[model_name]_qa_summary.json`: A human-readable summary of the evaluation results.

---

## 📚 Reference & Citation

This project is built upon the work by Maharana et al. (ACL 2024). Please cite the original paper if you use this benchmark:

```bibtex
@inproceedings{maharana2024locomo,
  title={Evaluating Very Long-Term Conversational Memory of LLM Agents},
  author={Maharana, Adyasha and Lee, Dong-Ho and Tulyakov, Sergey and Bansal, Mohit and Barbieri, Francesco and Fang, Yuwei},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2024}
}
```

Original Repository: [snap-research/locomo](https://github.com/snap-research/locomo)

---

## 🛠️ Advanced Usage

You can customize the evaluation parameters in `run_evaluation.py`:

```python
run_test(
    model_name="gpt-4o-mini", 
    batch_size=15,
    max_context=65536,
    data_file="data/locomo10.json",
    category=1,
    overwrite=False
)
```

- **model_name**: The identifier of the model to test.
- **batch_size**: Number of concurrent API calls.
- **max_context**: Maximum context length (tokens) passed to the model.
- **category**: (Optional) Filter evaluation for a specific category (1-5). Useful for re-testing specific subsets.
- **overwrite**: Whether to re-run evaluations for already predicted samples.

---

## License

This project follows the licensing of the original LoCoMo repository. See `LICENSE` for details.
