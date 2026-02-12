# Vanilla RAG Example

This example demonstrates how to build a simple Retrieval-Augmented Generation (RAG) pipeline using the extracted data from EasyLocomo.

## Model
This example uses **Google Gemini** models via the OpenAI-compatible API endpoint:
- **Embedding**: `models/gemini-embedding-001`
- **Chat**: `models/gemini-2.0-flash`

## Prerequisites

Ensure you have generated the evidence and QA files:
1. `data/locomo10_evidence.json`
2.  `data/locomo10_qa.json`

## Dependencies

- `openai`
- `numpy`
- `tqdm`
- `python-dotenv`
- `tenacity`

(All included in project `requirements.txt`)

## Running the Example

```bash
python examples/vanilla_rag.py --limit 1
```
