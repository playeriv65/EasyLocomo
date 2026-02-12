
import json
import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from tenacity import retry,  stop_after_attempt, wait_random_exponential
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_API_BASE")
)

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text, model="models/gemini-embedding-001"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_completion(messages, model="models/gemini-2.0-flash"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
         print(f"Error with {model}: {e}")
         # Fallback to no-prefix or other model if needed
         return None

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def chunk_conversation(conversation_data):
    """
    Splits conversation into smaller chunks for embedding.
    """
    chunks = []
    
    # Sort sessions chronologically
    session_keys = sorted([k for k in conversation_data.keys() if k.startswith('session_') and 'date_time' not in k], 
                          key=lambda x: int(x.split('_')[1]))
    
    for session_key in session_keys:
        date_key = f"{session_key}_date_time"
        date_str = conversation_data.get(date_key, "Unknown Date")
        turns = conversation_data[session_key]
        
        # Group every 5 turns into a chunk
        window_size = 5
        stride = 3
        
        for i in range(0, len(turns), stride):
            window_turns = turns[i : i + window_size]
            if not window_turns:
                break
                
            chunk_text = f"Date: {date_str}\n"
            for turn in window_turns:
                chunk_text += f"{turn['speaker']}: {turn['text']}\n"
            
            chunks.append(chunk_text)
            
    return chunks

def main():
    parser = argparse.ArgumentParser(description="Vanilla RAG Example for EasyLocomo")
    parser.add_argument("--data-dir", default="data", help="Path to data directory")
    parser.add_argument("--limit", type=int, default=1, help="Number of samples to process (demo mode)")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    evidence_path = data_dir / "locomo10_evidence.json"
    qa_path = data_dir / "locomo10_qa.json"
    
    if not evidence_path.exists() or not qa_path.exists():
        print("Error: content files not found. Please run scripts/extract_data.py first.")
        return

    print("Loading data...")
    evidence_data = load_json(evidence_path)
    qa_data = load_json(qa_path)
    
    # Process only the first N samples for demonstration
    samples_to_process = evidence_data[:args.limit]
    
    for sample in samples_to_process:
        sample_id = sample['sample_id']
        print(f"\nProcessing Sample: {sample_id}")
        
        # 1. Ingestion (Chunking + Embedding)
        print("  - Chunking conversation...")
        chunks = chunk_conversation(sample['conversation'])
        print(f"  - Generated {len(chunks)} chunks.")
        
        print(f"  - embedding chunks using models/gemini-embedding-001 (this may take a moment)...")
        chunk_embeddings = []
        for chunk in tqdm(chunks):
            chunk_embeddings.append(get_embedding(chunk))
        
        chunk_embeddings = np.array(chunk_embeddings)
        
        # 2. Retrieval & Generation
        # Find corresponding QA
        qa_sample = next((item for item in qa_data if item["sample_id"] == sample_id), None)
        if not qa_sample:
            print("  - No matching QA found.")
            continue
            
        questions = qa_sample['qa'][:3] # Demo: take first 3 questions
        
        for q_item in questions:
            query = q_item['question']
            print(f"\n  Q: {query}")
            
            # Embed query
            query_embedding = np.array(get_embedding(query))
            
            # Cosine similarity
            similarities = np.dot(chunk_embeddings, query_embedding) / (
                np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Top K
            k = 3
            top_indices = np.argsort(similarities)[-k:][::-1]
            
            retrieved_context = ""
            for idx in top_indices:
                retrieved_context += f"---\n{chunks[idx]}\n"
            
            # Generate Answer
            prompt = f"""Based on the retrieved validation, answer the question.
            
Context:
{retrieved_context}

Question: {query}
Answer:"""
            
            answer = get_completion([{"role": "user", "content": prompt}])
            print(f"  A (RAG): {answer}")
            print(f"  A (Ref): {q_item['answer']}")

if __name__ == "__main__":
    main()
