#!/usr/bin/env python3
import os
import argparse
import json
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings
from kblam.utils.data_utils import DataPoint
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import time

def parser_args():
    parser = argparse.ArgumentParser(description="Generate KB embeddings via SBERT or OpenAIEmbeddings proxy")
    parser.add_argument(
        "--model_name",
        type=str,
        default="text-embedding-3-large",
        choices=["all-MiniLM-L6-v2", "text-embedding-3-large", "ada-embeddings", "e5-mistral-7b-instruct"],
        help="Which model to use for embeddings"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="synthetic_data_ger",
        help="Name of the dataset (used in output filenames)"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset JSON file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="dataset",
        help="Directory to write output .npy files"
    )
    return parser.parse_args()

def compute_embeddings(encoder_model_spec: str, texts: list[str], batch_size: int = 100) -> np.ndarray:
    model = SentenceTransformer(encoder_model_spec, device="cuda")
    embeddings = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        emb = model.encode(chunk, convert_to_numpy=True)
        embeddings.append(emb)
    return np.concatenate(embeddings, axis=0)

if __name__ == "__main__":
    load_dotenv()
    args = parser_args()

    # Load dataset
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    dataset = [DataPoint(**item) for item in raw]

    # Prepare texts
    key_texts = [dp.key_string for dp in dataset]
    val_texts = [dp.description for dp in dataset]

    # Compute embeddings
    if args.model_name == "all-MiniLM-L6-v2":
        print("🔄 Embedding keys with SBERT...")
        key_embeds = []
        for chunk in tqdm([key_texts[i:i+100] for i in range(0, len(key_texts), 100)], desc="SBERT key chunks"):
            key_embeds.append(SentenceTransformer(args.model_name, device="cuda").encode(chunk, convert_to_numpy=True))
        key_embeds = np.concatenate(key_embeds, axis=0)

        print("🔄 Embedding values with SBERT...")
        value_embeds = []
        for chunk in tqdm([val_texts[i:i+100] for i in range(0, len(val_texts), 100)], desc="SBERT value chunks"):
            value_embeds.append(SentenceTransformer(args.model_name, device="cuda").encode(chunk, convert_to_numpy=True))
        value_embeds = np.concatenate(value_embeds, axis=0)
    elif args.model_name == "e5-mistral-7b-instruct":
        print("🔄 Embedding via LiteLLM API (e5-mistral-7b-instruct)...")

        def request_litellm_embedding_batch(batch_inputs: list[str], max_retries: int = 3, backoff: float = 1.5) -> list[list[float]]:
            request_body = {
                "model": args.model_name,
                "input": batch_inputs
            }
            headers = {"Authorization": f"Bearer {str(os.getenv('PROXY_EMBEDDING_API_KEY'))}"}
            url = f"{os.getenv('PROXY_EMBEDDING_PATH')}/v1/embeddings"

            for attempt in range(1, max_retries + 1):
                try:
                    response = requests.post(url, json=request_body, headers=headers, timeout=30)
                    response.raise_for_status()
                    return [entry["embedding"] for entry in response.json()["data"]]
                except Exception as e:
                    print(f"⚠️ Retry {attempt}/{max_retries} failed for batch with error: {e}")
                    if attempt == max_retries:
                        raise
                    time.sleep(backoff * attempt)

        def embed_all_litellm(texts, batch_size=20, max_workers=8):
            results = [None] * len(texts)
            batches = [(i, texts[i:i+batch_size]) for i in range(0, len(texts), batch_size)]

            def embed_and_store(start_idx, batch):
                try:
                    embeddings = request_litellm_embedding_batch(batch)
                    return (start_idx, embeddings)
                except Exception as e:
                    print(f"❌ Failed to embed batch starting at {start_idx}: {[repr(t) for t in batch]}\n{e}")
                    return (start_idx, [[0.0]*1024]*len(batch))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(embed_and_store, start, batch): start for start, batch in batches}
                for future in tqdm(as_completed(futures), total=len(futures), desc="LiteLLM embedding"):
                    start_idx, batch_embeds = future.result()
                    for i, emb in enumerate(batch_embeds):
                        results[start_idx + i] = emb

            return np.array(results)

        key_embeds = embed_all_litellm(key_texts)
        value_embeds = embed_all_litellm(val_texts)
    else:
        embedder = OpenAIEmbeddings(
            model=args.model_name,
            openai_api_key=os.getenv("PROXY_EMBEDDING_API_KEY"),
            openai_api_base=os.getenv("PROXY_EMBEDDING_PATH")
        )
        embedder.chunk_size = 10  # smaller chunks for stability

        def embed_chunk(chunk):
            return embedder.embed_documents(chunk)

        # Embed keys with threading
        key_chunks = [key_texts[i:i+embedder.chunk_size] for i in range(0, len(key_texts), embedder.chunk_size)]
        print("🔄 Embedding keys (OpenAI, threaded)...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(embed_chunk, chunk) for chunk in key_chunks]
            key_embeds = []
            for future in tqdm(as_completed(futures), total=len(futures), desc="Embedding keys"):
                key_embeds.extend(future.result())
        key_embeds = np.array(key_embeds)

        # Embed values with threading
        val_chunks = [val_texts[i:i+embedder.chunk_size] for i in range(0, len(val_texts), embedder.chunk_size)]
        print("🔄 Embedding values (OpenAI, threaded)...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(embed_chunk, chunk) for chunk in val_chunks]
            value_embeds = []
            for future in tqdm(as_completed(futures), total=len(futures), desc="Embedding values"):
                value_embeds.extend(future.result())
        value_embeds = np.array(value_embeds)

    # Determine save name
    save_name = (
        "all-MiniLM-L6-v2" if args.model_name == "all-MiniLM-L6-v2"
        else "OAI" if args.model_name == "ada-embeddings"
        else "BigOAI" if args.model_name == "text-embedding-3-large"
        else "e5-mistral-7b-instruct"
    )

    # Ensure output dir
    os.makedirs(args.output_path, exist_ok=True)

    # Save arrays
    np.save(f"{args.output_path}/{args.dataset_name}_{save_name}_embd_key.npy", key_embeds)
    np.save(f"{args.output_path}/{args.dataset_name}_{save_name}_embd_value.npy", value_embeds)

    print(f"✅ Saved key embeddings to {args.output_path}/{args.dataset_name}_{save_name}_embd_key.npy")
    print(f"✅ Saved value embeddings to {args.output_path}/{args.dataset_name}_{save_name}_embd_value.npy")