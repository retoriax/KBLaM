import argparse
import json
import os
import random
import re
import time
import itertools
import sys

from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from dotenv import load_dotenv
from datasets import load_dataset

import threading
from dataclasses import dataclass

from wikipedia_data_prompts import PROMPT_LANGUAGE_MAP, construct_prompts, get_key_string
@dataclass
class DataPoint:
    name: str
    description_type: str
    description: str
    Q: str = None
    A: str = None
    key_string: str = None
    extended_Q: str = None
    extended_A: str = None

def postprocess_llm_output(text: str) -> str:
    text = text.strip()
    return text[0].lower() + text[1:] if text and text[0].isupper() else text

def clean_json_str(text):
    # Removes ```json, ``` and leading/trailing newlines/whitespace
    return re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()

def infer_all_description_types(llm, text, language):
    prompt = PROMPT_LANGUAGE_MAP[language]["description_types"] + text[:1500]
    output = call_chat(llm, prompt, prompt_type="description_types", language=language)
    try:
        parsed = json.loads(clean_json_str(output))
        # Filter fields that are not meaningful
        for k in ["description", "objective", "purpose"]:
            if k in parsed and (not parsed[k] or parsed[k].strip().lower() in ["not applicable", "n/a", "", "nicht anwendbar", "keine angabe", "k.a.", "n.v.", "-"]):
                del parsed[k]
        return parsed
    except Exception as e:
        print(f"Failed to parse types: {output} — {e}")
        return {}
    
    
def call_chat(llm: ChatOpenAI, prompt: str, prompt_type: str, language: str) -> str:
    messages = [
        SystemMessage(content=PROMPT_LANGUAGE_MAP[language]["system"]),
        HumanMessage(content=prompt)
    ]
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages).content.strip()
            #print(f"[{prompt_type}] LLM response: {response}")
            return postprocess_llm_output(response)
        except Exception as e:
            if attempt > 0:
                print(f"[{prompt_type}] LLM error on attempt {attempt+1}: {type(e).__name__} – {e}")
            time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"Failed to get response from LLM after {max_retries} retries for prompt type '{prompt_type}'.")


def generate_extended_QA(llm: ChatOpenAI, datapoint: DataPoint, language: str, full_text: str) -> None:
    # Check if description is "not applicable", "n/a", or typical meta/list phrase
    _desc = (datapoint.description or datapoint.A or "").strip().lower()
    skip_phrases = [
        "not applicable", "n/a", "see also", "list of", "this article", "overview of", "table of contents",
        "nicht anwendbar", "keine angabe", "k.a.", "n.v.", "-"
    ]
    if any(_desc.startswith(phrase) for phrase in skip_phrases):
        # fallback: extended_Q/A = Q/A, never leave empty
        datapoint.extended_Q = datapoint.Q
        datapoint.extended_A = datapoint.A
        return

    prompt = PROMPT_LANGUAGE_MAP[language]["extended_qa"].format(Q=datapoint.Q, A=datapoint.description, TEXT=full_text[:1500])
    output = call_chat(llm, prompt, prompt_type="extended", language=language)

    # Try to parse JSON output first
    try:
        parsed_json = json.loads(clean_json_str(output))
        ext_q = parsed_json.get("extended_Q", "").strip()
        ext_a = parsed_json.get("extended_A", "").strip()
        if not ext_q or not ext_a or ext_q.lower() in ["not applicable", "n/a", "nicht anwendbar", "keine angabe", "k.a.", "n.v.", "-"] or ext_a.lower() in ["not applicable", "n/a", "nicht anwendbar", "keine angabe", "k.a.", "n.v.", "-"]:
            datapoint.extended_Q = datapoint.Q
            datapoint.extended_A = datapoint.A
            return
        datapoint.extended_Q = ext_q
        datapoint.extended_A = ext_a
    except Exception:
        # Attempt to fix missing quotes around extended_Q and extended_A fields
        try:
            # Find the JSON object with extended_Q and extended_A keys but missing quotes on values
            pattern = r'{\s*"extended_Q"\s*:\s*([^",\n]+|"[^"]*")\s*,\s*"extended_A"\s*:\s*([^",\n]+|"[^"]*")\s*}'
            match = re.search(pattern, output, re.DOTALL)
            if match:
                ext_q_val = match.group(1).strip()
                ext_a_val = match.group(2).strip()
                # Add quotes if missing
                if not (ext_q_val.startswith('"') and ext_q_val.endswith('"')):
                    ext_q_val = f'"{ext_q_val}"'
                if not (ext_a_val.startswith('"') and ext_a_val.endswith('"')):
                    ext_a_val = f'"{ext_a_val}"'
                fixed_json_str = f'{{"extended_Q": {ext_q_val}, "extended_A": {ext_a_val}}}'
                parsed_json = json.loads(fixed_json_str)
                ext_q = parsed_json.get("extended_Q", "").strip()
                ext_a = parsed_json.get("extended_A", "").strip()
                if not ext_q or not ext_a or ext_q.lower() in ["not applicable", "n/a", "nicht anwendbar", "keine angabe", "k.a.", "n.v.", "-"] or ext_a.lower() in ["not applicable", "n/a", "nicht anwendbar", "keine angabe", "k.a.", "n.v.", "-"]:
                    datapoint.extended_Q = datapoint.Q
                    datapoint.extended_A = datapoint.A
                    return
                datapoint.extended_Q = ext_q
                datapoint.extended_A = ext_a
                return
        except Exception:
            pass

        # fallback to regex parsing with warning
        match = re.search(r"[qQ]\s*:\s*(.*?)\n[aA]\s*:\s*(.*)", output, re.DOTALL)
        if match:
            datapoint.extended_Q = match.group(1).strip()
            datapoint.extended_A = match.group(2).strip()
        else:
            print(f"[Warning] Extended QA format not found for {datapoint.name}: {output}")
            datapoint.extended_Q = datapoint.Q
            datapoint.extended_A = datapoint.A

    # After LLM call: If extended_Q or extended_A is "not applicable", "n/a", or empty, fallback to Q/A
    _extq = (datapoint.extended_Q or "").strip().lower()
    _exta = (datapoint.extended_A or "").strip().lower()
    _fallback_phrases = ["not applicable.", "not applicable", "n/a", "", "nicht anwendbar", "keine angabe", "k.a.", "n.v.", "-"]
    if _extq in _fallback_phrases or _exta in _fallback_phrases:
        datapoint.extended_Q = datapoint.Q
        datapoint.extended_A = datapoint.A


def process_entity(idx, ds, seen_keys, llm, lock, language):
    entry = ds[idx]
    name = entry["title"]
    full_text = entry["text"]
    # 4. Skip too short entities
    if not full_text or len(full_text.split()) < 8:
        print(f"Skipped entity '{name}' due to empty or short description.")
        return None

    try:
        type_contents = infer_all_description_types(llm, full_text, language)
        descriptions = {}
        for k in ["description", "objective", "purpose"]:
            with lock:
                if k in type_contents and get_key_string(name, k, language) not in seen_keys:
                    descriptions[k] = type_contents[k]
    except Exception as e:
        print(f"Type and content detection failed for entry '{name}':", e)
        return None

    results = []
    for dtype, content in descriptions.items():
        dp = DataPoint(name=name, description_type=dtype, description=content)
        dp.Q, dp.A, dp.key_string = construct_prompts(dp, language)
        with lock:
            seen_keys.add(dp.key_string)
        generate_extended_QA(llm, dp, language, full_text)
        results.append(dp)
        
    #print(f"Processed entity '{name}' with {len(results)} data points.")    
    return results


def load_existing_entries(path: str) -> tuple[list[DataPoint], set[str]]:
    dataset = []
    seen_keys = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    seen_keys.add(entry["key_string"])
                    dataset.append(DataPoint(**entry))
                except Exception as e:
                    print(f"Error loading existing entry: {e}")
    return dataset, seen_keys


def append_datapoint_to_tmpfile(datapoint: DataPoint, tmp_path: str):
    with open(tmp_path, "a", encoding="utf-8") as f:
        json.dump(datapoint.__dict__, f, ensure_ascii=False)
        f.write("\n")


def consolidate_tmp_to_final(tmp_path: str, final_path: str) -> list[dict]:
    final_dataset = []
    with open(tmp_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                final_dataset.append(entry)
            except Exception as e:
                print(f"Error reading entry from temporary dataset: {e}")

    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, indent=2, ensure_ascii=False)
    return final_dataset


def generate_dataset(name_dataset, size, output_path, llm, max_workers=8, language="en"):
    os.makedirs(output_path, exist_ok=True)
    ds = load_dataset("wikimedia/wikipedia", f"20231101.{language}")["train"]
    existing_path = os.path.join(output_path, f"{name_dataset}.tmp.json")
    dataset, seen_keys = load_existing_entries(existing_path)
    print(f"Loaded {len(dataset)} existing entries.")

    lock = threading.Lock()

    shuffled_indices = itertools.cycle(random.sample(range(len(ds)), len(ds)))

    to_generate = size - len(dataset)
    if to_generate <= 0:
        print(f"Dataset already contains {len(dataset)} entries, which meets or exceeds the requested size ({size}). Nothing to generate.")
        temp_path = os.path.join(output_path, f"{name_dataset}.tmp.json")
        final_path = os.path.join(output_path, f"{name_dataset}.json")
        if not os.path.exists(final_path):
            print("Final dataset does not exist yet. Consolidating temporary file to final dataset...")
            final_dataset = consolidate_tmp_to_final(temp_path, final_path)
            print(f"Final dataset written to {final_path} with {len(final_dataset)} entries.")
        else:
            print(f"Final dataset already exists at {final_path}.")
        return

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for _ in range(to_generate):
            idx = next(shuffled_indices)
            futures.append(executor.submit(
                process_entity, idx, ds, seen_keys, llm, lock, language
            ))

        completed = 0
        for future in as_completed(futures):
            result = future.result()
            if result:
                with lock:
                    for dp in result:
                        dataset.append(dp)
                        append_datapoint_to_tmpfile(dp, os.path.join(output_path, f"{name_dataset}.tmp.json"))
                    completed += len(result)
                    if completed % 50 == 0 or completed == to_generate:
                        elapsed = time.time() - start_time
                        it_per_sec = completed / elapsed if elapsed > 0 else 0
                        eta = (to_generate - completed) / it_per_sec if it_per_sec > 0 else 0
                        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
                        percent = completed / to_generate * 100 if to_generate > 0 else 0
                        print(f"Generate: {completed}/{to_generate} ({percent:.1f}%) [{elapsed_str}<{eta_str}, {it_per_sec:.2f}it/s]")

    print(f"Saved {len(dataset)} data points to {output_path}/{name_dataset}.tmp.json")
    print("Consolidating final dataset...")
    temp_path = os.path.join(output_path, f"{name_dataset}.tmp.json")
    final_path = os.path.join(output_path, f"{name_dataset}.json")
    final_dataset = consolidate_tmp_to_final(temp_path, final_path)

    print(f"Final dataset written to {final_path} with {len(final_dataset)} entries.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1000, help="Number of examples to generate")
    parser.add_argument("--output_path", type=str, default="dataset", help="Output directory")
    parser.add_argument("--worker", type=int, default=1, help="How many workers to use for generation")
    parser.add_argument("--dataset_name", type=str, default="wiki_dataset", help="Name of the dataset to generate")
    parser.add_argument("--language", type=str, default="en", choices=["en", "de"], help="Prompt language: 'en' (default) or 'de'")

    args = parser.parse_args()

    load_dotenv()
    llm = ChatOpenAI(
        model_name=os.getenv("WIKIDATA_MODEL_NAME"),
        openai_api_base=os.getenv("WIKIDATA_PATH"),
        openai_api_key=os.getenv("WIKIDATA_API_KEY")
    )

    generate_dataset(
        args.dataset_name,
        args.size,
        args.output_path,
        llm,
        max_workers=args.worker,
        language=args.language
    )