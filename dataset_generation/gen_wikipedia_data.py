import argparse
import json
import os
import random
import re
import time
import itertools
import sys

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from dotenv import load_dotenv
from datasets import load_dataset

import threading
from dataclasses import dataclass

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


def construct_prompts(data: DataPoint) -> tuple[str, str, str]:
    Q = f"What is the {data.description_type} of {data.name}?"
    A = f"The {data.description_type} of {data.name} is {data.description}"
    key_string = f"the {data.description_type} of {data.name}"
    return Q, A, key_string

def postprocess_llm_output(text: str) -> str:
    text = text.strip()
    return text[0].lower() + text[1:] if text and text[0].isupper() else text

def clean_json_str(text):
    # Removes ```json, ``` and leading/trailing newlines/whitespace
    return re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()

def infer_all_description_types(llm, text):
    prompt = '''
    For the following text, output a JSON object with the following keys (include only those that make sense):
    - description: a concise, specific, factual noun-phrase summary (what is it?)
    - objective: the practical function or goal (if not a person, place, event, season, list, or disambiguation)
    - purpose: the broader human or societal intention or value (if present)
    For subjects like persons, places, events, lists, or disambiguations, set 'objective' and/or 'purpose' to "Not applicable" if they don't make sense.
    Example:
    {
      "description": "a quarterly DVD magazine published by McSweeney’s featuring short films and documentaries that had limited theatrical release",
      "objective": "to provide a curated selection of rare films for film enthusiasts",
      "purpose": "to make independent and obscure films accessible to a wider audience"
    }
    Text:
    ''' + text[:3000]
    output = call_chat(llm, prompt, prompt_type="description_types")
    try:
        parsed = json.loads(clean_json_str(output))
        # Filter fields that are not meaningful
        for k in ["description", "objective", "purpose"]:
            if k in parsed and (not parsed[k] or parsed[k].strip().lower() in ["not applicable", "n/a", ""]):
                del parsed[k]
        return parsed
    except Exception as e:
        print(f"Failed to parse types: {output} — {e}")
        return {}
    
    
def call_chat(llm: ChatOpenAI, prompt: str, prompt_type: str) -> str:
    messages = [
        SystemMessage(content=(
            "You are an expert language model tasked with generating high-quality, structured knowledge base entries. Each response must:\n"
            "- Start directly with the appropriate structure:\n"
            "  - For 'description': generate a concise but informative summary of 1–2 sentences. It should include category, notable facts, time period, location, and 1–2 specific distinguishing features if available. Do not write a long paragraph or add redundant explanations.\n"
            "  - For 'objective': use a precise verb phrase describing the functional goal (e.g., 'to document the evolution of...').\n"
            "  - For 'purpose': use a verb phrase capturing the broader intention or benefit (e.g., 'to inform readers about...').\n"
            "- Do NOT repeat or rephrase the question.\n"
            "- Avoid filler, generic descriptions, or meta-commentary.\n"
            "- Be as detailed and specific as possible while remaining concise.\n"
        )),
        HumanMessage(content=prompt)
    ]
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages).content.strip()
            if response.lower().startswith(f"the {prompt_type}".lower()):
                response = response[len(f"the {prompt_type}"):].lstrip(" :.-").strip()
            return postprocess_llm_output(response)
        except Exception as e:
            if attempt > 0:
                print(f"[{prompt_type}] LLM error on attempt {attempt+1}: {type(e).__name__} – {e}")
            time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"Failed to get response from LLM after {max_retries} retries for prompt type '{prompt_type}'.")


def generate_extended_QA(llm: ChatOpenAI, datapoint: DataPoint) -> None:
    # Check if description is "not applicable", "n/a", or typical meta/list phrase
    _desc = (datapoint.description or datapoint.A or "").strip().lower()
    skip_phrases = [
        "not applicable", "n/a", "see also", "list of", "this article", "overview of", "table of contents"
    ]
    if any(_desc.startswith(phrase) for phrase in skip_phrases):
        # fallback: extended_Q/A = Q/A, never leave empty
        datapoint.extended_Q = datapoint.Q
        datapoint.extended_A = datapoint.A
        return

    prompt = (
        "Rewrite the following Q&A for a curious but knowledgeable user. "
        "The answer (extended_A) should be at most 2–3 sentences, focus on unique or noteworthy details not already in the original description, "
        "and avoid unnecessary repetition or general information. Be concise and relevant.\n"
        "Return the result as a JSON object with keys \"extended_Q\" and \"extended_A\".\n"
        'Both "extended_Q" and "extended_A" must be valid JSON strings enclosed in double quotes. Only output valid JSON.\n\n'
        f"Q: {datapoint.Q}\nA: {datapoint.description}"
    )
    output = call_chat(llm, prompt, prompt_type="extended")

    # Try to parse JSON output first
    try:
        parsed_json = json.loads(clean_json_str(output))
        ext_q = parsed_json.get("extended_Q", "").strip()
        ext_a = parsed_json.get("extended_A", "").strip()
        if not ext_q or not ext_a or ext_q.lower() in ["not applicable", "n/a"] or ext_a.lower() in ["not applicable", "n/a"]:
            datapoint.extended_Q = datapoint.Q
            datapoint.extended_A = datapoint.A
            return
        datapoint.extended_Q = ext_q
        datapoint.extended_A = ext_a
        # Only append if not already redundant
        if not datapoint.extended_A.lower().startswith(f"the {datapoint.description_type}".lower()):
            datapoint.extended_A = f"The {datapoint.description_type} of {datapoint.name} is {datapoint.extended_A}"
        else:
            # Only insert name if not included
            if datapoint.name not in datapoint.extended_A:
                datapoint.extended_A = f"The {datapoint.description_type} of {datapoint.name} is {datapoint.extended_A}"
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
                if not ext_q or not ext_a or ext_q.lower() in ["not applicable", "n/a"] or ext_a.lower() in ["not applicable", "n/a"]:
                    datapoint.extended_Q = datapoint.Q
                    datapoint.extended_A = datapoint.A
                    return
                datapoint.extended_Q = ext_q
                datapoint.extended_A = ext_a
                # Only append if not already redundant
                if not datapoint.extended_A.lower().startswith(f"the {datapoint.description_type}".lower()):
                    datapoint.extended_A = f"The {datapoint.description_type} of {datapoint.name} is {datapoint.extended_A}"
                else:
                    # Only insert name if not included
                    if datapoint.name not in datapoint.extended_A:
                        datapoint.extended_A = f"The {datapoint.description_type} of {datapoint.name} is {datapoint.extended_A}"
                return
        except Exception:
            pass

        # fallback to regex parsing with warning
        match = re.search(r"[qQ]\s*:\s*(.*?)\n[aA]\s*:\s*(.*)", output, re.DOTALL)
        if match:
            datapoint.extended_Q = match.group(1).strip()
            datapoint.extended_A = match.group(2).strip()
            # Only append if not already redundant
            if not datapoint.extended_A.lower().startswith(f"the {datapoint.description_type}".lower()):
                datapoint.extended_A = f"The {datapoint.description_type} of {datapoint.name} is {datapoint.extended_A}"
            else:
                # Only insert name if not included
                if datapoint.name not in datapoint.extended_A:
                    datapoint.extended_A = f"The {datapoint.description_type} of {datapoint.name} is {datapoint.extended_A}"
        else:
            print(f"[Warning] Extended QA format not found for {datapoint.name}: {output}")
            datapoint.extended_Q = datapoint.Q
            datapoint.extended_A = datapoint.A

    # After LLM call: If extended_Q or extended_A is "not applicable", "n/a", or empty, fallback to Q/A
    _extq = (datapoint.extended_Q or "").strip().lower()
    _exta = (datapoint.extended_A or "").strip().lower()
    _fallback_phrases = ["not applicable.", "not applicable", "n/a", ""]
    if _extq in _fallback_phrases or _exta in _fallback_phrases:
        datapoint.extended_Q = datapoint.Q
        datapoint.extended_A = datapoint.A


def process_entity(idx, ds, seen_keys, llm, lock):
    entry = ds[idx]
    name = entry["title"]
    full_text = entry["text"]
    # 4. Skip too short entities
    if not full_text or len(full_text.split()) < 8:
        print(f"Skipped entity '{name}' due to empty or short description.")
        return None

    try:
        type_contents = infer_all_description_types(llm, full_text)
        descriptions = {}
        for k in ["description", "objective", "purpose"]:
            key_str = f"the {k} of {name}"
            with lock:
                if k in type_contents and key_str not in seen_keys:
                    descriptions[k] = type_contents[k]
    except Exception as e:
        print(f"Type and content detection failed for entry '{name}':", e)
        return None

    results = []
    for dtype, content in descriptions.items():
        dp = DataPoint(name=name, description_type=dtype, description=content)
        dp.Q, dp.A, dp.key_string = construct_prompts(dp)
        with lock:
            seen_keys.add(dp.key_string)
        generate_extended_QA(llm, dp)
        results.append(dp)
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


def generate_dataset(name_dataset, size, output_path, llm, max_workers=8):
    os.makedirs(output_path, exist_ok=True)
    ds = load_dataset("wikimedia/wikipedia", "20231101.en")["train"]
    existing_path = os.path.join(output_path, "realworld_dataset.tmp.json")
    dataset, seen_keys = load_existing_entries(existing_path)
    print(f"Loaded {len(dataset)} existing entries.")

    lock = threading.Lock()

    shuffled_indices = itertools.cycle(random.sample(range(len(ds)), len(ds)))

    to_generate = size - len(dataset)
    if to_generate <= 0:
        print(f"Dataset already contains {len(dataset)} entries, which meets or exceeds the requested size ({size}). Nothing to generate.")
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        with tqdm(total=size, desc="Generating data", initial=len(dataset), file=sys.stdout, disable=not sys.stdout.isatty()) as pbar:
            for _ in range(to_generate):
                idx = next(shuffled_indices)
                futures.append(executor.submit(
                    process_entity, idx, ds, seen_keys, llm, lock
                ))

            for future in as_completed(futures):
                result = future.result()
                if result:
                    for dp in result:
                        dataset.append(dp)
                        append_datapoint_to_tmpfile(dp, os.path.join(output_path, f"{name_dataset}.tmp.json"))
                        pbar.update(1)

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
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--dataset_name", type=str, default="realworld_dataset", help="Name of the dataset to generate")

    args = parser.parse_args()

    load_dotenv()
    llm_main = ChatOpenAI(
        model_name=args.model_name,
        openai_api_base=os.getenv("PROXY_PATH"),
        openai_api_key=os.getenv("PROXY_API_KEY")
    )

    generate_dataset(
        args.dataset_name,
        args.size,
        args.output_path,
        llm_main,
        max_workers=2
    )