import torch
import json
import numpy as np
from pathlib import Path
from src.kblam.utils.eval_utils import (
    _format_Q_llama,
    _format_Q_phi3,
    _prune_for_llama,
    _prune_for_phi3
)

from experiments.eval import _prepare_models, KBRetriever
import os

# === Konfiguration ===
DATASET_DIR = "/home/julian_sammet/Documents/repos/KBLaM/datasets/train_test_split"
DATASET_FILE = "train_synthetic.json"
EMBED_KEYS_FILE = "train_synthetic_all-MiniLM-L6-v2_embd_key.npy"
EMBED_VALUES_FILE = "train_synthetic_all-MiniLM-L6-v2_embd_value.npy"
ENCODER_PATH = "/home/julian_sammet/Documents/datadisk/experiments/kblam/exp_v0.0.13/stage1_lr_0.0001KBTokenLayerFreq3UseOutlier1KBSize200SepQueryHeadUseDataAugKeyFromkey_all-MiniLM-L6-v2_train_synthetic_llama3_step_1000_encoder/encoder.pt"
ENCODER_SPEC = "all-MiniLM-L6-v2"
LLM_TYPE = "llama3"  # oder "phi3"
LLM_BASE_DIR = "/home/julian_sammet/Documents/datadisk/tk/Llama-3.2-1B-Instruct"
MODEL_PATH = "/home/julian_sammet/Documents/datadisk/experiments/kblam/exp_v0.0.16/stage1_lr_0.0001KBTokenLayerFreq3UseOutlier1SepQueryHeadUseDataAugKeyFromkey_BigOAI_synthetic_llama3_step_4000"  # enthält safetensors mit query head
#QUERY_HEAD = "/home/julian_sammet/Documents/datadisk/experiments/kblam/exp_v0.0.13/stage1_lr_0.0001KBTokenLayerFreq3UseOutlier1KBSize200SepQueryHeadUseDataAugKeyFromkey_all-MiniLM-L6-v2_train_synthetic_llama3_step_1000/query_head.pth"
KB_LAYER_FREQUENCY = 3
KB_SCALE_FACTOR = None

# === Schritt 1: Modelle vorbereiten ===
tokenizer, encoder, model, kb_config = _prepare_models(
    ENCODER_SPEC,
    ENCODER_PATH,
    LLM_TYPE,
    LLM_BASE_DIR,
    MODEL_PATH,
    None,
    KB_LAYER_FREQUENCY,
    None,
)

# === Schritt 2: Daten laden ===
dataset = json.load(open(os.path.join(DATASET_DIR, DATASET_FILE)))

# === Schritt 3: Precomputed Embeddings laden und kürzen ===
kb_retriever = KBRetriever(
    encoder,
    dataset,
    precomputed_embed_keys_path=os.path.join(DATASET_DIR, EMBED_KEYS_FILE),
    precomputed_embed_values_path=os.path.join(DATASET_DIR, EMBED_VALUES_FILE),
)

# === Schritt 4: KB anbinden ===
kb_idx = list(range(len(dataset)))
#kb_idx = list(range(len(dataset)))
kb_embedding = kb_retriever.get_key_embeddings(kb_idx)

# === Schritt 5: Frage stellen ===
Q = "What is the description of Seraphine Moonshadow"
format_func_map = {"llama3": _format_Q_llama, "phi3": _format_Q_phi3}

input_strs = [format_func_map[LLM_TYPE](Q)]
    
tokenizer_output = tokenizer(input_strs, return_tensors="pt", padding=True).to(
    "cuda"
)
input_ids, attention_masks = (
    tokenizer_output["input_ids"],
    tokenizer_output["attention_mask"],
)


with torch.autograd.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_masks,
        kb_kvs=kb_embedding,
        max_new_tokens=60,
        tokenizer=tokenizer,
        output_attentions=True,
        save_attention_weights=False,
        kb_config=kb_config,
        attention_save_loc="",
        attention_file_base_name="",
    )
    outputs = tokenizer.batch_decode(outputs.squeeze(), skip_special_tokens=False)

print("\nFrage:", Q)
prune_func_map = {"llama3": _prune_for_llama, "phi3": _prune_for_phi3}

print("Antwort:", prune_func_map[LLM_TYPE](''.join(outputs)).split(Q)[1])