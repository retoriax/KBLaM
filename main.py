from fastapi import FastAPI
from pydantic import BaseModel
import torch
import uvicorn
import time
import json
import os
import argparse
import numpy as np
from src.kblam.utils.eval_utils import (
    _format_Q_llama,
    _format_Q_phi3,
    _prune_for_llama,
    _prune_for_phi3,
)
from experiments.eval import _prepare_models, KBRetriever, KBEncoder

def get_args():
    parser = argparse.ArgumentParser(description="OpenAI-kompatibles KB-Modell mit FastAPI")

    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing the dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset JSON file")
    parser.add_argument("--encoder_spec", type=str, required=True, help="Encoder model spec")
    parser.add_argument("--encoder_dir", type=str, required=True, help="Path to encoder weights")
    parser.add_argument("--llm_base_dir", type=str, required=True, help="Path to base LLM directory")
    parser.add_argument("--llm_type", type=str, required=True, choices=["llama3", "phi3"], help="LLM type")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to trained model weights")
    parser.add_argument("--precomputed_embed_keys_name", type=str, required=False, help="Name to precomputed embedding keys (.npy)")
    parser.add_argument("--precomputed_embed_values_name", type=str, required=False, help="Name to precomputed embedding values (.npy)")
    parser.add_argument("--query_head_path", type=str, required=False, default=None, help="Path to query head weights (optional)")
    return parser.parse_args()

args = get_args()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0
    max_tokens: int = 4000

class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

app = FastAPI()

# ---------- Globale Model-Objekte ----------
class ModelState:
    tokenizer = None
    encoder = None
    model = None
    kb_config = None
    kb_retriever = None
    kb_embedding = None

state = ModelState()

@app.on_event("startup")
def load_model():
    global state

    # === Schritt 1: Modelle vorbereiten ===
    state.tokenizer, state.encoder, state.model, state.kb_config = _prepare_models(
        args.encoder_spec,
        args.encoder_dir,
        args.llm_type,
        args.llm_base_dir,
        args.model_dir,
        args.query_head_path,
        3,
        1,
    )

    # === Schritt 2: Daten laden ===
    dataset = json.load(open(os.path.join(args.dataset_dir, args.dataset)))

    # === Schritt 3: Precomputed Embeddings laden und kürzen ===
    kb_retriever = KBRetriever(
        state.encoder,
        dataset,
        precomputed_embed_keys_path=os.path.join(args.dataset_dir, args.precomputed_embed_keys_name),
        precomputed_embed_values_path=os.path.join(args.dataset_dir, args.precomputed_embed_values_name),
    )

    # === Schritt 4: KB anbinden ===
    kb_idx = list(range(len(dataset)))
    state.kb_embedding = kb_retriever.get_key_embeddings(kb_idx)



@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """
    OpenAI-kompatible Chat-Completion-API.
    Nimmt das letzte User-Message-Feld für die Modell-Eingabe.
    """
    
    Q = ""
    for m in reversed(req.messages):
        if m.role == "user":
            Q = m.content
            break
    if not Q:
        return {"error": "No user message found."}


    format_func_map = {"llama3": _format_Q_llama, "phi3": _format_Q_phi3}

    input_strs = [format_func_map[args.llm_type](Q)]
        
    tokenizer_output = state.tokenizer(input_strs, return_tensors="pt", padding=True).to(
        "cuda"
    )
    input_ids, attention_masks = (
        tokenizer_output["input_ids"],
        tokenizer_output["attention_mask"],
    )


    with torch.autograd.no_grad():
        outputs = state.model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            kb_kvs=state.kb_embedding,
            max_new_tokens=500,
            do_sample=False,
            tokenizer=state.tokenizer,
            output_attentions=True,
            save_attention_weights=False,
            pad_token_id=state.tokenizer.eos_token_id,
            kb_config=state.kb_config,
            attention_save_loc=None,
            attention_file_base_name=None,
        ) 
        outputs = state.tokenizer.batch_decode(outputs.squeeze(), skip_special_tokens=False)

    print("\nFrage:", Q)
    prune_func_map = {"llama3": _prune_for_llama, "phi3": _prune_for_phi3}

    output = prune_func_map[args.llm_type](''.join(outputs)).split(Q)[1]        
    
    message = ChatMessage(
        role="assistant",
        content=output
    )

    choice = Choice(
        index=0,
        message=message,
        finish_reason="stop"
    )

    response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [choice.dict()]
    }

    print("Antwort:", output)
    return response

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8087, reload=False)