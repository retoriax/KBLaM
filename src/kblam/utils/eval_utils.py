from typing import Optional

import numpy as np
import torch
import transformers

from src.kblam.models.kblam_config import KBLaMConfig
from src.kblam.models.llama3_model import KblamLlamaForCausalLM
from src.kblam.models.phi3_model import KBLaMPhi3ForCausalLM

instruction_prompts = """
Please answer questions based on the given text with format: "The {property} of {name} is {description}"
"""

instruction_prompts_multi_entities = """
Please answer questions based on the given text with format: "The {property} of {name1} is {description}; The {property} of {name2} is {description}; ..."
"""

zero_shot_prompt = """
Please answer the question in a very compact manner with format: The {property} of {name} is {description}
"""

zero_shot_prompt_multi_entities = """
Please answer the question in a very compact manner with format: "The {property} of {name1} is {description}; The {property} of {name2} is {description}; ...
"""


def _prune_for_llama(S: str) -> str:
    S = S.replace("<|eot_id|>", "")
    S = S.replace("<|start_header_id|>assistant<|end_header_id|>", "")
    S = S.replace("<|start_header_id|>user<|end_header_id|>", "")
    S = S.replace("<|end_of_text|>", "")
    return S


def _prune_for_phi3(S: str) -> str:
    S = S.replace("<|end|>", "")
    S = S.replace("<|assistant|>", "")
    S = S.replace("<|user|>", "")
    return S


def softmax(x: np.array, axis: int) -> np.array:
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis)


def _format_Q_llama(Q: str):
    return (
        "<|start_header_id|>user<|end_header_id|> " + Q + "<|eot_id|>" + "<|start_header_id|>assistant<|end_header_id|>"
    )


def _format_Q_phi3(Q: str):
    return "<|user|>\n" + Q + "<|end|>\n" + "<|assistant|>\n"


model_question_format_mapping = {
    KblamLlamaForCausalLM: _format_Q_llama,
    KBLaMPhi3ForCausalLM: _format_Q_phi3,
}
model_prune_format_mapping = {
    KblamLlamaForCausalLM: _prune_for_llama,
    KBLaMPhi3ForCausalLM: _prune_for_phi3,
}


def answer_question(
    tokenizer: transformers.PreTrainedTokenizer,
    model: KBLaMPhi3ForCausalLM | KblamLlamaForCausalLM,
    Q: str,
    kb=None,
    kb_config: Optional[KBLaMConfig] = None,
    attention_save_loc: Optional[str] = None,
    save_attention_weights: bool = False,
    attention_file_base_name: Optional[str] = None,
):
    model_type = type(model)
    if model_type not in model_question_format_mapping:
        raise ValueError(f"Unbekannter Modelltyp: {model_type}")
    input_str = model_question_format_mapping[model_type](Q)

    tokenizer_output = tokenizer(input_str, return_tensors="pt", padding=True).to("cuda")
    input_ids, attention_masks = (
        tokenizer_output["input_ids"],
        tokenizer_output["attention_mask"],
    )

    with torch.autograd.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            kb_kvs=kb,
            max_new_tokens=150,
            do_sample=False,
            tokenizer=tokenizer,
            output_attentions=False,
            kb_config=kb_config,
            pad_token_id=tokenizer.eos_token_id,
            save_attention_weights=save_attention_weights,
            attention_file_base_name=attention_file_base_name,
            attention_save_loc=attention_save_loc,
        ).squeeze()
    outputs = tokenizer.decode(outputs, skip_special_tokens=False)

    model_type = type(model)
    if model_type not in model_prune_format_mapping:
        raise ValueError(f"Unbekannter Modelltyp für Pruning: {model_type}")
    pruned_output = model_prune_format_mapping[model_type](outputs)
    return pruned_output