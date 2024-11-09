# Evaluating counter factual dataset
# Setting: Randomly shuffle key and values of a KB, such that the content
# of the KB becomes very counter factual

import argparse
import json
import os
import re
import textwrap

import evaluate

from kblam.models.kblam_config import KBLaMConfig
import nltk
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, logging


from kblam.utils.data_utils import aug_row, generate_multi_entity_qa, get_i_dont_know_ans
from kblam.kb_encoder import KBEncoder

from kblam.utils.train_utils import get_kb_embd
from kblam.models.llama_model import KblamLlamaForCausalLM
from kblam.models.phi3_model import KBLaMPhi3ForCausalLM
from typing import List, Dict, Optional
from pathlib import Path

nltk.download('wordnet')
logging.set_verbosity_warning()

rouge = evaluate.load('rouge')
bert_score = evaluate.load('bertscore')

instruction_prompts = '''
Please answer questions based on the given text with format: "The {property} of {name} is {description}"
'''

instruction_prompts_multi_entities = '''
Please answer questions based on the given text with format: "The {property} of {name1} is {description}; The {property} of {name2} is {description}; ..."
'''

zero_shot_prompt = '''
Please answer the question in a very compact manner with format: The {property} of {name} is {description}
'''

zero_shot_prompt_multi_entities = '''
Please answer the question in a very compact manner with format: "The {property} of {name1} is {description}; The {property} of {name2} is {description}; ...
'''


def _format_Q_llama(Q: str):
    return (
        "<|start_header_id|>user<|end_header_id|> " + Q + "<|eot_id|>" + "<|start_header_id|>assistant<|end_header_id|>"
    )


def _format_Q_phi3(Q: str):
    return "<|user|>\n" + Q + "<|end|>\n" + "<|assistant|>\n"


model_question_format_mapping = {KblamLlamaForCausalLM: _format_Q_llama, KBLaMPhi3ForCausalLM: _format_Q_phi3}


def _prune_for_llama(S):
    S = S.replace('<|eot_id|>', '')
    S = S.replace('<|start_header_id|>assistant<|end_header_id|>', '')
    S = S.replace('<|start_header_id|>user<|end_header_id|>', '')
    S = S.replace('<|end_of_text|>', '')
    return S


def _prune_for_phi3(S):
    S = S.replace('<|end|>', '')
    S = S.replace('<|assistant|>', '')
    S = S.replace('<|user|>', '')
    return S


model_prune_format_mapping = {KblamLlamaForCausalLM: _prune_for_llama, KBLaMPhi3ForCausalLM: _prune_for_phi3}


def answer_question(tokenizer, model, Q, kb=None, kb_layer_frequency=3, topk_size=None, kb_scale_factor=-1):
    for m in model_question_format_mapping:
        if isinstance(model, m):
            input_str = model_question_format_mapping[m](Q)
    print(input_str)
    tokenizer_output = tokenizer(input_str, return_tensors='pt', padding=True).to('cuda')
    input_ids, attention_masks = tokenizer_output['input_ids'], tokenizer_output['attention_mask']

    with torch.autograd.no_grad():
        if topk_size != -1:
            dynamic_sparsify = True
        else:
            dynamic_sparsify = False
        if kb_scale_factor == -1:
            kb_scale_factor = None

        kb_config = KBLaMConfig(
            sep_query_head=True,
            kb_scale_factor=kb_scale_factor,
            top_k_kb=topk_size,
            dynamic_sparsify=dynamic_sparsify,
            kb_layer_frequency=kb_layer_frequency,
        )

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            kb_kvs=kb,
            max_new_tokens=150,
            tokenizer=tokenizer,
            output_attentions=True,
            kb_config=kb_config,
        ).squeeze()
    outputs = tokenizer.decode(outputs, skip_special_tokens=False)

    # pruned_output = model_prune_format_mapping[model](outputs)
    for m in model_prune_format_mapping:
        if isinstance(model, m):
            pruned_output = model_prune_format_mapping[m](outputs)
    return pruned_output


class KBRetriever:
    def __init__(
        self,
        encoder: KBEncoder,
        dataset: List[Dict],
        key_embds: Optional[np.ndarray],
        value_embds: Optional[np.ndarray],
    ):
        self.encoder = encoder
        self.key_embds = key_embds
        self.value_embds = value_embds
        self.dataset = dataset

    def _use_cached_embd(self):
        if self.key_embds is not None and self.value_embds is not None:
            return True
        else:
            return False

    def get_key_embeddings(self, batch_indices):
        if self._use_cached_embd():
            return get_kb_embd(
                self.encoder,
                batch_indices,
                precomputed_embd=(self.key_embds, self.value_embds),
            )
        else:
            return get_kb_embd(self.encoder, batch_indices, kb_dict=self.dataset)


def perform_eval(
    model,
    tokenizer,
    kb_retriever,
    encoder_model_spec,
    eval_mode='kb',
    kb_layer_frequency=3,
    kb_size=250,
    seed=1,
    precomputed_kb=None,
    topk_size=-1,
    kb_scale_factor=-1,
    multi_entites=-1,
    remove_sorry=False,
):
    np.random.seed(seed)
    kb_idx = np.random.randint(0, len(kb_retriever.dataset), kb_size)
    test_kb = [kb_retriever.dataset[idx] for idx in kb_idx]
    kb_embedding = ()
    key_str = [row['key_string'] for row in test_kb]
    value_str = [row['description'] for row in test_kb]
    prompt_strs = ''
    for k, v in zip(key_str, value_str):
        prompt_strs += f'{k} is {v}; '
    kv_pairs = (key_str, value_str)

    kb_embedding = kb_retriever.get_key_embeddings(kb_idx)

    model_outputs = []
    answers = []
    full_outputs = []
    # answer_question
    subset_size = min(
        400, len(test_kb)
    )  # Regardless of KB size, always test 250 questions, otherwise it will be too slow
    subset_size = min(
        400, len(test_kb)
    )  # Regardless of KB size, always test 250 questions, otherwise it will be too slow
    # subset_size = 50
    for row in tqdm(test_kb[:subset_size]):
        if multi_entites == -1:
            Q = row['Q']
            answer = row['A']
        else:
            kb_subset_idx = np.random.randint(0, len(test_kb), multi_entites)
            Q, A = generate_multi_entity_qa(
                [test_kb[i]['name'] for i in kb_subset_idx],
                [test_kb[i]['description_type'] for i in kb_subset_idx],
                [test_kb[i]['description'] for i in kb_subset_idx],
            )
            answer = A

        if eval_mode == 'kb':
            model_output = answer_question(
                tokenizer,
                model,
                Q,
                kb=kb_embedding,
                kb_layer_frequency=kb_layer_frequency,
                topk_size=topk_size,
                kb_scale_factor=kb_scale_factor,
            ).split(Q)[1]
        elif eval_mode == 'icl':
            if multi_entites != -1:
                ins_prompt = instruction_prompts_multi_entities
            else:
                ins_prompt = instruction_prompts
            model_output = answer_question(
                tokenizer, model, ins_prompt + prompt_strs + Q, kb=None, kb_layer_frequency=kb_layer_frequency
            ).split(Q)[1]
        elif eval_mode == 'zeroshot':
            if multi_entites != -1:
                ins_prompt = zero_shot_prompt_multi_entities
            else:
                ins_prompt = zero_shot_prompt
            model_output = answer_question(
                tokenizer, model, ins_prompt + Q, kb=None, kb_layer_frequency=kb_layer_frequency
            ).split(Q)[1]
        # print(model_output)
        if remove_sorry:
            if 'sorry' in model_output:
                continue
        full_outputs.append((model_output, answer))
        if multi_entites == -1:
            pattern = r'The\s+\w+\s+of\s+[^"]+\s+is\s+(.+)'
            match = re.search(pattern, model_output)
            answers.append(row['description'])
            if match:
                model_output = match.group(1)
        else:
            pattern = r'(?:is|are) (.*?)(?:\.|;)'
            matches = re.findall(pattern, model_output)
            model_output = '; '.join(matches)
            answers.append(';'.join(re.findall(r'(?:is|are) (.*?);', answer)))
        model_outputs.append(model_output)

    print(f'KB size: {kb_size}, mode: {eval_mode}')
    rouge = evaluate.load('rouge')

    for pred, gt in zip(model_outputs, answers):
        print(f"PREDICTION: {pred}")
        print(f"GT: {gt}")
    rogue_score = rouge.compute(predictions=model_outputs, references=answers)
    print(rogue_score)
    bertscore = bert_score.compute(
        predictions=model_outputs, references=answers, lang="en", model_type='microsoft/deberta-xlarge-mnli'
    )
    bert_scores = []
    for k, v in bertscore.items():
        if isinstance(v, list):
            bert_scores.append(np.mean(v))
            print(k, np.mean(v))
    results = ''
    for a, A in full_outputs:
        results += f'Model output: {a}\nTrue answer: {A}\n-------\n'
    if eval_mode == 'kb':
        eval_mode = encoder_model_spec + eval_mode
    return results, bert_scores + list(rogue_score.values())


parser = argparse.ArgumentParser(description="Evaluation script")

# Add arguments that will be shared across all subcommands
parent_parser = argparse.ArgumentParser(add_help=False)

parent_parser.add_argument('--seed', type=int)
parent_parser.add_argument('--dataset_dir', type=str)
parent_parser.add_argument('--test_dataset', type=str)  # Source of test KB, assume it is a KV pair
parent_parser.add_argument('--llm_base_dir', type=str)
parent_parser.add_argument('--model_dir', type=str)
parent_parser.add_argument('--encoder_dir', type=str)
parent_parser.add_argument('--save_dir', type=str)
parent_parser.add_argument('--kb_layer_frequency', type=int, default=3)
parent_parser.add_argument('--ckpt_idx', type=int, default=10000)
parent_parser.add_argument('--lr', type=float, default=0.0005)
parent_parser.add_argument('--kb_size', type=int, default=200)
parent_parser.add_argument('--fancy_instruction', action=argparse.BooleanOptionalAction, default=False)
parent_parser.add_argument('--encoder_spec', type=str, default='OAI')
parent_parser.add_argument('--kb_scale_factor', type=int, default=None)
# parser.add_argument('--train_dataset_name', type=str, default='gpt_data')
parent_parser.add_argument("--llm_type", type=str, default="phi3", choices=["llama3", "phi3"])

# Create subparsers
subparsers = parser.add_subparsers(dest='command', required=True)

# Create the parser for the generation command
gen_parser = subparsers.add_parser('generation', parents=[parent_parser], help='Evaluate generation')


gen_parser.add_argument('--use_precomputed_embd', action='store_true', default=False)
gen_parser.add_argument('--topk_size', type=int, default=-1)
gen_parser.add_argument('--multi_entites', type=int, default=-1)
gen_parser.add_argument(
    '--no_outlier', action=argparse.BooleanOptionalAction, default=False
)  # Use ckpts trained without outlier
# Evaluation modes:
# - kb: Append KB tokens in the front
# - icl: Flatten the KB as string as prompts
# - zeroshot: No prompts, answer the questions with LLM's imagination
gen_parser.add_argument('--eval_mode', type=str, choices=['kb', 'icl', 'zeroshot'], default='kb')
gen_parser.add_argument(
    '--remove_sorry', action=argparse.BooleanOptionalAction, default=False
)  # Filter out sorry answer in the output
gen_parser.add_argument(
    "--kb_token_layer_frequency", type=int, default=None, help="Introduce QA with extended open-ended parts"
)

# Create the parser for the generation command
acc_parser = subparsers.add_parser('accuracy', parents=[parent_parser], help='Evaluate accuracy')

acc_parser.add_argument("--attn_save_dir", type=str, default="")  # Where should the attention mask saved?
acc_parser.add_argument("--log_save_dir", type=str)  # Where is the accuracy results be saved?
acc_parser.add_argument("--fancy_question", action=argparse.BooleanOptionalAction, default=False)
acc_parser.add_argument("--use_shift_match", action=argparse.BooleanOptionalAction, default=False)


def eval_generate():
    args = parser.parse_args()
    seed = args.seed
    dataset_dir = args.dataset_dir
    test_dataset = args.test_dataset
    llm_base_dir = args.llm_base_dir
    model_path = args.model_dir
    encoder_path = args.encoder_dir
    kb_layer_frequency = args.kb_layer_frequency
    ckpt_idx = args.ckpt_idx
    eval_mode = args.eval_mode
    kb_size = args.kb_size
    fancy_instruction = args.fancy_instruction
    use_precomputed_embd = args.use_precomputed_embd
    kb_scale_factor = args.kb_scale_factor
    remove_sorry = args.remove_sorry
    no_outlier = args.no_outlier
    llm_type = args.llm_type
    actual_kb_token_layer_frequency = args.kb_token_layer_frequency

    validation_part_start_idx = 120000 if 'gpt' in test_dataset else 0

    dataset = json.load(open(os.path.join(dataset_dir, test_dataset + '.json')))[validation_part_start_idx:]

    encoder_model_spec = args.encoder_spec
    lr = args.lr

    precomputed_embd = None
    key_embds = None
    value_embds = None
    if use_precomputed_embd:
        key_embds = np.load(os.path.join(dataset_dir, f'{test_dataset}_{encoder_model_spec}_embd_key.npy')).astype(
            'float32'
        )[validation_part_start_idx:]
        value_embds = np.load(os.path.join(dataset_dir, f'{test_dataset}_{encoder_model_spec}_embd_value.npy')).astype(
            'float32'
        )[validation_part_start_idx:]
        precomputed_embd = (key_embds, value_embds)

    kb_layer_frequency = kb_layer_frequency
    encoder_spec = encoder_model_spec
    tokenizer = AutoTokenizer.from_pretrained(llm_base_dir, trust_remote_code=True, padding_side='left')
    tokenizer.pad_token = '^'

    if llm_type == "llama3":
        model = KblamLlamaForCausalLM.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
        )
    else:
        model = KBLaMPhi3ForCausalLM.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
        )

    model.generation_config.pad_token_id = tokenizer.pad_token_id
    # model.generation_config.eos_token_id = 128009
    model.eval()

    encoder = KBEncoder(
        encoder_name="OAI",
        projector_type="linear",
        endpoint_url="",
        out_dim=model.config.hidden_size  # type: ignore
        * (model.config.num_hidden_layers // actual_kb_token_layer_frequency + 1),  # type: ignore
        frozen_base_model=True,
        device=torch.device("cuda"),
    )
    encoder.load_state_dict(torch.load(encoder_path))

    kb_retriever = KBRetriever(
        encoder,
        dataset,
        key_embds=key_embds,
        value_embds=value_embds,
    )

    gen_results, score_output = perform_eval(
        model,
        tokenizer,
        kb_retriever,
        encoder_model_spec,
        eval_mode,
        kb_layer_frequency,
        seed=args.seed,
        kb_size=kb_size,
        precomputed_kb=precomputed_embd,
        topk_size=args.topk_size,
        multi_entites=args.multi_entites,
        kb_scale_factor=kb_scale_factor,
    )
    mem_cost = torch.cuda.max_memory_reserved('cuda')
    score_output.append(mem_cost)

    exp_config = (
        f'lr_{lr}_kblayer_frequency_{kb_layer_frequency}_encoder_{encoder_spec}_ckptidx_{ckpt_idx}'
        f'testset_{test_dataset}_kbsize_{kb_size}_evalmode_{eval_mode}_seed_{seed}'
    )

    if args.multi_entites != -1:
        exp_config = f'MultiEntites_{args.multi_entites}' + exp_config

    if fancy_instruction:
        exp_config = 'FancyInstruction_' + exp_config

    if args.topk_size != -1:
        exp_config = f'TopK_{args.topk_size}' + exp_config

    if kb_scale_factor != -1:
        exp_config = f'ScaleFactor_{kb_scale_factor}' + exp_config

    if remove_sorry:
        exp_config = f'RemoveSorry_' + exp_config

    if no_outlier:
        exp_config = f'NoOutlier_' + exp_config

    (Path(args.save_dir) / exp_config).mkdir(exist_ok=True, parents=True)
    np.save(os.path.join(args.save_dir, exp_config), np.array(score_output))
    text_file = open(os.path.join(args.save_dir, exp_config + '.txt'), "w")
    text_file.write(gen_results)


def eval_accuracy():

    TEST_BATCH_SIZE = 50

    args = parser.parse_args()
    seed = args.seed
    llm_base_dir = args.llm_base_dir
    dataset_dir = args.dataset_dir
    test_dataset = args.test_dataset
    kb_layer_frequency = args.kb_layer_frequency
    ckpt_idx = args.ckpt_idx
    train_kb_size = args.train_kb_size
    kb_size = args.kb_size
    fancy_instruction = args.fancy_instruction
    fancy_question = args.fancy_question
    kb_scale_factor = args.kb_scale_factor
    encoder_spec = args.encoder_spec
    use_shift_match = args.use_shift_match
    llm_type = llm_type = args.llm_type

    model_path = args.model_dir
    encoder_path = args.encoder_dir

    if kb_scale_factor == -1:
        kb_scale_factor = None

    validation_part_start_idx = 120000 if "gpt" in test_dataset else 0

    dataset = json.load(open(os.path.join(dataset_dir, test_dataset) + ".json"))[validation_part_start_idx:]
    encoder_model_spec = encoder_spec

    sm_string = "" if not use_shift_match else "_sm"

    key_embds = np.load(
        os.path.join(dataset_dir, f"{test_dataset}_{encoder_model_spec}_embd_key{sm_string}.npy")
    ).astype("float32")[validation_part_start_idx:]
    value_embds = np.load(
        os.path.join(dataset_dir, f"{test_dataset}_{encoder_model_spec}_embd_value{sm_string}.npy")
    ).astype("float32")[validation_part_start_idx:]

    # train_dataset_name = args.train_dataset_name
    # lr = args.lr

    if args.kb_layer_frequency == -1:
        kb_layer_frequency = 3
    # config_name = (
    #     f"traindataset_{train_dataset_name}_testdataset_{test_dataset}_"
    #     f"kb_layer_freq_{kb_layer_frequency}_trainkbsize_{train_kb_size}"
    #     f"encoder_{encoder_model_spec}_step_{ckpt_idx}_kbsize_{kb_size}_seed_{seed}"
    # )

    # if fancy_question:
    #     config_name = "UseFancyQuestion_" + config_name

    # if fancy_instruction:
    #     config_name = "UseFancyIns_" + config_name

    # if kb_scale_factor is not None:
    #     config_name = f"ScaleFactor_{kb_scale_factor}" + config_name

    # if use_shift_match:
    #     config_name = f"UseSM_" + config_name

    # if fancy_instruction:
    #     extended_qa_spec = "UseExtendedQA"
    #     outlier_spec = "UseOutlier1"
    #     multi_entity_string = "MultiEntities2"
    # else:
    #     extended_qa_spec = ""
    #     outlier_spec = ""
    #     multi_entity_string = ""

    # kb_size_spec = f"KBSize{train_kb_size}"
    # duplicate_spec = "NoDuplicate"

    kb_layer_frequency = kb_layer_frequency
    encoder_spec = encoder_spec
    tokenizer = AutoTokenizer.from_pretrained(llm_base_dir, trust_remote_code=True, padding_side="left")
    tokenizer.pad_token = "^"

    if llm_type == "llama3":
        model = KblamLlamaForCausalLM.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
        )
    else:
        model = KBLaMPhi3ForCausalLM.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
        )

    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = 128009
    model.eval()
    encoder = KBEncoder(
        encoder_name=encoder_spec.upper(),
        projector_type="linear",
        endpoint_url="",
        out_dim=model.config.hidden_size * (model.config.num_hidden_layers // kb_layer_frequency + 1),
        frozen_base_model=True,
        projector_kwargs={"mlp_depth": 1, "mlp_hidden_dim": 512},
        device=torch.device("cuda"),
    )

    encoder.load_state_dict(torch.load(encoder_path))

    print("******")
    dataset_subset_idx = np.random.choice(len(dataset), kb_size, replace=False)
    # dataset_subset_idx = np.random.choice(np.arange(100000, len(dataset)), subset_size, replace=False)
    dataset_subset = [dataset[i] for i in dataset_subset_idx]
    encoder.eval()
    with torch.autograd.no_grad():
        random_index = np.random.choice(len(dataset), kb_size, replace=False)
        # kb_embedding_random = get_kb_embd(encoder, random_index, precomputed_embd=(key_embds, value_embds))
        kb_embedding_real = get_kb_embd(encoder, dataset_subset_idx, precomputed_embd=(key_embds, value_embds))
        kb_embedding_key, kb_embedding_val = kb_embedding_real
        kb_embedding_real = (kb_embedding_key, kb_embedding_val)

    format_func_map = {"llama3": _format_Q_llama, "phi3": _format_Q_phi3}

    config_name = "random_config_name"  # TODO:fix

    N = kb_size
    if not fancy_question:
        input_strs_gen = (dataset_subset[i]["Q"] for i in range(TEST_BATCH_SIZE))
    else:
        input_strs_gen = (aug_row(dataset_subset[i]) for i in range(TEST_BATCH_SIZE))
    input_strs = [format_func_map[llm_type](ex) for ex in input_strs_gen]

    tokenizer_output = tokenizer(input_strs, return_tensors="pt", padding=True).to("cuda")
    input_ids, attention_masks = (
        tokenizer_output["input_ids"],
        tokenizer_output["attention_mask"],
    )
    kb_embedding_real = (kb_embedding_real[0], kb_embedding_real[1])
    print(kb_embedding_real[0].shape)

    kb_config = KBLaMConfig(
        sep_query_head=True,
        kb_layer_frequency=kb_layer_frequency,
        kb_scale_factor=kb_scale_factor,
    )
    print(f"starting {type(model)} {args.attn_save_dir} {config_name}")
    with torch.autograd.no_grad():
        outputs_true_kb = model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            kb_kvs=kb_embedding_real,
            max_new_tokens=60,
            tokenizer=tokenizer,
            output_attentions=True,
            kb_config=kb_config,
            save_attention_weights=True,
            attention_save_loc=args.attn_save_dir,
            attention_file_base_name=config_name,
        )

    ranges = range(0, 32, kb_layer_frequency)

    save_dir = args.log_save_dir
    accs, confidences = [], []
    for idx in ranges:
        weights = []
        weight = np.load(os.path.join(args.attn_save_dir, f"{config_name}_{idx}.npy"))[..., :kb_size]
        label = np.arange(TEST_BATCH_SIZE)
        weight = weight.reshape(TEST_BATCH_SIZE, -1, kb_size)
        acc = (weight.sum(1).argmax(1) == label).mean()
        top_5_predictions = torch.topk(torch.from_numpy(weight.sum(1)), 5, dim=1)[1]
        top_5_acc = (top_5_predictions.numpy() == label[:, None]).any(1).mean()
        accs.append((acc, top_5_acc))
        # confidence = softmax(weight.mean(1), -1).max()
        # confidences.append(confidence)
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)

    np.save(save_path / f"{config_name}_acc.npy", np.array(accs))


def main():
    args = parser.parse_args()
    print(args)
    if args.command == 'generate':
        eval_generate()
    elif args.command == 'accuracy':
        eval_accuracy()
    else:
        raise ValueError(f"command {args.command} not recognised")


if __name__ == "__main__":
    main()
