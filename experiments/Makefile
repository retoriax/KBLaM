.DEFAULT_GOAL := help # Sets default action to be help


define PRINT_HELP_PYSCRIPT # start of Python section
import re, sys

output = []
# Loop through the lines in this file
for line in sys.stdin:
    # if the line has a command and a comment start with
    #   two pound signs, add it to the output
    match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
    if match:
        target, help = match.groups()
        output.append("%-10s %s" % (target, help))
# Sort the output in alphanumeric order
output.sort()
# Print the help result
print('\n'.join(output))
endef
export PRINT_HELP_PYSCRIPT # End of python section

LLAMA_BASE_DIR='/home/julian_sammet/Documents/datadisk/tk/Llama-3.2-1B-Instruct'
LLAMA_HF='/kblam_llama_unified/kblam_unified'
PHI3_HF='/datadisk/data/gcrbackup/hf_models/phi3'
DATASET_DIR='/datadisk/data/gcrbackup/oai_embd'
TEST_DATASET_DIR='/home/julian_sammet/Documents/datadisk/data/datasets/BigOAI'
CKPT_SAVE_DIR='/home/msalvaris/data/experiments/kblam/exp_v0.3.2'
PHI3_MODEL_CHKPT='/home/julian_sammet/Documents/datadisk/data/gcrbackup/experiments/kblam/exp_v0.1/stage1_lr_0.0001KBTokenLayerFreq3UseExtendedQAMultiEntities2UseOutlier1NoDuplicateKBSizedynamicSepQueryHeadUseDataAugKeyFromkey_OAI_synthetic_data_phi3_step_20000'
ENCODER_FOR_PHI3_CHKPT='/home/julian_sammet/Documents/datadisk/data/gcrbackup/experiments/kblam/exp_v0.1/stage1_lr_0.0001KBTokenLayerFreq3UseExtendedQAMultiEntities2UseOutlier1NoDuplicateKBSizedynamicSepQueryHeadUseDataAugFineTuneQueryKeyFromkey_synthetic_data_OAI_step_20000'
LLAMA_MODEL_CHKPT='/home/julian_sammet/Documents/datadisk/experiments/kblam/exp_v0.0.16/stage1_lr_0.0001KBTokenLayerFreq3UseOutlier1SepQueryHeadUseDataAugKeyFromkey_BigOAI_synthetic_llama3_step_4000' 
ENCODER_FOR_LLAMA_CHKPT='/home/julian_sammet/Documents/datadisk/experiments/kblam/exp_v0.0.16/stage1_lr_0.0001KBTokenLayerFreq3UseOutlier1SepQueryHeadUseDataAugKeyFromkey_BigOAI_synthetic_llama3_step_4000_encoder/encoder.pt'
QUERY_HEAD_PATH='/datadisk/tk/learned_query_head_20000_OAI.pth'

ATTN_SAVE_DIR='/home/julian_sammet/Documents/datadisk/kblamatt2'
LOG_SAVE_DIR='/datadisk/kblamatt2/acc_results'


LR=1e-4
TRAIN_KB_SIZE=0 # Randomly pick KB size during training time
OUTLIER_RATIO=-1 # Ratio of no-answer question in the batch, -1 stands for no such samples
KB_LAYER_FREQ=3 # How frequent to inject kb tokens into the layers
MULTI_ENTITIES_NUM=2 # For questions that involve multiple entities, how many entities are involved.


help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

create_train_test_split:
	python ../dataset_generation/create_train_test_split.py --data_path /home/julian_sammet/Documents/repos/KBLaM/datasets/synthetic.json \
															--embedding_keys_path /home/julian_sammet/Documents/datadisk/data/datasets/all-MiniLM-L6-v2/synthetic_all-MiniLM-L6-v2_embd_key.npy \
															--embeddings_values_path /home/julian_sammet/Documents/datadisk/data/datasets/all-MiniLM-L6-v2/synthetic_all-MiniLM-L6-v2_embd_value.npy \
															--output_path ../datasets \
															--split_index 2000


train-julian_sammet: ## Train kb adapter julian_sammet
	python train.py --dataset ../datasets \ 
						--B 1 \
						--gradient_accm_step 400 \
						--llm_type llama3 \
						--hf_model_spec "DiscoResearch/Llama3-DiscoLeo-Instruct-8B-v0.1" \
						--use_cached_embd \
						--train_dataset synthetic_data_ger \
						--N 120000 \
						--total_steps 20001 \
						--encoder_spec all-MiniLM-L6-v2 \
						--key_embd_src key \
						--use_data_aug 

train: ## Train kb adapter
	python train.py --model_dir ${LLAMA_BASE_DIR} \
					   --dataset_dir ${DATASET_DIR} \
					   --model_save_dir ${CKPT_SAVE_DIR} \
					   --seed 1607 \
					   --dataset gpt_data \
					   --N 120000 \
					   --B 20 \
					   --steps 20001 \
					   --encoder_spec all-MiniLM-L6-v2 \
					   --use_cached_embd \
					   --key_embd_src key \
					   --use_data_aug \
					   --use_lr_decay \
					   --tune_llm_q \
					   --sep_query_head \
					   --lr ${LR} \
					   --kb_size ${TRAIN_KB_SIZE} \
					   --kb_token_layer_frequency ${KB_LAYER_FREQ} \
					   --multi_entities ${MULTI_ENTITIES_NUM} \
					   --use_extended_qa \
					   --outlier_ratio ${OUTLIER_RATIO} \
					   --gradient_accm_step 20

train-oai: ## Train kb adapter
	python train.py --model_dir ${LLAMA_HF} \
					   --dataset_dir ${DATASET_DIR} \
					   --model_save_dir ${CKPT_SAVE_DIR} \
					   --seed 1607 \
					   --dataset gpt_data \
					   --N 120000 \
					   --B 2 \
					   --steps 30001 \
					   --encoder_spec OAI \
					   --use_cached_embd \
					   --key_embd_src key \
					   --use_data_aug \
					   --use_lr_decay \
					   --tune_llm_q \
					   --sep_query_head \
					   --lr ${LR} \
					   --no-duplicate_true_kb \
					   --kb_size ${TRAIN_KB_SIZE} \
					   --kb_token_layer_frequency ${KB_LAYER_FREQ} \
					   --multi_entities ${MULTI_ENTITIES_NUM} \
					   --use_extended_qa \
					   --outlier_ratio ${OUTLIER_RATIO} \
					   --gradient_accm_step 20




train-phi3-oai: ## Train kb adapter
	python train.py --model_dir ${PHI3_HF} \
					--dataset_dir ${DATASET_DIR} \
					--model_save_dir ${CKPT_SAVE_DIR} \
					--seed 1607 \
					--train_dataset synthetic_data \
					--N 120000 \
					--B 32 \
					--steps 30001 \
					--encoder_spec OAI \
					--use_cached_embd \
					--key_embd_src key \
					--use_data_aug \
					--use_lr_decay \
					--tune_llm_q \
					--sep_query_head \
					--lr ${LR} \
					--no-duplicate_true_kb \
					--kb_size ${TRAIN_KB_SIZE} \
					--kb_token_layer_frequency ${KB_LAYER_FREQ} \
					--multi_entities ${MULTI_ENTITIES_NUM} \
					--use_extended_qa \
					--outlier_ratio ${OUTLIER_RATIO} \
					--gradient_accm_step 20 \
					--llm_type 'phi3' \
					--use_cuda


#----------------- phi eval --------------------------------------


eval-acc-phi3-oai: ## Eval kb adapter
	python eval.py accuracy \
		--seed 1607 \
		--dataset_dir ${TEST_DATASET_DIR} \
		--test_dataset test_synthetic_data.json \
		--llm_base_dir ${PHI3_HF} \
		--model_dir ${PHI3_MODEL_CHKPT} \
		--encoder_dir ${ENCODER_FOR_PHI3_CHKPT} \
		--save_dir ${ATTN_SAVE_DIR} \
		--kb_layer_frequency ${KB_LAYER_FREQ} \
		--kb_size 3200 \
		--kb_scale_factor 100 \
		--no-fancy_instruction \
		--encoder_spec oai \
		--llm_type "phi3" \
		--attn_save_dir ${ATTN_SAVE_DIR} \
		--log_save_dir ${LOG_SAVE_DIR} \
		--precomputed_embed_keys_path ${TEST_DATASET_DIR}/test_synthetic_data_oai_embd_key.npy \
		--precomputed_embed_values_path ${TEST_DATASET_DIR}/test_synthetic_data_oai_embd_value.npy


eval-acc-eval-phi3-oai: ## Eval kb adapter
	python eval.py acc_results \
		--seed 1607 \
		--dataset_dir ${TEST_DATASET_DIR} \
		--test_dataset test_synthetic_data.json \
		--llm_base_dir ${PHI3_HF} \
		--model_dir ${PHI3_MODEL_CHKPT} \
		--encoder_dir ${ENCODER_FOR_PHI3_CHKPT} \
		--save_dir ${ATTN_SAVE_DIR} \
		--kb_layer_frequency ${KB_LAYER_FREQ} \
		--kb_size 3200 \
		--kb_scale_factor 100 \
		--no-fancy_instruction \
		--encoder_spec oai \
		--llm_type "phi3" \
		--attn_save_dir ${ATTN_SAVE_DIR} \
		--log_save_dir ${LOG_SAVE_DIR} \
		--precomputed_embed_keys_path ${TEST_DATASET_DIR}/test_synthetic_data_oai_embd_key.npy \
		--precomputed_embed_values_path ${TEST_DATASET_DIR}/test_synthetic_data_oai_embd_value.npy


eval-gen-phi3-oai: ## Eval kb adapter
	python eval.py generation \
		--seed 1607 \
		--dataset_dir ${TEST_DATASET_DIR} \
		--test_dataset test_synthetic_data.json \
		--llm_base_dir ${PHI3_HF} \
		--model_dir ${PHI3_MODEL_CHKPT} \
		--encoder_dir ${ENCODER_FOR_PHI3_CHKPT} \
		--save_dir ${ATTN_SAVE_DIR} \
		--kb_layer_frequency ${KB_LAYER_FREQ} \
		--kb_size 3200 \
		--kb_scale_factor 100 \
		--no-fancy_instruction \
		--encoder_spec oai \
		--llm_type "phi3" \
		--precomputed_embed_keys_path ${TEST_DATASET_DIR}/test_synthetic_data_oai_embd_key.npy \
		--precomputed_embed_values_path ${TEST_DATASET_DIR}/test_synthetic_data_oai_embd_value.npy


eval-ref-phi3-oai: ## Eval kb adapter
	python eval.py refusal \
		--seed 1607 \
		--dataset_dir ${TEST_DATASET_DIR} \
		--test_dataset test_synthetic_data.json \
		--llm_base_dir ${PHI3_HF} \
		--model_dir ${PHI3_MODEL_CHKPT} \
		--encoder_dir ${ENCODER_FOR_PHI3_CHKPT} \
		--save_dir ${ATTN_SAVE_DIR} \
		--kb_layer_frequency ${KB_LAYER_FREQ} \
		--kb_size 3200 \
		--kb_scale_factor 100 \
		--no-fancy_instruction \
		--encoder_spec oai \
		--llm_type "phi3" \
		--precomputed_embed_keys_path ${TEST_DATASET_DIR}/test_synthetic_data_oai_embd_key.npy \
		--precomputed_embed_values_path ${TEST_DATASET_DIR}/test_synthetic_data_oai_embd_value.npy



eval-basic-phi3-oai: ## Eval kb adapter
	python eval.py standard \
		--seed 1607 \
		--dataset_dir ${TEST_DATASET_DIR} \
		--test_dataset test_synthetic_data.json \
		--llm_base_dir ${PHI3_HF} \
		--model_dir ${PHI3_MODEL_CHKPT} \
		--encoder_dir ${ENCODER_FOR_PHI3_CHKPT} \
		--save_dir ${ATTN_SAVE_DIR} \
		--kb_layer_frequency ${KB_LAYER_FREQ} \
		--kb_size 3200 \
		--kb_scale_factor 100 \
		--no-fancy_instruction \
		--encoder_spec oai \
		--llm_type "phi3" \
		--precomputed_embed_keys_path ${TEST_DATASET_DIR}/test_synthetic_data_oai_embd_key.npy \
		--precomputed_embed_values_path ${TEST_DATASET_DIR}/test_synthetic_data_oai_embd_value.npy



#----------------- llama eval --------------------------------------

eval-acc-llama: ## Eval kb adapter
	python eval.py standard \
		--seed 1 \
		--dataset_dir /home/julian_sammet/Documents/repos/KBLaM/datasets \
		--test_dataset train_synthetic.json \
		--llm_base_dir /home/julian_sammet/Documents/datadisk/tk/Meta-Llama-3-8B-Instruct \
		--model_dir /home/julian_sammet/Documents/datadisk/experiments/kblam/exp_v0.0.15/stage1_lr_0.0001KBTokenLayerFreq3UseOutlier1SepQueryHeadUseDataAugKeyFromkey_all-MiniLM-L6-v2_synthetic_llama3_step_1500 \
		--encoder_dir /home/julian_sammet/Documents/datadisk/experiments/kblam/exp_v0.0.15/stage1_lr_0.0001KBTokenLayerFreq3UseOutlier1SepQueryHeadUseDataAugKeyFromkey_all-MiniLM-L6-v2_synthetic_llama3_step_1500_encoder/encoder.pt \
		--save_dir /home/julian_sammet/Documents/datadisk/kblamatt2 \
		--kb_layer_frequency 3 \
		--kb_size 200 \
		--kb_scale_factor 1 \
		--no-fancy_instruction \
		--encoder_spec all-MiniLM-L6-v2 \
		--llm_type "llama3" \
		--precomputed_embed_keys_path /home/julian_sammet/Documents/repos/KBLaM/datasets/train_synthetic_all-MiniLM-L6-v2_embd_key.npy \
		--precomputed_embed_values_path /home/julian_sammet/Documents/repos/KBLaM/datasets/train_synthetic_all-MiniLM-L6-v2_embd_value.npy


eval-acc-llama-oai: ## Eval kb adapter
	python eval.py accuracy \
		--seed 1607 \
		--dataset_dir ${TEST_DATASET_DIR} \
		--test_dataset synthetic.json \
		--llm_base_dir ${LLAMA_BASE_DIR} \
		--model_dir ${LLAMA_MODEL_CHKPT} \
		--encoder_dir ${ENCODER_FOR_LLAMA_CHKPT} \
		--save_dir ${ATTN_SAVE_DIR} \
		--kb_layer_frequency ${KB_LAYER_FREQ} \
		--kb_size 3200 \
		--kb_scale_factor 100 \
		--no-fancy_instruction \
		--encoder_spec oai \
		--llm_type "llama3" \
		--attn_save_dir ${ATTN_SAVE_DIR} \
		--log_save_dir ${LOG_SAVE_DIR} \
		--query_head_path ${QUERY_HEAD_PATH} \
		--precomputed_embed_keys_path ${TEST_DATASET_DIR}/test_synthetic_data_oai_embd_key.npy \
		--precomputed_embed_values_path ${TEST_DATASET_DIR}/test_synthetic_data_oai_embd_value.npy 


eval-acc-eval-llama-oai: ## Eval kb adapter
	python eval.py acc_results \
		--seed 1607 \
		--dataset_dir ${TEST_DATASET_DIR} \
		--test_dataset test_synthetic_data.json \
		--llm_base_dir ${LLAMA_BASE_DIR} \
		--model_dir ${LLAMA_MODEL_CHKPT} \
		--encoder_dir ${ENCODER_FOR_LLAMA_CHKPT} \
		--save_dir ${ATTN_SAVE_DIR} \
		--kb_layer_frequency ${KB_LAYER_FREQ} \
		--kb_size 100 \
		--kb_scale_factor 100 \
		--no-fancy_instruction \
		--encoder_spec oai \
		--llm_type "llama3" \
		--attn_save_dir ${ATTN_SAVE_DIR} \
		--log_save_dir ${LOG_SAVE_DIR} \
		--query_head_path ${QUERY_HEAD_PATH} \
		--precomputed_embed_keys_path ${TEST_DATASET_DIR}/test_synthetic_data_oai_embd_key.npy \
		--precomputed_embed_values_path ${TEST_DATASET_DIR}/test_synthetic_data_oai_embd_value.npy 


eval-ref-llama-oai: ## Eval kb adapter
	python eval.py refusal \
		--seed 1607 \
		--dataset_dir ${TEST_DATASET_DIR} \
		--test_dataset synthetic.json \
		--llm_base_dir ${LLAMA_BASE_DIR} \
		--model_dir ${LLAMA_MODEL_CHKPT} \
		--encoder_dir ${ENCODER_FOR_LLAMA_CHKPT} \
		--save_dir ${ATTN_SAVE_DIR} \
		--kb_layer_frequency ${KB_LAYER_FREQ} \
		--kb_size 100 \
		--kb_scale_factor 100 \
		--no-fancy_instruction \
		--encoder_spec BigOAI \
		--llm_type "llama3" \
		--precomputed_embed_keys_path ${TEST_DATASET_DIR}/synthetic_BigOAI_embd_key.npy \
		--precomputed_embed_values_path ${TEST_DATASET_DIR}/synthetic_BigOAI_embd_value.npy


eval-basic-llama-oai: ## Eval kb adapter
	python eval.py generation \
		--seed 1607 \
		--dataset_dir ${TEST_DATASET_DIR} \
		--test_dataset synthetic.json \
		--llm_base_dir ${LLAMA_BASE_DIR} \
		--model_dir ${LLAMA_MODEL_CHKPT} \
		--encoder_dir ${ENCODER_FOR_LLAMA_CHKPT} \
		--save_dir ${ATTN_SAVE_DIR} \
		--kb_layer_frequency ${KB_LAYER_FREQ} \
		--kb_size 100 \
		--kb_scale_factor 100 \
		--no-fancy_instruction \
		--encoder_spec BigOAI \
		--llm_type "llama3" \
		--precomputed_embed_keys_path ${TEST_DATASET_DIR}/synthetic_BigOAI_embd_key.npy \
		--precomputed_embed_values_path ${TEST_DATASET_DIR}/synthetic_BigOAI_embd_value.npy 




.PHONY: train train-oai
