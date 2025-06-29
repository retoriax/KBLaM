# Makefile für FastAPI KBLaM Server

# ---- Konfigurierbare Variablen ----
DATASET_DIR ?= /home/julian_sammet/Documents/repos/KBLaM/datasets
DATASET ?= train_synthetic.json
ENCODER_SPEC ?= all-MiniLM-L6-v2
KB_LAYER_FREQ ?= 3
LLM_BASE_DIR ?= /home/julian_sammet/Documents/datadisk/tk/Meta-Llama-3-8B-Instruct
LLM_TYPE ?= llama3
MODEL_DIR ?= /home/julian_sammet/Documents/datadisk/experiments/kblam/exp_v0.0.15/stage1_lr_0.0001KBTokenLayerFreq3UseOutlier1SepQueryHeadUseDataAugKeyFromkey_all-MiniLM-L6-v2_synthetic_llama3_step_1500
PRECOMPUTED_KEYS ?= train_synthetic_all-MiniLM-L6-v2_embd_key.npy
PRECOMPUTED_VALUES ?= train_synthetic_all-MiniLM-L6-v2_embd_value.npy
QUERY_HEAD ?= /home/julian_sammet/Documents/repos/KBLaM-vs-Rag/Creation/model_dir/query_head.pth



# ---- Ziel ----
run:
	TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py \
		--dataset_dir "$(DATASET_DIR)" \
		--dataset "$(DATASET)" \
		--encoder_spec "$(ENCODER_SPEC)" \
		--encoder_dir "$(MODEL_DIR)_encoder/encoder.pt" \
		--llm_base_dir "$(LLM_BASE_DIR)" \
		--llm_type "$(LLM_TYPE)" \
		--model_dir "$(MODEL_DIR)" \
		--precomputed_embed_keys_name "$(PRECOMPUTED_KEYS)" \
		--precomputed_embed_values_name "$(PRECOMPUTED_VALUES)" \
		$(if $(QUERY_HEAD),--query_head_path "$(QUERY_HEAD)")

.PHONY: run