export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="/data/dwc/.cache/huggingface/"

export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="/data/pretrain_model/"
MODEL_NAME="Meta-Llama-3-8B-Instruct/"

### Molecule ###
# Step 1
python evaluation/molecule/generate_example.py \
    --is_llama True \
    --base_model ${MODEL_PATH}${MODEL_NAME} \
    --input_dir Mol-Instructions/data/Molecule-oriented_Instructions/ \
    --output_dir output/${MODEL_NAME}/ \
    --batch_size 16 \
    --max_new_tokens 128
# Step 2
python evaluation/molecule/evaluate.py --model_name ${MODEL_NAME}
# Step 3
python evaluation/molecule/mol_translation_selfies.py --model_name ${MODEL_NAME}
python evaluation/molecule/fingerprint_metrics.py --model_name ${MODEL_NAME}
python evaluation/molecule/text_translation_metrics.py --model_name ${MODEL_NAME}

# Modify generate_example.py and evaluate.py for our models
# Modify Prompter for limited tokens in generate_example.py
# Modify Reg in metrics() in evaluate.py for task 'property_pred'
# Modify selfies-smiles trans in evaluate.py
# nltk_data in /data/miniconda3/envs/LM/