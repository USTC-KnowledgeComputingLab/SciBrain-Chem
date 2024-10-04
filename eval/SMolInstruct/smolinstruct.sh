export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="/data/dwc/.cache/huggingface/"

export CUDA_VISIBLE_DEVICES=0,1

MODEL_PATH="/data/pretrain_model/"
MODEL_NAME="Meta-Llama-3-8B-Instruct/"
# echo ${MODEL_PATH}${MODEL_NAME}
LOAD_DATA_CACHE_DIR="/data/dwc/.cache/huggingface/datasets"

# Step 1. Generate responses for samples
echo ""
echo "******Generate responses for samples******"
echo ""
python generate_on_dataset.py --is_llama True --model_name ${MODEL_PATH}${MODEL_NAME} --batch_size 8 --max_new_tokens 128 --data_path SMolInstruct/ --output_dir eval/${MODEL_NAME}/output --load_data_cache_dir ${LOAD_DATA_CACHE_DIR}
# Step 2. Extract predicted answer from model outputs
echo ""
echo "******Extract predicted answer from model outputs******"
echo ""
python extract_prediction.py --output_dir eval/${MODEL_NAME}/output --prediction_dir eval/${MODEL_NAME}/prediction
# Step 3. Calculate metrics
echo ""
echo "******Calculate metrics******"
echo ""
python compute_metrics.py --prediction_dir eval/${MODEL_NAME}/prediction --load_data_cache_dir ${LOAD_DATA_CACHE_DIR}

# Comment out some lines related to 'pad_token_id' in model.py
# Modify torch_dtype for "triu_tril_cuda_template" not implemented for 'BFloat16' in model.py
# Modify prompter in generation.py
# !!! Modify cache_dir in load_dataset() in generate_on_dataset.py and compute_metrics.py