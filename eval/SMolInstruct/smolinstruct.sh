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


# model.config.pad_token_id = tokenizer.pad_token_id; model.config.bos_token_id = tokenizer.bos_token_id; model.config.eos_token_id = tokenizer.eos_token_id
# inputs = tokenizer(prompt, truncation=False, padding=False, return_tensors="pt", add_special_tokens=False,); input_ids = inputs["input_ids"].to(device)
# generation_config = GenerationConfig(pad_token_id=model.config.pad_token_id, bos_token_id=model.config.bos_token_id, eos_token_id=model.config.eos_token_id, num_beams=8, num_return_sequences=5, max_input_tokens=512, max_new_tokens=128)
# generation_config = GenerationConfig(pad_token_id=model.config.pad_token_id, bos_token_id=model.config.bos_token_id, eos_token_id=[model.config.eos_token_id, 128001], num_beams=8, num_return_sequences=5, max_input_tokens=512, max_new_tokens=128)
# generation_config = GenerationConfig(pad_token_id=model.config.pad_token_id, bos_token_id=model.config.bos_token_id, eos_token_id=128001, num_beams=8, num_return_sequences=5, max_input_tokens=512, max_new_tokens=128)
# generation_config = GenerationConfig(early_stopping=True, stop_strings=['<|end_of_text|>', '<|eot_id|>'], pad_token_id=model.config.pad_token_id, bos_token_id=model.config.bos_token_id, eos_token_id=model.config.eos_token_id, num_beams=8, num_return_sequences=5, max_input_tokens=512, max_new_tokens=128)
# generation_output = model.generate(
#                 input_ids=input_ids,
#                 generation_config=generation_config,
#                 return_dict_in_generate=True,
#                 output_scores=True,
#                 max_new_tokens=128,
#             )
# s = generation_output.sequences
# output = tokenizer.batch_decode(s, skip_special_tokens=False)
# output_text = []
# for output_item in output:
#     text = output_item.split("### Response:")[-1].strip()
#     output_text.append(text)