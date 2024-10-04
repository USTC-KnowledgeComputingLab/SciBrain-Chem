export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=1

export CUDA_VISIBLE_DEVICES=0

python run.py \
    --datasets scibench_gen \
    --hf-type chat \
    --hf-path /data/pretrain_model/Meta-Llama-3-8B-Instruct/ \
    --hf-num-gpus 1 \
    --batch-size 1 \
    --max-out-len 1024 \
    --stop-words '<|end_of_text|>' '<|eot_id|>' \
    --debug


### ======
# We have modified the subsets to only evaluate models on chemical tasks
### ======

# scibench_subsets = [
#     'atkins',
#     'chemmc',
#     'matter',
#     'quan'
# ]