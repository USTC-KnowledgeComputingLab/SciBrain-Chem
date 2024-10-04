export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=1

export CUDA_VISIBLE_DEVICES=0

python run.py \
    --datasets sciq_gen \
    --hf-type chat \
    --hf-path /your/path/to/model \
    --hf-num-gpus 1 \
    --batch-size 1 \
    --max-out-len 1024 \
    --stop-words '<|end_of_text|>' '<|eot_id|>' \
    --debug