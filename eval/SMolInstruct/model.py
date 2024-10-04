import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModelForCausalLM

from config import BASE_MODELS


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    return device


def load_tokenizer_and_model(model_name, base_model=None, device=None, is_llama=False):
    # if base_model is None:
    #     if model_name in BASE_MODELS:
    #         base_model = BASE_MODELS[model_name]
    # assert base_model is not None, "Please assign the corresponding base model to the argument 'base_model'."

    base_model = model_name

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    # tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True) # Added trust_remote_code.
    tokenizer.padding_side = 'left' # default 'right' for llama
    if is_llama:
        tokenizer.eos_token_id=128009
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.sep_token = '<unk>'
    # tokenizer.cls_token = '<unk>'
    # tokenizer.mask_token = '<unk>'

    if device is None:
        device = get_device()
    
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            # trust_remote_code=True, # Added trust_remote_code.
        )
            
        # model = PeftModelForCausalLM.from_pretrained(
        #     model,
        #     model_name,
        #     torch_dtype=torch.bfloat16,
        # )
    else:
        raise NotImplementedError("No implementation for loading model on CPU yet.")
    
    # model = model.merge_and_unload()

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    return tokenizer, model
