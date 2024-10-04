import os
import sys
import fire
import json 
#import gradio as gr
import torch
import transformers
#from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM

from tqdm import tqdm

# from utils.prompter import Prompter
# Modified as below
# ===============================================
import os.path as osp
from typing import Union
class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("demo/templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        max_new_tokens: int,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        # if input:
        #     res = self.template["prompt_input"].format(
        #         instruction=instruction, input=input
        #     )
        # else:
        #     res = self.template["prompt_no_input"].format(
        #         instruction=instruction
        #     )
        
        if input:
            res = self.template["prompt_input_with_limited_tokens"].format(
                max_new_tokens=max_new_tokens, instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input_with_limited_tokens"].format(
                max_new_tokens=max_new_tokens, instruction=instruction
            )

        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()
# ===============================================

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# try:
#     if torch.backends.mps.is_available():
#         device = "mps"
# except:  # noqa: E722
#     pass


def main(
    is_llama: bool = True,
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    input_dir: str = '',
    output_dir: str = '',
    batch_size: int = 16, # 假的，没用
    max_new_tokens: int = 128,
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    prompter = Prompter(prompt_template)
    # tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if device == "cuda":
        # model = LlamaForCausalLM.from_pretrained(
        #     base_model,
        #     load_in_8bit=load_8bit,
        #     torch_dtype=torch.float16,
        #     device_map="auto",
        # )
        # model = PeftModel.from_pretrained(
        #     model,
        #     lora_weights,
        #     torch_dtype=torch.float16,
        # )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        # model = PeftModel.from_pretrained(
        #     model,
        #     lora_weights,
        #     device_map={"": device},
        #     torch_dtype=torch.float16,
        # )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        # model = PeftModel.from_pretrained(
        #     model,
        #     lora_weights,
        #     device_map={"": device},
        # )

    # unwind broken decapoda-research config
    # model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    # model.config.bos_token_id = 1
    # model.config.eos_token_id = 2
    if is_llama:
        tokenizer.eos_token_id=128009
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # if not load_8bit:
    #     model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # Create output directory
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(max_new_tokens, instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return prompt, prompter.get_response(output)

    # Tasks in molecule
    for task in ["description_guided_molecule_design","forward_reaction_prediction","molecular_description_generation","property_prediction","retrosynthesis","reagent_prediction"]:
        input_file = os.path.join(input_dir, task+".json")
        print(f"Reading {input_file}......")
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # output_file = os.path.join(output_dir, task+".txt")
        output_file = os.path.join(output_dir, task+".jsonl")
        print(f"Writing output into {output_file}......")
        with open(output_file, 'w') as f:
            # f.write("description\tground_truth\toutput\n")
            count = 0
            for item in tqdm(data):
                if item['metadata']['split'] != "test":
                    continue
                count += 1
                description = item['input']
                ground_truth = item['output']
                prompt, output = evaluate(instruction=item['instruction'], input=item['input']) 

                # f.write(f"{description}\t{ground_truth}\t{output}\n")
                data_to_jsonl = {
                    'description': description, 
                    'ground_truth': ground_truth,
                    'prompt': prompt,
                    'output': output,
                    }
                json.dump(data_to_jsonl, f)
                f.write('\n')

            print(f"Finished testing on {count} problems on {task}......")

    # with open(input_dir, 'r') as f:
    #     data = json.load(f)

    # with open(output_dir, 'w') as f:
    #     f.write("description\tground_truth\toutput\n")
    #     for item in tqdm(data):
    #         description = item['input']
    #         ground_truth = item['output']
    #         output = evaluate(instruction=item['instruction'], input=item['input']) 

    #         f.write(f"{description}\t{ground_truth}\t{output}\n")
                
if __name__ == "__main__":
    fire.Fire(main)
