import re
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, LlamaTokenizer, LlamaForCausalLM
from openai import OpenAI
def readfiles(infile):

    if infile.endswith('json'):
        lines = json.load(open(infile, 'r', encoding='utf8'))
    elif infile.endswith('jsonl'):
        lines = open(infile, 'r', encoding='utf8').readlines()
        lines = [json.loads(l) for l in lines]
    else:
        raise NotImplementedError

    return lines


def prompt_gen(context, module):
    prompt = """Answer the question as truthfully as possible using the provided context. Please summarize the following details in a dictionaries list:\ncompound name or chemical formula (if the name is not provided), metal source, metal amount, organic linker(s), linker amount, modulator, modulator amount or volume, solvent(s), solvent volume(s), reaction temperature, and reaction time.\nIf any information is not provided or you are unsure, use "N/A". The list can contain multiple dictionaries, corresponding to multiple pieces of synthetic infomation. The dictionaries should have 11 columns, all in lowercase, must end with reply in format as:\n{module}\nHere is the provided context:\n{context}
""".format(context=context,module=module)
    return prompt


def extract_json_from_string(data, module):

    pattern = r"\{.*?\}"
    # pattern = r"\{[\s\S]*?\}" ## For Llama, LlaSMol

    match = re.search(pattern, data)
    
    if match:
        json_str = match.group()
        try:
            json_obj = json.loads(json_str.replace("'", "\""))
            return json_obj
        except json.JSONDecodeError as e:
            return module
    else:
        return module


def model_init(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model = model.to(device)
    return tokenizer, model

def model_init_dfm(model_name_or_id):
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_id)
    model = LlamaForCausalLM.from_pretrained(model_name_or_id, torch_dtype=torch.float16, device_map="auto")
    return tokenizer,model

def llama_response(prompt, tokenizer, model, device="auto"):

    messages = [
        {'role': 'user', 'content': prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    model_input = tokenizer(text, return_tensors='pt').to(device)
    outputs = model.generate(
        model_input.input_ids,
        max_new_tokens=1024,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_input.input_ids, outputs)]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(prompt)
    print(response)

    return response



def gpt_response(prompt):
    client = OpenAI(
    api_key="YOUR_API_KEY",
    base_url="API_URL"
    )
    # print(prompt)
    chat_completions = client.chat.completions.create(
        messages=[
            {
                "role": "user", 
                "content": prompt
                }
            ],
        # model="gpt-4o-mini-2024-07-18",
        model = "gpt-3.5-turbo-0125"
    )

    response = chat_completions.choices[0].message.content
    print(response)
    return response



def chemllm_response(prompt, tokenizer, model, device="auto"):

    model_input = tokenizer(prompt, return_tensors='pt').to(device)

    generation_config = GenerationConfig(
        do_sample=False,
        # top_k=1,
        # temperature=0.9,
        max_new_tokens=1024,
        repetition_penalty=1.5,
        pad_token_id=tokenizer.eos_token_id
    )

    outputs = model.generate(**model_input, generation_config=generation_config)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)
    return response



def chemdfm_response(prompt, tokenizer, model, device="auto"):

    input_text = f"[Round 0]\nHuman: {prompt}\nAssistant:"
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    
    generation_config = GenerationConfig(
        do_sample=False,
        # top_k=20,
        # top_p=0.9,
        # temperature=0.9,
        max_new_tokens=1024,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.eos_token_id
    )

    outputs = model.generate(**inputs, generation_config=generation_config)
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(input_text):]
    
    return generated_text.strip()



def main():

    module = {
    'compound name': '',
    'metal source': '',
    'metal amount': '',
    'linker': '',
    'linker amount': '',
    'modulator': '',
    'modulator amount or volume': '',
    'solvent': '',
    'solvent volume': '',
    'reaction temperature': '',
    'reaction time': ''
    }
    # Model Init
    data_path = ""#test dataset path
    res_path = "" #output result path
    gt_path = "" # output ground truth path
    model_path = ""#your model path
    device = 'cuda'
    tokenizer, model = model_init(model_path, device)
    # tokenizer,model = model_init_dfm(model_path) #for ChemDfm
    # Load Data
    testData = readfiles(data_path)

    for testdata in tqdm(testData):

        # Generate
        response = None
        i = 0
        while i < 5 and response == None:
            # response = chemdfm_response(prompt_gen(testdata["paragraph"], module), tokenizer, model, device)
            response = gpt_response(prompt_gen(testdata["paragraph"], module))#Change gpt_response to the corresponding model
            i = i + 1
        
        response = extract_json_from_string(response, module)
        with open(res_path, "a", encoding='utf-8') as f1:
            json.dump(response, f1, ensure_ascii=False)
            f1.write("\n")

        # Ground Truth
        with open(gt_path, "a", encoding='utf-8') as f1:
            json.dump(testdata["data"], f1, ensure_ascii=False)
            f1.write("\n")
    
main()