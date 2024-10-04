import re
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def readfiles(infile):

    if infile.endswith('json'):
        lines = json.load(open(infile, 'r', encoding='utf8'))
    elif infile.endswith('jsonl'):
        lines = open(infile, 'r', encoding='utf8').readlines()
        lines = [json.loads(l) for l in lines]
    else:
        raise NotImplementedError

    return lines


def prompt_gen(context,module):
    prompt = """Please extract the relevant information from the text I gave you and finally output a json object without outputting other information.

###context:
{context}

###json:
{module}
""".format(context=context,module=module)
    return prompt


def extract_json_from_string(data,module):
 
    pattern = r"\{.*?\}" 

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


def model_init(model_path,device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model = model.to(device)
    return tokenizer,model

def get_response(prompt,tokenizer,model,device="auto"):

    messages = [
        {'role': 'user', 'content': prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_input = tokenizer([text], return_tensors='pt').to(device)
    attention_mask = torch.ones(model_input.input_ids.shape, dtype=torch.long, device=device)
    generated_ids = model.generate(
        model_input.input_ids,
        max_new_tokens=512,
        attention_mask=attention_mask,  
        pad_token_id=tokenizer.eos_token_id,  
    )

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_input.input_ids, generated_ids)]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response



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
    
    model_path = "/your/path/to/model"
    data_path = "/your/path/to/data"
    res_path = "/your/path/to/response" # jsonl
    gt_path = "/your/path/to/ground_truth" # jsonl

    # Model Init
    device = 'cuda:0'
    tokenizer,model = model_init(model_path,device) 
    # Load Data
    testData = readfiles(data_path)

    for testdata in tqdm(testData):

        # Generate
        response = None
        i = 0
        while i < 5 and response == None:
            response = get_response(prompt_gen(testdata["paragraph"],module),tokenizer,model,device)
            i = i + 1
        
        response = extract_json_from_string(response,module)
        with open(res_path,"a",encoding='utf-8') as f1:
            json.dump(response,f1,ensure_ascii=False)
            f1.write("\n")

        # Ground Truth
        with open(gt_path,"a",encoding='utf-8') as f1:
            json.dump(testdata["data"],f1,ensure_ascii=False)
            f1.write("\n")
    
main()