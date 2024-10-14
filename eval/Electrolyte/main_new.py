
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LlamaTokenizer, LlamaForCausalLM
import torch
import transformers
import json
import os
import http.client

def question_1(input_file):
    formatted_data = []
    with open(input_file, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            question_str = f"This is a multiple-choice question about chemistry. Answer the question by replying A, B, C or D.\nQuestion: {item['question']}\nA: {item['A']}\nB: {item['B']}\nC: {item['C']}\nD: {item['D']}\nYour answer is: "
            formatted_data.append(question_str)
    return formatted_data

def question_2(input_file):
    formatted_data = []
    with open(input_file, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            question_str = f"{item['question']}\n "
            formatted_data.append(question_str)
    return formatted_data

def question_3(input_file):
    formatted_data = []
    with open(input_file, 'r') as f:
        for line in f:
            item = json.loads(line.strip())         
            question_str = f"This is a multiple-choice question about chemistry with support material. Answer the question by replying A, B, C or D.\nSupport: {item['support']}\nQuestion: {item['question']}\nA: {item['A']}\nB: {item['B']}\nC: {item['C']}\nD: {item['D']}\nYour answer is: "
            formatted_data.append(question_str)
    return formatted_data

def KALE_LM(questions, model_path):
    model_path = model_path
    responses = []
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    
    for ques in questions:
        messages = []
        messages.append({"role": "user", "content": ques})

        prompt = pipeline.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
        )

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        outputs = pipeline(
            prompt,
            max_new_tokens=1024,
            eos_token_id=terminators,
        )

        res = outputs[0]["generated_text"][len(prompt):]
        print(res)
        responses.append(res)

    return responses

def GPT(questions):
    class LLM:
        #engine = gpt-4o-2024-08-06 or gpt-3.5-turbo
        def __init__(self, engine="gpt-4o-2024-08-06", temperature=0.1, sleep_time=1) -> None:
            self.engine = engine
            self.temperature = temperature
            self.sleep_time = sleep_time
        
        def call(self, message):
            status = 0
            while status != 1:
                try:
                    conn = http.client.HTTPSConnection("")
                    headers = {
                    'Authorization': '',
                    'User-Agent': '',
                    'Content-Type': 'application/json'
                    }
                    payload = json.dumps(
                        {"model": self.engine,
                        "messages": message,
                        "temperature":self.temperature,
                        "max_tokens":1024,
                        }
                    )
                    conn.request("POST", "/v1/chat/completions", payload, headers)
                    res = conn.getresponse()
                    data = res.read()
                    # print("read over")
                    RESPONSE = data.decode("utf-8")
                    # print("decoded")
                    # print(RESPONSE)
                    RESPONSE = json.loads(RESPONSE)["choices"][0]["message"]["content"]
                    # print("status == 1")
                    status = 1

                except Exception as e:
                    print(e)
                    pass
            return RESPONSE
    llm_instance = LLM()
    responses = []
    for ques in questions:
        # 测试消息
        messages = []
        # messages = [{"role": "system", "content": "You are the assistant of a chemical researcher.You need to answer all chemistry questions accurately."}]
        message = {"role": "user", "content": ques}
        messages.append(message)
            # 调用call方法
        response = llm_instance.call(messages)
        print("Response:", response)
        responses.append(response)
    # 处理返回的响应
    return responses

def Llama3(questions):
    model_path = "/data/pretrained_models/Meta-Llama-3-8B-Instruct"

    responses = []
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    
    for ques in questions:
        messages = []
        messages.append({"role": "user", "content": ques})

        prompt = pipeline.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
        )

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        outputs = pipeline(
            prompt,
            max_new_tokens=1024,
            eos_token_id=terminators,
        )

        res = outputs[0]["generated_text"][len(prompt):]
        print(res)
        responses.append(res)

    return responses

def ChemDFM(questions):
    #code and settings are from https://huggingface.co/OpenDFM/ChemDFM-13B-v1.0
    model_name_or_id = "/data/pretrained_models/ChemDFM"
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_id)
    model = LlamaForCausalLM.from_pretrained(model_name_or_id, torch_dtype=torch.float16, device_map="auto")
    responses = []
    
    for ques in questions:
        input_text = f"[Round 0]\nHuman: {ques}\nAssistant:"

        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        generation_config = GenerationConfig(
            do_sample=True,
            top_k=20,
            top_p=0.9,
            temperature=0.9,
            max_new_tokens=1024,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id
        )

        outputs = model.generate(**inputs, generation_config=generation_config)
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(input_text):]
        #print(generated_text)
        responses.append(generated_text)
    
    return responses

def ChemLLM_7B_Chat(questions):
    # code and settings are from  https://huggingface.co/AI4Chem/ChemLLM-7B-Chat
    model_name_or_id = "/data/pretrained_models/ChemLLM/"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_id, torch_dtype=torch.float16, device_map="auto",trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_id,trust_remote_code=True)
    responses = []
    
    for ques in questions:
        prompt = ques

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        generation_config = GenerationConfig(
            do_sample=True,
            top_k=1,
            temperature=0.9,
            max_new_tokens=1000,
            repetition_penalty=1.5,
            pad_token_id=tokenizer.eos_token_id
        )

        outputs = model.generate(**inputs, generation_config=generation_config)
        
        res = tokenizer.decode(outputs[0], skip_special_tokens=True)
        res = res[len(prompt):]
        #print(res)
        responses.append(res)
    
    return responses

def ChemLLM_7B_V1_5_Chat(questions):
    #code and settings are from  https://huggingface.co/AI4Chem/ChemLLM-7B-Chat
    model_name_or_id = "/data/pretrained_models/ChemLLM-1.5"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_id, torch_dtype=torch.float16, device_map="auto",trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_id,trust_remote_code=True)
    responses = []
    
    for ques in questions:
        prompt = ques

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        generation_config = GenerationConfig(
            do_sample=True,
            top_k=1,
            temperature=0.9,
            max_new_tokens=1000,
            repetition_penalty=1.5,
            pad_token_id=tokenizer.eos_token_id
        )

        outputs = model.generate(**inputs, generation_config=generation_config)
        
        res = tokenizer.decode(outputs[0], skip_special_tokens=True)
        res = res[len(prompt):]
        print(res)
        responses.append(res)
    return responses

def LlaSMol(questions):

    model_path = "/data/pretrained_models/LlaSMol-Mistral-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    responses = []
    for ques in questions:

        text = '<s>' + f'[INST] {ques} [/INST]'

        model_input = tokenizer([text], return_tensors='pt')
        generated_ids = model.generate(
            model_input.input_ids,
            max_new_tokens=1024,
        )

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_input.input_ids, generated_ids)]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        #print(response)
        responses.append(response)
    
    return responses



if __name__ == "__main__":
    
    ### Fill the blank
    model_name = "KALE_LM" # KALE_LM, KALE_LM_V1_5, GPT, Llama3, ChemDFM, ChemLLM, ChemLLM_V1_5, LlaSMol
    testcase = 1 # 1, 2, 3, 4
    ### ==============

    test1 = "./data/1.jsonl"
    test2 = "./data/2.jsonl"
    test3 = "./data/3.jsonl"
    test4 = "./data/4.jsonl"

    if testcase == 1 :
        questions = question_1(test1)
    elif testcase == 2 :
        questions = question_2(test2)
    elif testcase == 3 :
        questions = question_3(test3)
    elif testcase == 4 :
        questions = question_1(test4)

    if model_name == "KALE_LM":
        model_path = "/data/AI4Science/checkpoints/Llama3-KALE-LM-Chem-8B-e299fff"
        responses = KALE_LM(questions, model_path)
    elif model_name == "KALE_LM_V1_5":
        model_path = "/data/AI4Science/checkpoints/Llama3-KALE-LM-Chem-1.5-8B-32dcd7a"
        responses = KALE_LM(questions, model_path)
    elif model_name == "GPT":
        responses = GPT(questions)
    elif model_name == "Llama3":
        responses = Llama3(questions)
    elif model_name == "ChemDFM":
        responses = ChemDFM(questions)
    elif model_name == "ChemLLM":
        responses = ChemLLM_7B_Chat(questions)
    elif model_name == "ChemLLM_V1_5":
        responses = ChemLLM_7B_V1_5_Chat(questions)
    elif model_name == "LlaSMol":
        responses = LlaSMol(questions)

    output_file = "./result/" + model_name + "_test" + str(testcase) + ".jsonl"
    i = 1
    with open(output_file,"a") as o:
        for res in responses:    
            o.write(json.dumps({f"Question{i}":res}))
            o.write("\n")
            i += 1