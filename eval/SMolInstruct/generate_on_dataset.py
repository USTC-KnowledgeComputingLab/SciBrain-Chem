import os
import json
from tqdm.auto import tqdm

import fire
from datasets import load_dataset

from config import TASKS_GENERATION_SETTINGS, TASKS, DEFAULT_MAX_INPUT_TOKENS, DEFAULT_MAX_NEW_TOKENS
from generation import LlaSMolGeneration


def generate(
    generator: LlaSMolGeneration,
    # Data
    data_path: str = "osunlp/SMolInstruct",
    split: str = 'test',
    task: str = '',
    # Output
    output_file: str = '',
    # Running configs
    batch_size: int = 1,
    max_input_tokens: int = None,
    max_new_tokens: int = None,
    print_out=False,
    load_data_cache_dir: str = None,
    **generation_kargs,
):
    # Setting default params for certain tasks
    task_settings = TASKS_GENERATION_SETTINGS.get(task)
    if task_settings is not None:
        print('Setting configurations for %s' % task)
        for key in task_settings:
            value = task_settings[key]
            if key == 'generation_kargs':
                assert isinstance(value, dict)
                eval(key).update(value)
                print(key, '<-', value)
            else:
                if key in ('max_input_tokens', 'max_new_tokens') and eval(key) is not None:
                    pass
                else:
                    statement = '{key} = {value}'.format(key=key, value=value)
                    print(statement)
                    exec(statement)
    if max_input_tokens is None:
        max_input_tokens = DEFAULT_MAX_INPUT_TOKENS
    if max_new_tokens is None:
        max_new_tokens = DEFAULT_MAX_NEW_TOKENS

    print(f'max_input_tokens = {max_input_tokens}')
    print(f'max_new_tokens = {max_new_tokens}')

    # Load dataset
    data = load_dataset(data_path, split=split, tasks=(task,), cache_dir=load_data_cache_dir)
    data = list(data)

    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Check the output and continue from the break point
    mode = 'w'
    num_exist_lines = 0
    if os.path.isfile(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                num_exist_lines += 1
        if num_exist_lines > 0:
            mode = 'a'
    
    if num_exist_lines >= len(data):
        print('Already done %d / %d.' % (num_exist_lines, len(data)))
        return
    else:
        print('Todo: %d / %d' % (len(data) - num_exist_lines, len(data)))
    
    if num_exist_lines > 0:
        print('Continue with the existing %d' % num_exist_lines)

    with open(output_file, mode) as f, tqdm(total=len(data)) as pbar:
        k = num_exist_lines
        pbar.update(k)
        
        while True:
            if k >= len(data):
                break
            e = min(k + batch_size, len(data))

            batch_input = []
            
            for item in data[k: e]:
                sample_input = item['input']
                batch_input.append(sample_input)
                # print(sample_input) 
                # <SMILES> C(=NC1CCCCC1)=NC1CCCCC1.CC#N.CC(=O)OC[C@H](NC(=O)[C@H](CC(C)C)NC(=O)OCC1=CC=CC=C1)[C@@H](OC(C)=O)[C@@H](OC(C)=O)[C@H](OC(C)=O)C(=O)O.CCN(CC)CC.CCOC(=O)C[C@H](N)C1=CC=C(C)C=C1 </SMILES> Based on the reactants and reagents given above, suggest a possible product.

            if len(batch_input) == 0:
                return
            
            batch_samples = data[k: e]
            
            batch_outputs = generator.generate(batch_input, task=task, batch_size=batch_size, max_input_tokens=max_input_tokens, max_new_tokens=max_new_tokens, canonicalize_smiles=False, print_out=False, **generation_kargs)

            assert len(batch_input) == len(batch_outputs)
            for sample, sample_outputs in zip(batch_samples, batch_outputs):

                # print(sample)
                # {
                #     'input': 
                #         '<SMILES> C(=NC1CCCCC1)=NC1CCCCC1.CC#N.CC(=O)OC[C@H](NC(=O)[C@H](CC(C)C)NC(=O)OCC1=CC=CC=C1)[C@@H](OC(C)=O)[C@@H](OC(C)=O)[C@H](OC(C)=O)C(=O)O.CCN(CC)CC.CCOC(=O)C[C@H](N)C1=CC=C(C)C=C1 </SMILES> Based on the reactants and reagents given above, suggest a possible product.', 
                #     'output': 
                #         'A possible product can be <SMILES> CCOC(=O)C[C@H](NC(=O)[C@@H](OC(C)=O)[C@H](OC(C)=O)[C@H](OC(C)=O)[C@H](COC(C)=O)NC(=O)[C@H](CC(C)C)NC(=O)OCC1=CC=CC=C1)C1=CC=C(C)C=C1 </SMILES> .', 
                #     'raw_input': 
                #         'C(=NC1CCCCC1)=NC1CCCCC1.CC#N.CC(=O)OC[C@H](NC(=O)[C@H](CC(C)C)NC(=O)OCC1=CC=CC=C1)[C@@H](OC(C)=O)[C@@H](OC(C)=O)[C@H](OC(C)=O)C(=O)O.CCN(CC)CC.CCOC(=O)C[C@H](N)C1=CC=C(C)C=C1', 
                #     'raw_output': 
                #         'CCOC(=O)C[C@H](NC(=O)[C@@H](OC(C)=O)[C@H](OC(C)=O)[C@H](OC(C)=O)[C@H](COC(C)=O)NC(=O)[C@H](CC(C)C)NC(=O)OCC1=CC=CC=C1)C1=CC=C(C)C=C1', 
                #     'split': 'test', 
                #     'task': 'forward_synthesis', 
                #     'input_core_tag_left': '<SMILES>', 
                #     'input_core_tag_right': '</SMILES>', 
                #     'output_core_tag_left': '<SMILES>', 
                #     'output_core_tag_right': '</SMILES>', 
                #     'target': None
                # }
                # print(sample_outputs)
                # {
                #     'input_text': 
                #         '<SMILES> C(=NC1CCCCC1)=NC1CCCCC1.CC#N.CC(=O)OC[C@H](NC(=O)[C@H](CC(C)C)NC(=O)OCC1=CC=CC=C1)[C@@H](OC(C)=O)[C@@H](OC(C)=O)[C@H](OC(C)=O)C(=O)O.CCN(CC)CC.CCOC(=O)C[C@H](N)C1=CC=C(C)C=C1 </SMILES> Based on the reactants and reagents given above, suggest a possible product.', 
                #     'real_input_text': 
                #         '<s>[INST] <SMILES> C(=NC1CCCCC1)=NC1CCCCC1.CC#N.CC(=O)OC[C@H](NC(=O)[C@H](CC(C)C)NC(=O)OCC1=CC=CC=C1)[C@@H](OC(C)=O)[C@@H](OC(C)=O)[C@H](OC(C)=O)C(=O)O.CCN(CC)CC.CCOC(=O)C[C@H](N)C1=CC=C(C)C=C1 </SMILES> Based on the reactants and reagents given above, suggest a possible product. [/INST]', 
                #     'output': 
                #         ['<SMILES> C(=NC1CCCCC1)=NC1CCCCC1.CC#N.CC(=O)OC[C@H](NC(=O)[C@', '[INST] <SMILES> C(=NC1CCCCC1)=NC', '<SMILES> C(=NC1CCCCC1)=NC1CCCCC1.CC#N.CC(=O)OC[C@H](NC(=O)[C@', '<SMILES> C(=NC1CCCCC1)=NC1CCCCC1.CC#N.CC(=O)OC[C@H](NC(=O)[C@H](CC(C)C)NC(=O)OCC1=CC=CC=C1', '[INST] <SMILES> C(=NC1CCCCC1)=NC']
                # }

                if print_out:
                    tqdm.write(sample['task'])
                    tqdm.write(sample['input_text'])
                    tqdm.write(sample_outputs)
                    tqdm.write('\n')

                log = {
                    'input': sample['raw_input'], 
                    'gold': sample['raw_output'], 
                    'output': sample_outputs['output'], 
                    'task': sample['task'], 
                    'split': split, 
                    'target': sample['target'],
                    'input_text': sample_outputs['input_text'],
                    'real_input_text': sample_outputs['real_input_text'],
                }

                f.write(json.dumps(log, ensure_ascii=False) + '\n')

            pbar.update(e - k)
            k = e


def main(
    # Model
    model_name: str = "",
    base_model: str = None,
    # Data
    data_path: str = "osunlp/SMolInstruct",
    split: str = 'test',
    tasks = None,
    # Output
    output_dir: str = 'eval',
    # Running configs
    batch_size: int = 1,
    max_input_tokens: int = None,
    max_new_tokens: int = None,
    print_out=False,
    device = None,
    load_data_cache_dir: str = None,
    is_llama = False,
    **generation_kargs,
):
    if tasks is None:
        tasks = TASKS
    elif isinstance(tasks, str):
        tasks = (tasks,)
    
    # if is_llama:
    #     print('is_llama == True')
    #     return
    # else:
    #     print('is_llama == False')
    #     return

    generator = LlaSMolGeneration(model_name=model_name, base_model=base_model, device=device, is_llama=is_llama)
    
    os.makedirs(output_dir, exist_ok=True)

    for task in tasks:
        generate(
            generator,
            data_path=data_path,
            split=split,
            task=task,
            output_file=os.path.join(output_dir, task + '.jsonl'),
            batch_size=batch_size,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
            print_out=print_out,
            load_data_cache_dir=load_data_cache_dir,
            **generation_kargs
        )


if __name__ == "__main__":
    fire.Fire(main)
