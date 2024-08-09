import pandas as pd
import selfies as sf
from rdkit import Chem
from sklearn.metrics import mean_absolute_error
import os
import argparse

def sf_encode(selfies):
    try:
        smiles = sf.decoder(selfies)
        return smiles
    except Exception:
        return None

def sf_trans(formula):
    try:
        smiles = sf.decoder(formula)
        if smiles == None or smiles == '': # 生成的可能直接就是smile表达式
            smiles = sf.decoder(sf.encoder(formula))
        return smiles
    except Exception:
        return None

def convert_to_canonical_smiles(smiles):
    if smiles is None:
        return None
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is not None:
        canonical_smiles = Chem.MolToSmiles(molecule, isomericSmiles=False, canonical=True)
        return canonical_smiles
    else:
        return None
    
def metrics(input_path, output_path, task):
    # data = pd.read_table(input_path, sep='\t', on_bad_lines='skip')
    raw_data = []
    with open(input_path, 'r') as f:
        for line in f.readlines():
            raw_data.append(eval(line[:-1]))
    data = pd.DataFrame(raw_data)
    all_data = data.shape[0]
    
    if task == 'property_pred':
        # data['output'] = data['output'].astype(str).str.extract(r'(-?\d+\.?\d*)\s*</s>')
        data['output'] = data['output'].astype(str).str.extract(r'(-?\d*\.?\d+)(?!.*\d)') # To check later

        data.dropna(axis=0, how='any', inplace=True)
        
        data['output'] = pd.to_numeric(data['output'])
        data['ground_truth'] = pd.to_numeric(data['ground_truth'])

        data.dropna(axis=0, how='any', inplace=True)
        
        mae = mean_absolute_error(data['ground_truth'], data['output'])
        print(mae)
        
        # data.to_csv(output_path, index=None, sep='\t', header=True)
        data.to_json(output_path, orient='records', lines=True)

    elif task == 'mol_gen':
        data.dropna(axis=0, how='any', inplace=True)
        
        # data['output'] = data['output'].apply(lambda x: x.rsplit(']', 1)[0] + ']' if isinstance(x, str) else x)
        data['output'] = data['output'].apply(lambda x: x.rsplit('<|end_of_text|>', 1)[0] if isinstance(x, str) else x)

        # data['output_smiles'] = data['output'].map(sf_encode)
        # input: sf_encode("This molecular's SMILES name is [C][C][=N][C][Branch1][C][C][=C][Branch1][#Branch1][C][=Branch1][C][=O][C][Br][NH1][Ring1][#Branch2]")
        # output: 'CC1=NC(C)=C(C(=O)CBr)[NH1]1'
        data['output_smiles'] = data['output'].map(sf_trans)
        
        data.dropna(axis=0, how='any', inplace=True)
        
        data['output_smiles'] = data['output_smiles'].map(convert_to_canonical_smiles)
        
        data.dropna(axis=0, how='any', inplace=True)
        
        # data['ground truth'] = data['ground_truth']
        data['ground_smiles'] = data['ground_truth'].map(sf_encode)
        data['ground_smiles'] = data['ground_smiles'].map(convert_to_canonical_smiles)
        
        data.dropna(axis=0, how='any', inplace=True)
        
        # data.to_csv(output_path, index=None, sep='\t')
        data.to_json(output_path, orient='records', lines=True)
        
    else:
        data.dropna(axis=0, how='any', inplace=True)

        data['SELFIES'] = data['description']
        data['SMILES_org'] = data['description'].map(sf_encode)
        data['SMILES'] = data['SMILES_org'].map(convert_to_canonical_smiles)

        # data['output'] = data['output'].apply(lambda x: x.rsplit('.', 1)[0] + '.' if isinstance(x, str) else x)
        # # data['output'] = data['output'].str.rsplit('.', 1).str[0] + '.'
        data['output'] = data['output'].apply(lambda x: x.rsplit('<|end_of_text|>', 1)[0] if isinstance(x, str) else x)

        # data['ground truth'] = data['ground_truth']
        data.dropna(axis=0, how='any', inplace=True)

        # data[['SMILES', 'SELFIES', 'ground truth', 'output']].to_csv(output_path, index=None, sep='\t')
        data = data[['SMILES', 'SELFIES', 'ground_truth', 'output']]
        data.to_json(output_path, orient='records', lines=True)

        
if __name__ == '__main__':
    # input_path = 'property.txt'
    # output_path = 'pre_property.txt'
    # task = 'property_pred'

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    args = parser.parse_args()
    model_name = args.model_name

    # Create output directory
    output_dir = f'prediction/{model_name}/'
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    for subtask in ["description_guided_molecule_design","forward_reaction_prediction","molecular_description_generation","property_prediction","retrosynthesis","reagent_prediction"]:
        if subtask == 'property_prediction':
            task = 'property_pred'
        elif subtask == 'molecular_description_generation':
            task = 'understand'
        else:
            task = 'mol_gen'
        
        input_file = os.path.join('output/', model_name, subtask+'.jsonl')
        output_file = os.path.join(os.path.dirname(output_dir), subtask+'.jsonl')
        print(f'Reading {input_file}......Writing predictions into {output_file}......')

        metrics(input_file, output_file, task)