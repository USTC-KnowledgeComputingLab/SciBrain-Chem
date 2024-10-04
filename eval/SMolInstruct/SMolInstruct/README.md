---
language:
- en
license: cc-by-4.0
tags:
- chemistry
- molecule
- small molecule
- instructions
---

<h1 align="center">  ⚛️ SMolInstruct  </h1>

SMolInstruct is a **large-scale**, **comprehensive**, and **high-quality instruction tuning dataset** crafted for **chemistry**. It centers around small molecules, and contains 14 meticulously selected tasks and over 3M samples.
This dataset has both **SMILES** and **SELFIES** versions, and you could switch to SELFIES by using `use_selfies=True` when loading.

**Version History**

- v1.2.0 (2024.04.21): Add a small test subset with at most 200 samples for each task. You could use it by assigning `use_test_subset=True`. Also add `use_first` to load the first specific number of samples for each task. See below for details.
- v1.1.1 (2024.04.18): Fix double tag problem (`<SMILES> <SMILES> ... </SMILES> </SMILES>`) for retrosynthesis. We recommend all to use this or newer version.
- v1.1.0 (2024.03.05): Delete a small amount of samples with invalid molecules, and add SELFIES.
- v1.0.0 (2024.02.13): Upload the first version.

**Paper**: [LlaSMol: Advancing Large Language Models for Chemistry with a Large-Scale, Comprehensive, High-Quality Instruction Tuning Dataset](https://arxiv.org/abs/2402.09391)

**Page**: [https://osu-nlp-group.github.io/LlaSMol](https://osu-nlp-group.github.io/LlaSMol)

**Code**: [https://github.com/OSU-NLP-Group/LlaSMol](https://github.com/OSU-NLP-Group/LlaSMol)

**Models**: [https://huggingface.co/osunlp/LlaSMol](https://huggingface.co/osunlp/LlaSMol)


## 🔭 Overview

The following figure illustrates the tasks and corresponding examples.

![Overview of the tasks.](./fig/tasks.png)

The following table shows the tasks and statistics over the SMolInstruct dataset, where “Qry.” and “Resp.” are average lengths of queries and responses, respectively.

![Statistics of the SMolInstruct dataset.](./fig/statistics.png)

An example is shown below:

```python
{
    'input': 'Based on the given reactants and reagents: <SMILES> CCCCCCCC/C=C\\CCCCCCCC(=O)OCCNCCOC(=O)CCCCCCC/C=C\\CCCCCCCC.CCN=C=NCCCN(C)C.CN(C)C1=CC=NC=C1.CN(C)CCSCC(=O)O.CO.Cl.ClCCl.O.O=C(O)C(F)(F)F.O=C([O-])[O-].[K+] </SMILES>, what product could potentially be produced?',
    'output': 'The product can be <SMILES> CCCCCCCC/C=C\\CCCCCCCC(=O)OCCN(CCOC(=O)CCCCCCC/C=C\\CCCCCCCC)C(=O)CSCCN(C)C </SMILES> .',
    'raw_input': 'CCCCCCCC/C=C\\CCCCCCCC(=O)OCCNCCOC(=O)CCCCCCC/C=C\\CCCCCCCC.CCN=C=NCCCN(C)C.CN(C)C1=CC=NC=C1.CN(C)CCSCC(=O)O.CO.Cl.ClCCl.O.O=C(O)C(F)(F)F.O=C([O-])[O-].[K+]',
    'raw_output': 'CCCCCCCC/C=C\\CCCCCCCC(=O)OCCN(CCOC(=O)CCCCCCC/C=C\\CCCCCCCC)C(=O)CSCCN(C)C',
    'split': 'train',
    'task': 'forward_synthesis',
    'input_core_tag_left': '<SMILES>',
    'input_core_tag_right': '</SMILES>',
    'output_core_tag_left': '<SMILES>',
    'output_core_tag_right': '</SMILES>',
    'target': None
}
```

## ⚔️ Usage
You can use the following lines to load the dataset:
```python
from datasets import load_dataset

dataset = load_dataset('osunlp/SMolInstruct')

train_set = dataset['train']
validation_set = dataset['validation']
test_set = dataset['test']
```

A SELFIES version could also be used, by simplying adding an argument:
```python
dataset = load_dataset('osunlp/SMolInstruct', use_selfies=True)
```

You can also specify what tasks to load:
```python
ALL_TASKS = (
  'forward_synthesis',
  'retrosynthesis',
  'molecule_captioning',
  'molecule_generation',
  'name_conversion-i2f',
  'name_conversion-i2s',
  'name_conversion-s2f',
  'name_conversion-s2i',
  'property_prediction-esol',
  'property_prediction-lipo',
  'property_prediction-bbbp',
  'property_prediction-clintox',
  'property_prediction-hiv',
  'property_prediction-sider',
)

train_set = load_dataset('osunlp/SMolInstruct', tasks=ALL_TASKS)
```

You could use `use_test_subset=True` to use a subset of the test set, to quickly evaluate your models.
```python
test_set = load_dataset('osunlp/SMolInstruct', split='test', use_test_subset=True)
```
You could also `use_first=INTEGER` to load only first at most `INTEGER` samples for each task.
```python
# load first 500 samples for each task
test_set = load_dataset('osunlp/SMolInstruct', split='test', use_first=500)
```

## 🛠️ Evaluation

The evaluation code will be at [https://github.com/OSU-NLP-Group/LlaSMol](https://github.com/OSU-NLP-Group/LlaSMol).

## 🛠️ Data Construction

The construction of SMolInstruct goes through a four-step pipeline: 

- **data collection**: Collect data from various sources and organize it for the tasks.
- **quality control**: Rigorous scrutiny is applied to remove samples with chemically invalid SMILES and wrong or inaccurate information, as well as duplicated samples.
- **data splitting**: Samples are carefully splitted into train/validation/test set to avoid data leakage across tasks. Also, the splitting is compatible with previous work to faciliate fair comparison.
- **instruction construction**: We create natural and diverse templates for creating instructions. Molecular SMILES representations are canonicalized to provide a standardized data format. In addition, we use special tags to encapsulate corresponding segments (e.g., <SMILES>...</SMILES>} for SMILES, etc.) to promote model learning during training and faciliate answer extraction during inference.


## 🚨 License

The **SMolInstruct** dataset is licensed under CC BY 4.0.

We emphatically urge all users to adhere to the highest ethical standards when using our dataset, including maintaining fairness, transparency, and responsibility in their research. Any usage of the dataset that may lead to harm or pose a detriment to society is strictly **forbidden**.

## 🔍 Citation
If our paper or related resources prove valuable to your research, we kindly ask for citation. Please feel free to contact us with any inquiries.

```
@article{yu2024llasmol,
  title={LlaSMol: Advancing Large Language Models for Chemistry with a Large-Scale, Comprehensive, High-Quality Instruction Tuning Dataset},
  author={Botao Yu and Frazier N. Baker and Ziqi Chen and Xia Ning and Huan Sun},
  journal={arXiv preprint arXiv:2402.09391},
  year={2024}
}
```

Thank you for your interest in our work.
