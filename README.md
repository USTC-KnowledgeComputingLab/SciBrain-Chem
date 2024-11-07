# KALE-LM-Chem

This is the official code repository for *KALE-LM-Chem*.

For certain complex testing tasks, we modified part of the code for better evaluation.

- Paper: https://arxiv.org/abs/2409.18695
- Models:
  - Llama3-KALE-LM-Chem-8B: [https://huggingface.co/USTC-KnowledgeComputingLab/Llama3-KALE-LM-Chem-8B](https://huggingface.co/USTC-KnowledgeComputingLab/Llama3-KALE-LM-Chem-8B)
  - Llama3-KALE-LM-Chem-1.5-8B: [https://huggingface.co/USTC-KnowledgeComputingLab/Llama3-KALE-LM-Chem-1.5-8B](https://huggingface.co/USTC-KnowledgeComputingLab/Llama3-KALE-LM-Chem-1.5-8B)

## Eval In The Field Of Electrolyte

AI for Electrolyte is gaining increasing attention. 
To evaluate the performance of large models in the field of electrolyte, we collaborated with chemists to build a small-scale test set and compared the performance of our model with other models on this test set. 
The test data and testing code have been made open source.
- 1.jsonl for Molecular Property
- 2.jsonl for Electrolyte Formula
- 3.jsonl for Text Understanding
- 4.jsonl for College Battery QA

## Cite Our Work

```text
@article{dai2024kale,
  title={KALE-LM: Unleash The Power Of AI For Science Via Knowledge And Logic Enhanced Large Model},
  author={Dai, Weichen and Chen, Yezeng and Dai, Zijie and Huang, Zhijie and Liu, Yubo and Pan, Yixuan and Song, Baiyang and Zhong, Chengli and Li, Xinhe and Wang, Zeyu and others},
  journal={arXiv preprint arXiv:2409.18695},
  year={2024}
}
```
