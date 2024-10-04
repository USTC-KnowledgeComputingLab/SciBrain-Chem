import json
import os

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset

@LOAD_DATASET.register_module()
class SciQDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        dataset = DatasetDict()
        for split in ['train', 'test']:
            raw_data = []
            filename = os.path.join(path, split+'.json')
            with open(filename, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)

            count = 0
            for item in data:
                raw_data.append({
                    'question': item['question'],
                    'A': item["distractor1"],
                    'B': item["distractor2"],
                    'C': item["distractor3"],
                    'D': item["correct_answer"],
                    'answer': "D",
                })
                count += 1
            print(f"Loaded {count} questions for {split} set.")
            dataset[split] = Dataset.from_list(raw_data)
        return dataset

        # raw_data = []
        # mode = 'test'
        # filename = os.path.join(path, mode+'.json')
        # with open(filename, 'r') as infile:
        #     raw_data = json.load(infile)

        # count = 0
        # for entry in raw_data:
        #     raw_data.append(
        #         {
        #             # 'question': entry['question']+"\n"+entry['support'],
        #             'question': entry['question'],
        #             'A': entry["distractor1"],
        #             'B': entry["distractor2"],
        #             'C': entry["distractor3"],
        #             'D': entry["correct_answer"],
        #             'answer': "D", # 这里考虑到每次答题是独立的，直接使用D为标准答案了。
        #         }
        #     )
        #     count += 1

        # print(f"Loaded {count} questions.")
        # dataset = Dataset.from_list(raw_data)
        
        # return dataset