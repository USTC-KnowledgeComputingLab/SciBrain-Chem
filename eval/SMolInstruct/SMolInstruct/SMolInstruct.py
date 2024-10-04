# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SMolInstruct: A Large-Scale, Comprehensive, High-Quality Instruction Tuning Dataset for Small Molecules"""


import json
import os

import datasets


# Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@article{yu2024llasmol,
    title={LlaSMol: Advancing Large Language Models for Chemistry with a Large-Scale, Comprehensive, High-Quality Instruction Tuning Dataset},
    author={Botao Yu and Frazier N. Baker and Ziqi Chen and Xia Ning and Huan Sun},
    journal={arXiv preprint arXiv:2402.09391},
    year={2024}
}
"""

# Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
SMolInstruct is a large-scale instruction tuning dataset for chemistry tasks and centers around small molecules. It contains a total of 14 chemistry tasks and over 3 million samples. It is designed to be large-scale, comprehensive, and high-quality.
"""

# Add a link to an official homepage for the dataset here
_HOMEPAGE = "https://osu-nlp-group.github.io/LLM4Chem/"

# Add the licence for the dataset here if you can find it
_LICENSE = "cc-by-4.0"

# Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
# _URLS = {
#     "first_domain": "https://huggingface.co/great-new-dataset-first_domain.zip",
#     "second_domain": "https://huggingface.co/great-new-dataset-second_domain.zip",
# }


TASKS = (
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


class SmolInstructDatasetConfig(datasets.BuilderConfig):
    def __init__(self, tasks=None, sample_group='instruction_tuning', insert_core_tags=True, use_selfies=False, use_test_subset=False, use_first=None, **kwargs):
        """BuilderConfig for MyDataset
        Args:
          data_url: `string`, url to the dataset (word or raw level)
          **kwargs: keyword arguments forwarded to super.
        """
        super(SmolInstructDatasetConfig, self).__init__(
            **kwargs,
        )
        if tasks is None:
            tasks = TASKS
        else:
            tasks = set(tasks)
            all_tasks = set(TASKS)
            assert len(tasks - all_tasks) == 0, 'Unsupported task(s): {tasks}'.format(tasks=(tasks - all_tasks))
        self.tasks = tasks
        self.sample_group = sample_group
        self.insert_core_tags = insert_core_tags
        self.use_selfies = use_selfies
        if 'split' in kwargs:
            self.split = kwargs['split']
        else:
            self.split = None
        self.use_test_subset = use_test_subset
        if use_first is not None:
            assert use_first > 0, "use_first must be a positive integer."
            use_first = int(use_first)
        self.use_first = use_first


class SMolInstruct(datasets.GeneratorBasedBuilder):
    """SMolInstruct: A large-scale chemistry instruction tuning dataset."""

    VERSION = datasets.Version("1.2.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    BUILDER_CONFIG_CLASS = SmolInstructDatasetConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    # BUILDER_CONFIGS = [
    #     datasets.BuilderConfig(name="instruction_tuning", version=VERSION, description="Default set for instruction tuning."),
    #     datasets.BuilderConfig(name="second_domain", version=VERSION, description="This part of my dataset covers a second domain"),
    # ]
    # BUILDER_CONFIGS = [
    #     CheMIDatasetConfig(
    #         name='instruction',
    #         tasks=TASKS,
    #         sample_group='instruction_tuning',
    #         description="Molecule instructions.",
    #     ),
    #     CheMIDatasetConfig(
    #         name='raw',
    #         tasks=TASKS,
    #         sample_group=None,
    #         description="Molecule raw data.",
    #     ),
    # ]

    # DEFAULT_CONFIG_NAME = "instruction"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        # This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset

        features = datasets.Features(
            {
                "input": datasets.Value("string"),
                "output": datasets.Value("string"),
                "raw_input": datasets.Value("string"),
                "raw_output": datasets.Value("string"),
                "split": datasets.Value("string"),
                "task": datasets.Value("string"),
                'input_core_tag_left': datasets.Value("string"),
                'input_core_tag_right': datasets.Value("string"),
                'output_core_tag_left': datasets.Value("string"),
                'output_core_tag_right': datasets.Value("string"),
                'target': datasets.Value("string"),
            }
        )
        
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        # urls = _URLS[self.config.name]
        root = dl_manager.download_and_extract('./data.zip')

        sample_group = self.config.sample_group
        insert_core_tags = self.config.insert_core_tags
        use_selfies = self.config.use_selfies
        use_test_subset = self.config.use_test_subset
        use_first = self.config.use_first

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "root": root,
                    "sample_group": sample_group,
                    "split": "train",
                    "tasks": self.config.tasks,
                    "insert_core_tags": insert_core_tags,
                    "use_selfies": use_selfies,
                    "use_test_subset": False,
                    "use_first": use_first,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "root": root,
                    "sample_group": sample_group,
                    "split": "dev",
                    "tasks": self.config.tasks,
                    "insert_core_tags": insert_core_tags,
                    "use_selfies": use_selfies,
                    "use_test_subset": False,
                    "use_first": use_first,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "root": root,
                    "sample_group": sample_group,
                    "split": "test",
                    "tasks": self.config.tasks,
                    "insert_core_tags": insert_core_tags,
                    "use_selfies": use_selfies,
                    "use_test_subset": use_test_subset,
                    "use_first": use_first,
                },
            ),
        ]
    
    def _generate_instruction_examples(self, root, sample_group, split, tasks, insert_core_tags, use_selfies, use_test_subset, use_first):
        key = 0

        if split == 'test' and use_test_subset is True:
            real_split = 'test_subset'
        else:
            real_split = split

        for task in tasks:
            with open(os.path.join(root, 'sample', sample_group, real_split, task + '.json'), 'r') as fs:
                sample_record = json.load(fs)
                assert sample_record['task'] == task, (sample_record['task'], task, os.path.join(root, 'sample', sample_group, real_split, task + '.json'))
                assert sample_record['split'] == real_split
                template_name = sample_record['template_name']
                samples = sample_record['samples']

            with open(os.path.join(root, 'template', template_name, task + '.json'), 'r') as f:
                templates = json.load(f)
            if use_selfies:
                for template in templates:
                    input_template = template['input']
                    output_template = template['output']
                    input_template = input_template.replace("SMILES", "SELFIES")
                    output_template = output_template.replace("SMILES", "SELFIES")
                    template['input'] = input_template
                    template['output'] = output_template

            data = []
            with open(os.path.join(root, 'raw_selfies' if use_selfies else 'raw', split, task + '.jsonl'), 'r') as fr:
                for line in fr:
                    item = json.loads(line)
                    data.append(item)

            with open(os.path.join(root, 'core_tag', task + '.json'), 'r') as f:
                core_tags = json.load(f)
            input_core_tag_left = core_tags['input'][0]
            input_core_tag_right = core_tags['input'][1]
            if use_selfies and input_core_tag_left == '<SMILES>':
                assert input_core_tag_right == '</SMILES>'
                input_core_tag_left = '<SELFIES>'
                input_core_tag_right = '</SELFIES>'
            output_core_tag_left = core_tags['output'][0]
            output_core_tag_right = core_tags['output'][1]
            if use_selfies and output_core_tag_left == '<SMILES>':
                assert output_core_tag_right == '</SMILES>'
                output_core_tag_left = '<SELFIES>'
                output_core_tag_right = '</SELFIES>'
            
            for sample_item in (samples if use_first is None else samples[:use_first]):
                try:
                    data_item = data[sample_item['idx']]
                except IndexError:
                    raise IndexError('In %s for %s, data index exceeds the number of samples. The data size is %d, while the index is %d.' % (real_split, task, len(data), sample_item['idx']))
                assert data_item['task'] == task
                assert data_item['split'] == split
                template_id = sample_item['template_id']
                template = templates[template_id]
                input_template = template['input']
                output_template = template['output']
                input_data = data_item['input']
                if insert_core_tags and input_core_tag_left is not None:
                    assert input_core_tag_right is not None
                    input_data_str = '%s %s %s' % (input_core_tag_left, input_data, input_core_tag_right)
                else:
                    input_data_str = input_data
                input_str = input_template.replace('<INPUT>', input_data_str)
                output_data = data_item['output']
                if isinstance(output_data, str):
                    target = None
                elif isinstance(output_data, dict):
                    target = sample_item['target']
                    output_data = output_data[target]
                else:
                    raise ValueError
                if insert_core_tags and output_core_tag_left is not None:
                    assert output_core_tag_right is not None
                    output_data_str = '%s %s %s' % (output_core_tag_left, output_data, output_core_tag_right)
                else:
                    output_data_str = output_data
                output_str = output_template.replace('<OUTPUT>', output_data_str)
                output_sample = {
                    'input': input_str,
                    'output': output_str,
                    'raw_input': input_data,
                    'raw_output': output_data,
                    'task': task,
                    'split': real_split,
                    'input_core_tag_left': input_core_tag_left,
                    'input_core_tag_right': input_core_tag_right,
                    'output_core_tag_left': output_core_tag_left,
                    'output_core_tag_right': output_core_tag_right,
                    'target': target,
                }
                yield key, output_sample

                key += 1

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, *args, **kwargs):
        # This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.

        return self._generate_instruction_examples(*args, **kwargs)
