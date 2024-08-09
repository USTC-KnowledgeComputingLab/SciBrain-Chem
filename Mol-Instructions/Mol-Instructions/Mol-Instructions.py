"""TODO(MyDataset): Add a description here."""

import json
import os

import datasets


_CITATION = """\
@misc{merity2016pointer,
      title={},
      author={},
      year={2023},
}
"""

_DESCRIPTION = """\
 Mol-Instructions datasets.
"""
_HOMEPAGE = "https://github.com/zjunlp/Mol-Instructions"
_LICENSE = ""
_DATA_URL = "data"


class DatasetConfig(datasets.BuilderConfig):
    def __init__(self, data_url, **kwargs):
        """BuilderConfig for MyDataset

        Args:
          data_url: `string`, url to the dataset (word or raw level)
          **kwargs: keyword arguments forwarded to super.
        """
        super(DatasetConfig, self).__init__(
            version=datasets.Version(
                "1.0.0",
            ),
            **kwargs,
        )
        self.data_url = data_url


class MyDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.1.0")
    BUILDER_CONFIGS = [
        DatasetConfig(
            name="Molecule-oriented Instructions",
            data_url=_DATA_URL + "/" + "Molecule-oriented_Instructions.zip",
            description="Molecule-oriented Instructions",
        ),
        DatasetConfig(
            name="Protein-oriented Instructions",
            data_url=_DATA_URL + "/" + "Protein-oriented_Instructions.zip",
            description="Protein-oriented Instructions",
        ),
        DatasetConfig(
            name="Biomolecular Text Instructions",
            data_url=_DATA_URL + "/" + "Biomolecular_Text_Instructions.zip",
            description="Biomolecular Text Instructions",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "instruction": datasets.Value("string"),
                    "input": datasets.Value("string"),
                    "output": datasets.Value("string"),
                    "metadata": datasets.Value("string"),
                    # These are the features of your dataset like images, labels ...
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if self.config.name == "Molecule-oriented Instructions":
            data_file = dl_manager.download_and_extract(self.config.data_url)
            data_dir = os.path.join(data_file, "Molecule-oriented_Instructions")
            
            return [
                datasets.SplitGenerator(
                    name="description_guided_molecule_design",
                    gen_kwargs={"filepath": os.path.join(data_dir, "description_guided_molecule_design.json")},
                ),
                datasets.SplitGenerator(
                    name="forward_reaction_prediction",
                    gen_kwargs={"filepath": os.path.join(data_dir, "forward_reaction_prediction.json")},
                ),
                datasets.SplitGenerator(
                    name="molecular_description_generation",
                    gen_kwargs={"filepath": os.path.join(data_dir, "molecular_description_generation.json")},
                ),
                datasets.SplitGenerator(
                    name="property_prediction",
                    gen_kwargs={"filepath": os.path.join(data_dir, "property_prediction.json")},
                ),
                datasets.SplitGenerator(
                    name="reagent_prediction",
                    gen_kwargs={"filepath": os.path.join(data_dir, "reagent_prediction.json")},
                ),
                datasets.SplitGenerator(
                    name="retrosynthesis",
                    gen_kwargs={"filepath": os.path.join(data_dir, "retrosynthesis.json")},
                ),
            ]
        elif self.config.name == "Protein-oriented Instructions":
            data_file = dl_manager.download_and_extract(self.config.data_url)
            data_dir = os.path.join(data_file, "Protein-oriented_Instructions")
            
            return [
                datasets.SplitGenerator(
                    name="catalytic_activity",
                    gen_kwargs={"filepath": os.path.join(data_dir, "catalytic_activity.json")},
                ),
                datasets.SplitGenerator(
                    name="domain_motif",
                    gen_kwargs={"filepath": os.path.join(data_dir, "domain_motif.json")},
                ),
                datasets.SplitGenerator(
                    name="protein_function",
                    gen_kwargs={"filepath": os.path.join(data_dir, "protein_function.json")},
                ),
                datasets.SplitGenerator(
                    name="general_function",
                    gen_kwargs={"filepath": os.path.join(data_dir, "general_function.json")},
                ),
                datasets.SplitGenerator(
                    name="protein_design",
                    gen_kwargs={"filepath": os.path.join(data_dir, "protein_design.json")},
                ),
            ]
        elif self.config.name == "Biomolecular Text Instructions":
            data_file = dl_manager.download_and_extract(self.config.data_url)
            data_dir = os.path.join(data_file, "Biomolecular_Text_Instructions")
            
            return [
                datasets.SplitGenerator(
                    # name=datasets.Split.ALL,
                    name="chemical_disease_interaction_extraction",
                    gen_kwargs={"filepath": os.path.join(data_dir, "chemical_disease_interaction_extraction.json")},
                ),
                datasets.SplitGenerator(
                    name="chemical_entity_recognition",
                    gen_kwargs={"filepath": os.path.join(data_dir, "chemical_entity_recognition.json")},
                ),
                datasets.SplitGenerator(
                    name="chemical_protein_interaction_extraction",
                    gen_kwargs={"filepath": os.path.join(data_dir, "chemical_protein_interaction_extraction.json")},
                ),
                datasets.SplitGenerator(
                    name="multi_choice_question",
                    gen_kwargs={"filepath": os.path.join(data_dir, "multi_choice_question.json")},
                ),
                datasets.SplitGenerator(
                    name="open_question",
                    gen_kwargs={"filepath": os.path.join(data_dir, "open_question.json")},
                ),
                datasets.SplitGenerator(
                    name="true_or_false_question",
                    gen_kwargs={"filepath": os.path.join(data_dir, "true_or_false_question.json")},
                ),
            ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        # TODO(sciQ): Yields (key, example) tuples from the dataset
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for id_, row in enumerate(data):
                yield id_, row