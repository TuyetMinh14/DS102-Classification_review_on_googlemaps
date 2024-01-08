# coding=utf-8
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
"""The Hospital dataset for text classification."""


import csv

import json

import datasets


# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
None
}
"""

_DESCRIPTION = """\
None
"""

_HOMEPAGE = "None"

_LICENSE = "None"

_URLs = {
    "hospital": "./hospital.zip",
}


class HospitalConfig(datasets.BuilderConfig):
    """BuilderConfig for Hospital."""

    def __init__(self, **kwargs):
        """BuilderConfig for Hospital.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(HospitalConfig, self).__init__(**kwargs)


class Hospital(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        HospitalConfig(
            name="hospital", version=VERSION, description="Hospital dataset"
        ),
    ]

    def _info(self):
        features = datasets.Features(
            {
                "text": datasets.Value("string"),
                # "label": datasets.Value("string")
                "label": datasets.features.ClassLabel(
                    num_classes=4,
                    names=[
                        "positive",
                        "negative",
                        "neutral",
                        "other",
                    ]
                ),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        my_urls = _URLs[self.config.name]
        archive = dl_manager.download(my_urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": "train.json",
                    "files": dl_manager.iter_archive(archive),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": "test.json",
                    "files": dl_manager.iter_archive(archive),
                },
            ),
        ]

    def _generate_examples(self, filepath, files):
        """Yields examples."""
        for path, f in files:
            if path == filepath:
                data = json.load(f) 
                for id_, row in enumerate(data):
                    if isinstance(row["text"], str):
                        yield id_, {
                            "text": row["text"],
                            "label": row["label"],
                        }
                break