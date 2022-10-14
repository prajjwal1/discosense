# coding=utf-8
# Copyright 2022 Prajjwal Bhargava
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

# Lint as: python3
"""Discosense"""


import datasets
import json


_CITATION = """\

"""

_DESCRIPTION = """\
Discosense
"""


class DiscoSenseConfig(datasets.BuilderConfig):
    """BuilderConfig for DiscoSense."""

    def __init__(self, **kwargs):
        """BuilderConfig for DiscoSense.
            Args:
        .
              **kwargs: keyword arguments forwarded to super.
        """
        super(DiscoSenseConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)


class DiscoSense(datasets.GeneratorBasedBuilder):
    """DiscoSense"""

    BUILDER_CONFIGS = [
        DiscoSenseConfig(
            name="plain_text",
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "idx": datasets.Value("int64"),
                    "context": datasets.Value("string"),
                    "marker": datasets.Value("string"),
                    "label": datasets.Value("int64"),
                    "option_0": datasets.Value("string"),
                    "option_1": datasets.Value("string"),
                    "option_2": datasets.Value("string"),
                    "option_3": datasets.Value("string"),
                }
            ),
            # No default supervised_keys (as we have to pass both context
            # and marker as input).
            supervised_keys=None,
            homepage="https://github.com/prajjwal1/discosense",
            citation=_CITATION,
        )

    def _vocab_text_gen(self, filepath):
        for _, ex in self._generate_examples(filepath):
            yield " ".join([ex["context"], ex["marker"]])

    def _split_generators(self, dl_manager):

        train_path = dl_manager.download_and_extract(
           "https://raw.githubusercontent.com/prajjwal1/discosense/main/data/discosense_train.json"
        )
        test_path = dl_manager.download_and_extract(
            "https://raw.githubusercontent.com/prajjwal1/discosense/main/data/discosense_test.json"
        )

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": test_path}),
        ]

    def _generate_examples(self, filepath):
        """Generate Discosense examples.
        Args:
          filepath: a string
        Yields:
          dictionaries containing "context", "marker" and all four options
        """
        for idx, val in enumerate(json.load(open(filepath))):
            yield idx, val
