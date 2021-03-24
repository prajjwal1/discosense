import sys
import torch
import json
import re
from dataclasses import dataclass, field
from typing import Optional

from datasets import Dataset
from tqdm import tqdm
import pandas as pd

sys.path.append("..")
from data.discovery_con import LABELS

from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from generate import DatasetGenerate
from config import decoding_options, token_max_length


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to model"})


@dataclass
class DataTrainingArguments:
    #  dataset_mode: str = field(metadata={"help": "name or path"})
    raw_data: str
    output_file_path: str = field(metadata={"help": "File path for generated dataset"})
    context_col: str
    to_predict_col: str
    marker_col: str
    train_pct_range: Optional[str] = field(default=None)
    valid_pct_range: Optional[str] = field(default=None)


parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
model_args, data_args = parser.parse_args_into_dataclasses()


if data_args.valid_pct_range:
    start_valid_pct, end_valid_pct = map(int, data_args.valid_pct_range.split("-"))
    dataset = Dataset.from_pandas(pd.read_json(data_args.raw_data))
    ranges = list(range(len(dataset)))
    dataset = dataset.select(
        ranges[
            int(start_valid_pct * len(ranges) * 0.01) : int(
                end_valid_pct * len(ranges) * 0.01
            )
        ]
    )
    print(f"Using {start_valid_pct}-{end_valid_pct}% of validation data", len(dataset))
else:
    start_train_pct, end_train_pct = map(int, data_args.train_pct_range.split("-"))
    dataset = Dataset.from_pandas(pd.read_json(data_args.raw_data))
    ranges = list(range(len(dataset)))
    dataset = dataset.select(
        ranges[
            int(start_train_pct * len(ranges) * 0.01) : int(
                end_train_pct * len(ranges) * 0.01
            )
        ]
    )
    print(f"Using {start_train_pct}-{end_train_pct}% of training data", len(dataset))

#  discovery_test_ds = load_dataset("discovery", "discovery", split="test")
print("Loading the model")
model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)

# take greedy decoding config's max_length since output is the shortest
tokenizer = AutoTokenizer.from_pretrained(
    model_args.model_name_or_path,
    max_length=token_max_length,
    padding="max_length",
    add_special_tokens=True,
)

tokenizer.add_special_tokens({"pad_token": "[PAD]"})
#  tokenizer.add_special_tokens({"cls_token": "[CLS]"})
#  tokenizer.add_special_tokens({"sep_token": "[SEP]"})


dataset_gen_func = DatasetGenerate(
    dataset,
    model,
    tokenizer,
    decoding_options,
    [data_args.context_col, data_args.to_predict_col, data_args.marker_col],
    replace_one=None,
)

synthetic_dataset = []


print("Generation in progress")

for i in tqdm(range(len(dataset))):
    example = {}
    values = dataset[i]
    example["context"] = values[data_args.context_col]
    example["marker"] = values[data_args.marker_col]
    generated_options = dataset_gen_func.generate_synthetic_options(i, option_id=None)
    example.update(generated_options)
    #  for k, v in example.items():
        #  print(v)
    #  print()
    synthetic_dataset.append(example)

with open(data_args.output_file_path, "w") as fout:
    json.dump(synthetic_dataset, fout, indent=4)
