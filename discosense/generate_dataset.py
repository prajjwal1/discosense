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

from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from generate import DatasetGenerate
from config import decoding_options, token_max_length


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to model"})

@dataclass
class DataTrainingArguments:
    raw_data: str
    output_file_path: str = field(metadata={"help": "File path for generated dataset"})
    context_col: str
    to_predict_col: str
    marker_col: str
    option_id: int
    generate_new: Optional[bool] = field(default=False)
    resume_gen_file: Optional[str] = field(default=None)
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

print("Option ID has been set to ", data_args.option_id)
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

if not data_args.resume_gen_file:
    synthetic_dataset = []
else:
    print("Loading existing data from ", data_args.resume_gen_file)
    with open(data_args.resume_gen_file, "r") as f:
        synthetic_dataset = json.load(f)

print("Generation in progress")

for idx in tqdm(range(len(dataset))):
    example = {}
    values = dataset[idx]
    example["context"] = values[data_args.context_col]
    example["marker"] = values[data_args.marker_col]
    example["idx"] = idx
    generated_options = dataset_gen_func.generate_synthetic_options(idx, option_id=data_args.option_id)
    generated_options["ground_truth"] = values[data_args.to_predict_col]
    if data_args.option_id is not None and not data_args.generate_new:
        #  example[data_args.option_id] = generated_options
        synthetic_dataset[idx].update(generated_options)
        for k, v in synthetic_dataset[idx].items():
            print(k, v)
    else:
        example.update(generated_options)
        synthetic_dataset.append(example)
        for k, v in example.items():
            print(k, v)
    print()


with open(data_args.output_file_path, "w") as fout:
    json.dump(synthetic_dataset, fout, indent=4)
