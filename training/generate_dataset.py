import sys
import torch
import json
import re
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm

sys.path.append("..")
from data.discovery_con import LABELS

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser
)
from generate import DatasetGenerate
from config import decoding_options

@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to model"})

@dataclass
class DataTrainingArguments:
    #  dataset_mode: str = field(metadata={"help": "name or path"})
    output_file_path: str = field(metadata={"help": "File path for generated dataset"})
    context_col: str
    to_predict_next_col: str
    marker_col: str
    train_pct_range: Optional[str] = field(default=None)
    valid_pct_range: Optional[str] = field(default=None)

parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
model_args, data_args = parser.parse_args_into_dataclasses()


if data_args.valid_pct_range:
    start_valid_pct, end_valid_pct = map(int, data_args.valid_pct_range.split("-"))
    discovery_dataset = load_dataset("discovery", "discovery", split= f"validation[{start_valid_pct}:{end_valid_pct}%]")
    print(f"Using {start_valid_pct}-{end_valid_pct}% of validation data")
else:
    start_train_pct, end_train_pct = map(int, data_args.train_pct_range.split("-"))
    discovery_dataset = load_dataset("discovery", "discovery", split= f"train[{start_train_pct}:{end_train_pct}%]")
    print(f"Using {start_train_pct}-{end_train_pct}% of training data")

#  discovery_test_ds = load_dataset("discovery", "discovery", split="test")

model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)

# take greedy decoding config's max_length since output is the shortest
tokenizer = AutoTokenizer.from_pretrained(
    model_args.model_name_or_path, max_length=decoding_options[0]['max_length'], padding="max_length", add_special_tokens=True
)

tokenizer.add_special_tokens({"pad_token": "[PAD]"})
#  tokenizer.add_special_tokens({"cls_token": "[CLS]"})
#  tokenizer.add_special_tokens({"sep_token": "[SEP]"})


dataset = DatasetGenerate(
    discovery_dataset, model, tokenizer, LABELS, decoding_options
)

synthetic_dataset = []

for i in tqdm(range(len(dataset))):
    example = {}
    values = dataset[i]
    example["context"] = values[data_args.context_col]
    example["marker"] = values[data_args.marker_col]
    generated_options = discovery_ds.generate_synthetic_options(i, option_id=None, context_col=data_args.context_col, to_predict_next_col=data_args.to_predict_next_col)
    example.update(generated_options)
    synthetic_dataset.append(example)

with open(data_args.output_file_path, "w") as fout:
    json.dump(synthetic_dataset, fout, indent=4)
