import sys
import torch
import json
import re
from dataclasses import dataclass, field

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
    dataset_mode: str = field(metadata={"help": "name or path"})
    output_file_path: str = field(metadata={"help": "File path for generated dataset")
    train_pct: Optional[int] = field(default=None)
    valid_pct: Optional[int] = field(default=None)

parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
model_args, data_args = parser.parse_args_into_dataclasses()


if data_args.train_pct:
    discovery_dataset = load_dataset("discovery", "discovery", split= f"train[:{data_args.train_pct}%]")
    print(f"Using {data_args.train_pct} of training data")
else:
    discovery_dataset = load_dataset("discovery", "discovery", split= f"validation[:{data_args.valid_pct}%]")
    print(f"Using {data_args.valid_pct} of validation data")

#  discovery_test_ds = load_dataset("discovery", "discovery", split="test")

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(
    model_path, max_length=96, padding="max_length", add_special_tokens=True
)

tokenizer.add_special_tokens({"pad_token": "[PAD]"})
#  tokenizer.add_special_tokens({"cls_token": "[CLS]"})
#  tokenizer.add_special_tokens({"sep_token": "[SEP]"})


discovery_ds = DatasetGenerate(
    discovery_dataset, model, tokenizer, LABELS, decoding_options
)

synthetic_dataset = []

for i in tqdm(range(len(discovery_dataset))):
    example = {}
    values = discovery_dataset[i]
    example["context"] = values["sentence1"]
    example["marker"] = LABELS[values["label"]]
    generated_options = discovery_ds.generate_synthetic_options(i)
    example.update(generated_options)
    synthetic_dataset.append(example)

with open(output_file_path, "w") as fout:
    json.dump(synthetic_dataset, fout)
