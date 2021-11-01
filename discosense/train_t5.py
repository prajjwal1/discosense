import sys
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from datasets import Dataset
import torch
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from sklearn import metrics
from tqdm.auto import tqdm
from modeling_t5 import T5ForConditionalGeneration


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: Optional[str] = field(
        metadata={
            "help": (
                "Path to pretrained model or model identifier from"
                " huggingface.co/models"
            )
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )


@dataclass
class DataTrainingArguments:
    train_data_path: str = field(metadata={"help": "Path of generated data"})
    validation_data_path: str = field(metadata={"help": "Path for validation set"})
    padding_length: int = field(
        metadata={"help": "Specify the padding size of input sequence"}
    )
    overwrite_cache: bool = field(default=False)


def preprocess_function(examples, tokenizer):
    choice_0, choice_1, choice_2, choice_3 = (
        "1: " + examples["option_0"],
        "2: " + examples["option_1"],
        "3: " + examples["option_2"],
        "4: " + examples["option_3"],
    )

    choices = [choice_0, choice_1, choice_2, choice_3]
    choices = " ".join(choices)
    input_ = "context: %soptions: %s </s>" % (
        #  examples["context"][:-1] + " " + examples["marker"] + ".",
        examples["context"] + " ",
        choices,
    )
    target = "%s </s>" % str(examples["label"]+1)

    encoding = tokenizer(
        [input_],
        return_tensors="pt",
        max_length=data_args.padding_length,
        padding="max_length",
        truncation=True,
    )
    tokenized_target = tokenizer(
        [target], max_length=3, padding="max_length", return_tensors="pt", truncation=True
    )["input_ids"].squeeze()

    encoding["input_ids"] = encoding["input_ids"].squeeze()
    encoding["attention_mask"] = encoding["attention_mask"].squeeze()
    encoding["label"] = tokenized_target

    return encoding


def get_model(model_args):
    model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def get_datasets(data_args):
    train_dataset = Dataset.from_pandas(pd.read_json(data_args.train_data_path))
    validation_dataset = Dataset.from_pandas(
        pd.read_json(data_args.validation_data_path)
    )
    train_dataset = train_dataset.map(
        preprocess_function,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=train_dataset.column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    validation_dataset = validation_dataset.map(
        preprocess_function,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=validation_dataset.column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )
    return train_dataset, validation_dataset


parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

model, tokenizer = get_model(model_args)
train_dataset, validation_dataset = get_datasets(data_args)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
)
trainer.train()

model = trainer.model

outputs, targets = [], []

for sample in tqdm(validation_dataset):
    input_ids = torch.Tensor(sample["input_ids"]).unsqueeze(0).cuda()
    attention_mask = torch.Tensor(sample["attention_mask"]).unsqueeze(0).cuda()
    outs = model.generate(
        input_ids=input_ids, attention_mask=attention_mask, max_length=2
    )

    dec = [tokenizer.decode(ids) for ids in outs]
    target = [tokenizer.decode(ids) for ids in sample["label"]]

    outputs.extend(dec)
    targets.append(target[0])

preds = []
for val in outputs:
    preds.append(val.replace("<pad>", "").strip())

for i, out in enumerate(preds):
    if out not in "1234":
        print(i, "detected invalid prediction", out)

print(metrics.accuracy_score(targets, preds))
