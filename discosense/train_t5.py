from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from datasets import Dataset
from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)


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
        "0: " + examples["option_0"],
        "1: " + examples["option_1"],
        "2: " + examples["option_2"],
        "3: " + examples["option_3"],
    )

    choices = [choice_0, choice_1, choice_2, choice_3]
    choices = " ".join(choices)
    input_ = "context: %s options: %s </s>" % (
        examples["context"] + " " + examples["marker"],
        choices,
    )
    target = examples["label"]

    encoding = tokenizer(
        [input_],
        return_tensors="pt",
        max_length=data_args.padding_length,
        padding="max_length",
        truncation=True,
    )
    encoding['input_ids'] = encoding['input_ids'].squeeze()
    encoding['attention_mask'] = encoding['attention_mask'].squeeze()
    encoding["label"] = target
    return encoding


def get_model(model_args):
    model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
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
