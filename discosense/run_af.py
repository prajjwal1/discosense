import json
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import (AutoConfig, AutoModelForCausalLM,
                          AutoModelForMultipleChoice, AutoTokenizer,
                          HfArgumentParser, Trainer, TrainingArguments,
                          default_data_collator, set_seed)

from config import decoding_options
from generate import AdversarialFiltering, DatasetGenerate
from utils import compute_metrics, convert_dataset_to_json

#  from data.discovery_con import LABELS


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    classification_model_name_or_path: Optional[str] = field(
        metadata={
            "help": (
                "Path to pretrained model or model identifier from"
                " huggingface.co/models"
            )
        }
    )
    autoregressive_model_name_or_path: Optional[str] = field(
        metadata={
            "help": (
                "Path to pretrained model or model identifier from"
                " huggingface.co/models"
            )
        }
    )
    classification_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    classification_tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    autoregressive_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    autoregressive_tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )


@dataclass
class DataTrainingArguments:
    train_data_path: str = field(metadata={"help": "Path of generated data"})
    validation_data_path: str = field(metadata={"help": "Path for validation set"})
    file_output_path: str = field(metadata={"help": "File Path of generated dataset"})
    raw_data_path: str = field(metadata={"help": "Path of raw data"})
    context_col: Optional[str]
    to_predict_col: Optional[str]
    marker_col: Optional[str]
    overwrite_cache: bool = field(default=False)
    replace_one: Optional[bool] = field(default=False)


@dataclass
class CustomTrainingArguments(TrainingArguments):
    run_inference_only: Optional[bool] = field(default=False)
    no_af: Optional[bool] = field(default=False)
    random_seed: Optional[bool] = field(default=False)

def preprocess_function(examples, tokenizer, shuffle_labels):
    prompt = examples["context"] + " " + examples["marker"]

    choice_0, choice_1, choice_2, choice_3 = (
        examples["option_0"],
        examples["option_1"],
        examples["option_2"],
        examples["option_3"],
    )

    choices = [choice_0, choice_1, choice_2, choice_3]

    #  if shuffle_labels:
        #  random.shuffle(choices)

    encoding = tokenizer(
        [prompt, prompt, prompt, prompt],
        choices,
        return_tensors="pt",
        max_length=96,
        padding="max_length",
        truncation=True,
    )

#      if shuffle_labels:
        #  encoding["label"] = choices.index(choice_0)
    #  else:
#          encoding["label"] = 0
    encoding["label"] = examples["label"]

    return encoding


def train_classification(
    generated_train_dataset,
    generated_validation_dataset,
    model_args,
    run_inference_only,
):
    classification_config = AutoConfig.from_pretrained(
        model_args.classification_model_name_or_path
    )
    classification_model = AutoModelForMultipleChoice.from_pretrained(
        model_args.classification_model_name_or_path, config=classification_config
    )
    classification_tokenizer = AutoTokenizer.from_pretrained(
        model_args.classification_model_name_or_path
    )

    print("Performing tokenization of dataset")

    if not run_inference_only:
        generated_train_dataset = generated_train_dataset.map(
            preprocess_function,
            fn_kwargs={"tokenizer": classification_tokenizer, "shuffle_labels": True},
            remove_columns=generated_train_dataset.column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Different preprocess_function for valid because ground truth should not be removed during AF
    # By default, we will take argmin over preds[1:], this ensures that GT is preserved
    generated_validation_dataset = generated_validation_dataset.map(
        preprocess_function,
        fn_kwargs={"tokenizer": classification_tokenizer, "shuffle_labels": False},
        remove_columns=generated_validation_dataset.column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    trainer = Trainer(
        model=classification_model,
        args=training_args,
        train_dataset=generated_train_dataset,
        eval_dataset=generated_validation_dataset,
        compute_metrics=compute_metrics,
        tokenizer=classification_tokenizer,
        data_collator=default_data_collator,
    )
    if not run_inference_only:
        print("Training In-Progress")
        train_result = trainer.train()
        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        if not os.path.isdir(training_args.output_dir):
            os.mkdir(training_args.output_dir)
        trainer.save_model(training_args.output_dir)

    preds = trainer.predict(generated_validation_dataset)
    print(preds.metrics)

    return preds


def tests_check(original_dataset, generated_dataset):
    assert len(original_dataset) == len(generated_dataset)

    random_nums = []
    for i in range(100):
        random_nums.append(random.randint(0, len(original_dataset)))
    for i in random_nums:
        assert (
            original_dataset[i]['sentence1']
            == generated_dataset[i]['context']
        )
        assert (
            original_dataset[i]['sentence2']
            == generated_dataset[i]['ground_truth']
        )
        assert (
            original_dataset[i][data_args.marker_col]
            == original_dataset[i][data_args.marker_col]
        )


def run_adversarial_filtering(original_dataset, generated_dataset, preds, actions_col):
    tests_check(original_dataset, generated_dataset)

    autoregressive_model = AutoModelForCausalLM.from_pretrained(
        model_args.autoregressive_model_name_or_path
    )
    autoregressive_tokenizer = AutoTokenizer.from_pretrained(
        model_args.autoregressive_model_name_or_path
    )
    generate_dataset_func = DatasetGenerate(
        original_dataset,
        autoregressive_model,
        autoregressive_tokenizer,
        decoding_options,
        actions_col,
        replace_one=data_args.replace_one,
    )
    af = AdversarialFiltering(
        generate_dataset_func, autoregressive_model, generated_dataset, preds
    )
    generated_samples = af.generate_new_samples()
    return generated_samples


parser = HfArgumentParser(
    (ModelArguments, DataTrainingArguments, CustomTrainingArguments)
)
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

if training_args.random_seed:
    random_seed_int = random.randint(0, 1000)
    print("Setting the seed ", random_seed_int)
    set_seed(random_seed_int)

if not training_args.run_inference_only:
    print('Getting train data from ', data_args.train_data_path)
    generated_train_dataset = Dataset.from_pandas(
        pd.read_json(data_args.train_data_path)
    )
else:
    generated_train_dataset = None

print('Getting validation data from ', data_args.validation_data_path)
generated_validation_dataset = Dataset.from_pandas(
    pd.read_json(data_args.validation_data_path)
)

preds = train_classification(
    generated_train_dataset,
    generated_validation_dataset,
    model_args,
    run_inference_only=training_args.run_inference_only,
)

if not training_args.no_af:
    original_dataset = Dataset.from_pandas(pd.read_json(data_args.raw_data_path))
    print("Adversarial Filtering In Progress")

    actions_col = [
        data_args.context_col,
        data_args.to_predict_col,
        data_args.marker_col,
    ]
    generated_samples = run_adversarial_filtering(
        original_dataset, generated_validation_dataset, preds, actions_col
    )

    with open(data_args.file_output_path, "w") as fout:
        json.dump(generated_samples, fout, indent=4)
