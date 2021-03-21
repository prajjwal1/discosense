import os
import json
import random
from typing import Optional
from dataclasses import dataclass, field

import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoModelForMultipleChoice, AutoModelForCausalLM
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    default_data_collator,
    TrainingArguments,
)
from tqdm import tqdm

from generate import DatasetGenerate, AdversarialFiltering
from utils import compute_metrics, convert_dataset_to_json
from config import decoding_options
from data.discovery_con import LABELS


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
    freeze_encoder: bool = field(default=False)
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
    replace_one: Optional[bool] = field(default=False)
    #  replace_dataset: Optional[str] = field(default="validation")

@dataclass
class CustomTrainingArguments(TrainingArguments):
    run_inference_only: Optional[bool] = field(default=False)
    no_af: Optional[bool] = field(default=False)

def preprocess_function(examples, tokenizer, shuffle_labels):
    prompt = examples["context"] + "</s>" + examples["marker"]

    choice_0, choice_1, choice_2, choice_3 = (
        str(examples["ground_truth"]),
        str(examples["option_0"]),
        str(examples["option_1"]),
        str(examples["option_2"])
    )

    choices = [choice_0, choice_1, choice_2, choice_3]

    if shuffle_labels:
        random.shuffle(choices)

    encoding = tokenizer(
        [prompt, prompt, prompt, prompt],
        choices,
        return_tensors="pt",
        max_length=96,
        padding="max_length",
        truncation=True,
    )

    if shuffle_labels:
        encoding["label"] = choices.index(choice_0)
    else:
        encoding["label"] = 0

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
            preprocess_function, fn_kwargs = {'tokenizer': classification_tokenizer, 'shuffle_labels': True}, remove_columns=generated_train_dataset.column_names
        )

    # Different preprocess_function for valid because ground truth should not be removed during AF
    # By default, we will take argmin over preds[1:], this ensures that GT is preserved
    generated_validation_dataset = generated_validation_dataset.map(
        preprocess_function, fn_kwargs = {'tokenizer': classification_tokenizer, 'shuffle_labels': False}, remove_columns=generated_validation_dataset.column_names
    )

    # Only get the input dict provided by tokenizer, remove columns which contains text (only ones that have tensor).

    #  if model_args.freeze_encoder:
        #  for param in classification_model.roberta.parameters():
            #  param.requires_grad = False

    trainer = Trainer(
        model=classification_model,
        args=training_args,
        train_dataset=generated_train_dataset,
        eval_dataset = generated_validation_dataset,
        compute_metrics=compute_metrics,
        tokenizer=classification_tokenizer,
        data_collator=default_data_collator,
    )
    if not run_inference_only:
        print("Training In-Progress")
        trainer.train()

        if not os.path.isdir(training_args.output_dir) :
            os.mkdir(training_args.output_dir)
        trainer.save_model(training_args.output_dir)
    #  print("Running Inference")
    preds = trainer.predict(generated_validation_dataset)
    print(preds.metrics)
    return preds


def run_adversarial_filtering(original_dataset, generated_dataset, preds, actions_col):
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
        replace_one=data_args.replace_one
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

generated_train_dataset = Dataset.from_pandas(pd.read_json(data_args.train_data_path))
generated_validation_dataset = Dataset.from_pandas(
    pd.read_json(data_args.validation_data_path)
)

preds = train_classification(
    generated_train_dataset,
    generated_validation_dataset,
    model_args,
    run_inference_only=training_args.run_inference_only,
)

# If you need to run AF on Training set
# Not working since we also need to know label indices of Ground truth.
# GTs are shuffled in training set, so np.argmin can give index of GT which AF can remove

#  if data_args.replace_dataset == "validation":
original_dataset = Dataset.from_pandas(pd.read_json(data_args.raw_data_path))
#  else:
    #  original_dataset = load_dataset("discovery", "discovery", split="train[:7%]")


if not training_args.no_af:
    print("Adversarial Filtering In Progress")

    actions_col = [data_args.context_col, data_args.to_predict_col, data_args.marker_col]
    generated_samples = run_adversarial_filtering(
        original_dataset, generated_validation_dataset, preds, actions_col
    )

    with open(data_args.file_output_path, "w") as fout:
        json.dump(generated_samples, fout)
