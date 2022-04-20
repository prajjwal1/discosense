import sys
import pickle
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

import pandas as pd
import numpy as np
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
import transformers
from transformers import (AutoConfig, AutoModelForCausalLM,
                          AutoModelForMultipleChoice, AutoTokenizer,
                          HfArgumentParser, Trainer, TrainingArguments,
                          default_data_collator, set_seed)
from utils import compute_metrics


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
    data_path: str = field(metadata={"help": "Path of generated data"})
    padding_length: Optional[int] = field(metadata={"help": "Specify the padding size of input sequence"})


def preprocess_function(examples, tokenizer):
    prompt = examples["context"] + " " + examples["marker"]

    choice_0, choice_1, choice_2, choice_3 = (
        examples["option_0"],
        examples["option_1"],
        examples["option_2"],
        examples["option_3"],
    )

    choices = [choice_0, choice_1, choice_2, choice_3]

    encoding = tokenizer(
        [prompt, prompt, prompt, prompt],
        choices,
        return_tensors="pt",
        max_length=data_args.padding_length,
        padding="max_length",
        truncation=True,
    )
    encoding["label"] = examples["label"]

    return encoding


def get_dataset(file_path, preprocess_function, tokenizer):
    dataset = Dataset.from_pandas(
        pd.read_json(file_path)
        )
    non_vector_dataset = dataset
    dataset = dataset.map(
            preprocess_function,
            fn_kwargs={"tokenizer": tokenizer},
            remove_columns=dataset.column_names,
            load_from_cache_file=False,
        )
    return dataset, non_vector_dataset


def get_model(
    model_args
):
    print("Loading the model")

    classification_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path
    )
    classification_model = AutoModelForMultipleChoice.from_pretrained(
        model_args.model_name_or_path, config=classification_config
    )
    classification_tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path
    )

    return classification_model, classification_tokenizer


def perform_inference(trainer):
    preds = trainer.predict(trainer.eval_dataset)
    label_ids = preds.label_ids
    correct_ids, correct_predicted_label, incorrect_ids, incorrect_predicted_label = [], [], [], []
    preds = preds.predictions[0] if isinstance(preds.predictions, tuple) else preds.predictions
    preds = np.argmax(preds, axis=1)

    assert len(preds) == len(label_ids)

    for example_id, (pred, gt) in enumerate(zip(preds, label_ids)):
        if pred == gt:
            correct_ids.append(example_id)
            correct_predicted_label.append(pred)
        else:
            incorrect_ids.append(example_id)
            incorrect_predicted_label.append(pred)

    print('Length of correct IDs: ', len(correct_ids))
    print('Length of incorrect IDs: ', len(incorrect_ids))

    return correct_ids, correct_predicted_label, incorrect_ids, incorrect_predicted_label


parser = HfArgumentParser(
    (ModelArguments, DataTrainingArguments, TrainingArguments)
)
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

model, tokenizer = get_model(model_args)
dataset, non_vector_dataset = get_dataset(data_args.data_path, preprocess_function, tokenizer)

#  hashmap = defaultdict(int)
#  for data in non_vector_dataset:
    #  hashmap[data['marker']] += 1
#  print(hashmap)

trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

correct_ids, correct_predicted_label, incorrect_ids, incorrect_predicted_label = perform_inference(trainer)

non_vector_sent = []

for label in incorrect_ids:
    non_vector_sent.append(non_vector_dataset[label])

with open("mis_preds.pkl", "wb") as f:
    pickle.dump(non_vector_sent, f)

count = 0

#  print('#### Correctly classified examples ####')

#  for idx, correct_id in enumerate(correct_ids):
    #  print(non_vector_dataset[correct_id])
    #  print('Predicted Label: ', correct_predicted_label[idx])
    #  print()
    #  count += 1
    #  if count == 500:
        #  count = 0
        #  break

print('#### Incorrectly classified examples ####')
mis_preds_idx = []

for idx, incorrect_id in enumerate(incorrect_ids):
    #  print(non_vector_dataset[incorrect_id])
    print(incorrect_id)
    mis_preds_idx.append(incorrect_id)

    #  print('Predicted Label: ', incorrect_predicted_label[idx])
    #  print()
    #  count += 1
    #  if count == 1000:
        #  break

with open("mis_preds_idx.pkl", "wb") as f:
    pickle.dump(mis_preds_idx, f)




