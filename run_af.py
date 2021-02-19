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
    TrainingArguments
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
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    autoregressive_model_name_or_path: Optional[str] = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    freeze_encoder: bool = field(default=True)
    classification_config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    classification_tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    autoregressive_config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    autoregressive_tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )


@dataclass
class DataTrainingArguments:
    train_data_path: str = field(metadata={"help": "Path of generated data"})
    validation_data_path: str = field(metadata={"help": "Path for validation set"})
    file_output_path: str = field(metadata={"help":"File Path of generated dataset"})
    replace_dataset: Optional[str] = field(default="validation")

@dataclass
class CustomTrainingArguments(TrainingArguments):
    run_inference_only: Optional[bool] = field(default=False)

def train_classification(generated_train_dataset, generated_validation_dataset, model_args, run_inference_only):
    classification_config = AutoConfig.from_pretrained(model_args.classification_model_name_or_path)
    classification_model = AutoModelForMultipleChoice.from_pretrained(model_args.classification_model_name_or_path, config=classification_config)
    classification_tokenizer = AutoTokenizer.from_pretrained(model_args.classification_model_name_or_path)

    def preprocess_function(examples):
        prompt = examples['context'] + '</s>' + examples['marker']
        choice_0, choice_1, choice_2 = str(examples['ground_truth']), str(examples['option_0']), str(examples['option_1'])
        choice_3 = str(examples['option_2'])
        choices = [choice_0, choice_1, choice_2, choice_3]
        random.shuffle(choices)
        encoding = classification_tokenizer([prompt, prompt, prompt, prompt], choices, return_tensors='pt', max_length=96, padding='max_length', truncation=True)
        encoding["label"] = choices.index(choice_0)
        return encoding

    remove_columns = ['option_0', 'option_1', 'option_2', 'ground_truth']
    print("Performing tokenization of dataset")

    generated_train_dataset = generated_train_dataset.map(preprocess_function, remove_columns=remove_columns)
    generated_validation_dataset = generated_validation_dataset.map(preprocess_function, remove_columns=remove_columns)


    if model_args.freeze_encoder:
        for param in classification_model.roberta.parameters():
            param.requires_grad = False

    trainer = Trainer(
                model=classification_model,
                args=training_args,
                train_dataset=generated_train_dataset,
                compute_metrics=compute_metrics,
                tokenizer=classification_tokenizer,
                data_collator=default_data_collator
            )
    if not run_inference_only:
        print("Training In-Progress")
        trainer.train()
        trainer.save_model()
    print("Running Inference")
    preds = trainer.predict(generated_validation_dataset)
    print(preds.metrics)
    del trainer, classification_model
    return preds

def run_adversarial_filtering(original_dataset, generated_dataset, preds):
    autoregressive_model = AutoModelForCausalLM.from_pretrained(model_args.autoregressive_model_name_or_path)
    autoregressive_tokenizer = AutoTokenizer.from_pretrained(model_args.autoregressive_model_name_or_path)
    generate_dataset_func = DatasetGenerate(original_dataset, autoregressive_model, autoregressive_tokenizer, LABELS, decoding_options)
    af = AdversarialFiltering(generate_dataset_func, autoregressive_model, generated_dataset, preds)
    af.generate_new_samples()
    return af


parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

#  if not training_args.run_inference_only:
generated_train_dataset = Dataset.from_pandas(pd.read_json(data_args.train_data_path))
generated_validation_dataset = Dataset.from_pandas(pd.read_json(data_args.validation_data_path))

preds = train_classification(generated_train_dataset, generated_validation_dataset, model_args, run_inference_only=training_args.run_inference_only)

if data_args.replace_dataset =='validation':
    original_dataset = load_dataset('discovery', 'discovery', split='validation[:2%]')
else:
    original_dataset = load_dataset('discovery', 'discovery', split='train[:2%]')

print("Adversarial Filtering In Progress")

af = run_adversarial_filtering(original_dataset, generated_validation_dataset, preds)

with open(data_args.file_output_path, "w") as fout:
    json.dump(af.generated_dataset, fout)


