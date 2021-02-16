import random
from typing import Optional
from dataclasses import dataclass, field

from transformers import AutoModelForMultipleChoice, AutoModelForCausalLM
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser
)
from datasets import load_dataset
from tqdm import tqdm

from generate import DatasetGenerate, AdversarialFiltering
from utils import compute_metrics, convert_dataset_to_json

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    classification_model_name_or_path: Optional[str] = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    classification_config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    classification_tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    autoregressive_model_name_or_path: Optional[str] = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    autoregressive_config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    autoregressive_tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()


classification_config = AutoConfig.from_pretrained(model_args.classification_model_name_or_path)
classification_model = AutoModelForMultipleChoice(model_args.classification_model_name_or_path)
classification_tokenizer = AutoTokenizer.from_pretrained(model_args.classification_model_name_or_path)

def preprocess_function(examples):
    prompt = examples['context'] + '</s>' + examples['marker']
    choice_0, choice_1, choice_2 = str(examples['ground_truth']), str(examples['option_0']), str(examples['option_1'])
    choice_3 = str(examples['option_2'])
    choices = [choice_0, choice_1, choice_2, choice_3]
    random.shuffle(choices)
    encoding = tokenizer([prompt, prompt, prompt, prompt], choices, return_tensors='pt', max_length=96, padding='max_length', truncation=True)
    encoding["label"] = choices.index(choice_0)
    return encoding




