import sys
import torch
import json
import re

from datasets import load_dataset
from tqdm import tqdm

sys.path.append("..")
from data.discovery_con import LABELS

from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/home/nlp/apex/experiment/ctrl/"


discovery_train_ds = load_dataset("discovery", "discovery", split="train[:1%]")
discovery_valid_ds = load_dataset("discovery", "discovery", split="validation")
discovery_test_ds = load_dataset("discovery", "discovery", split="test")

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(
    model_path, max_length=64, padding="max_length", add_special_tokens=True
)

tokenizer.add_special_tokens({"pad_token": "[PAD]"})
tokenizer.add_special_tokens({"cls_token": "[CLS]"})
tokenizer.add_special_tokens({"sep_token": "[SEP]"})


class DiscoveryDatasetGenerate:
    def __init__(self, dataset, labels, tokenizer, model, decoding_options):
        self.dataset = dataset
        self.labels = labels
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.decoding_options = decoding_options
        self.fixed_sequences = 0

    def get_sentence_as_context(self, idx, sentence_order):
        context = self.dataset[idx]
        tokenized_context = self.tokenizer(
            self.labels[context["label"]] + " " + context[sentence_order],
            return_tensors="pt",
        )
        original_text_length = len(
            self.labels[context["label"]] + " " + context[sentence_order]
        )
        return tokenized_context, self.labels[context["label"]], original_text_length

    def check_model_output(self, output_from_model, original_text_length):
        if (
            len(
                tokenizer.decode(
                    output_from_model.squeeze(0), skip_special_tokens=True
                )[original_text_length:]
            )
            < 5
        ):
            self.fixed_sequences += 1
            print("Detected an empty greedy output. Count: ", self.fixed_sequences)
            return True
        return False

    def generate_from_model(self, tokenized_context, original_text_length):
        input_ids = tokenized_context["input_ids"].to(self.device)

        outputs = []

        greedy_output = model.generate(input_ids=input_ids, **self.decoding_options[0])
        beam_output = model.generate(input_ids=input_ids, **self.decoding_options[1])
        top_p_k_output = model.generate(input_ids=input_ids, **self.decoding_options[2])

        # Sometimes the greedy output is empty, so replace it with top-p-k
        if self.check_model_output(greedy_output, original_text_length):
            greedy_output, top_p_k_output = model.generate(
                input_ids=input_ids, **self.decoding_options[3]
            )[:]

        outputs.append(greedy_output)
        outputs.append(beam_output)
        outputs.append(top_p_k_output)

        return outputs

    def cleanup_generated_examples(self, outputs, idx, len_context, marker):
        example = {}
        example["ground_truth"] = self.dataset[idx]["sentence2"]

        for i, sample_output in enumerate(outputs):
            text = tokenizer.decode(
                sample_output.squeeze(0).tolist(), skip_special_tokens=True
            )[len_context:]
            if "." in text:
                prev_input_end_index = text.index(
                    "."
                )  # remove context in generated output
                text = text[prev_input_end_index:]
            text = re.sub("[^A-Za-z0-9\. ]+", "", text)  # remove special characters
            text = text.replace(marker, "")  # 2. remove marker
            if text[0] == ".":
                text = text[1:]
                text = text.replace(".", "").strip()
            example["option_" + str(i)] = text

        return example

    def generate_synthetic_options(self, idx, context="sentence1"):
        tokenized_context, marker, original_text_length = self.get_sentence_as_context(
            idx, context
        )
        len_context = len(self.dataset[idx][context])
        outputs = self.generate_from_model(tokenized_context, original_text_length)
        clean_examples = self.cleanup_generated_examples(
            outputs, idx, len_context, marker
        )
        return clean_examples


decoding_options_0 = {"max_length": 64, "repetition_penalty": 1.2, "temperature": 0}

decoding_options_1 = {
    "max_length": 64,
    "num_beams": 5,
    "no_repeat_ngram_size": 2,
    "early_stopping": True,
}

decoding_options_2 = {
    "max_length": 64,
    "do_sample": True,
    "max_length": 64,
    "top_k": 50,
    "top_p": 0.95,
}

fallback_decoding = {
    "max_length": 64,
    "num_beams": 25,
    "no_repeat_ngram_size": 2,
    "num_return_sequences": 2,
    "temperature": 0.7,
    "early_stopping": True,
}

decoding_options = []
decoding_options.append(decoding_options_0)
decoding_options.append(decoding_options_1)
decoding_options.append(decoding_options_2)
decoding_options.append(fallback_decoding)

discovery_ds = DiscoveryDatasetGenerate(
    discovery_train_ds, LABELS, tokenizer, model, decoding_options
)

synthetic_dataset = []

for i in tqdm(range(1, 7001)):
    example = {}
    values = discovery_train_ds[i]
    example["context"] = values["sentence1"]
    example["marker"] = LABELS[values["label"]]
    generated_options = discovery_ds.generate_synthetic_options(i)
    example.update(generated_options)
    synthetic_dataset.append(example)

    if i==7000:
        with open("../data/ctrl_main.json", "w") as fout:
            json.dump(synthetic_dataset, fout)
        sys.exit(0)
