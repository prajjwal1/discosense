import re
import torch

import numpy as np
from tqdm import tqdm

from data.discovery_con import LABELS
from utils import convert_dataset_to_json

class DatasetGenerate:
    def __init__(self, dataset, model, tokenizer, labels, decoding_options):
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
                self.tokenizer.decode(
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

        greedy_output = self.model.generate(
            input_ids=input_ids, **self.decoding_options[0]
        )
        beam_output = self.model.generate(
            input_ids=input_ids, **self.decoding_options[1]
        )
        top_p_k_output = self.model.generate(
            input_ids=input_ids, **self.decoding_options[2]
        )

        # Sometimes the greedy output is empty, so replace it with top-p-k
        if self.check_model_output(greedy_output, original_text_length):
            greedy_output, top_p_k_output = self.model.generate(
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
            text = self.tokenizer.decode(
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


class AdversarialFiltering:
    def __init__(self, generate_dataset, model, preds):
        self.generate_dataset = generate_dataset
        self.model = model
        self.preds = preds  # PredictionOutput, contains predictions, label_ids
        self.dataset = convert_dataset_to_json(generate_dataset.dataset)

    def get_solved_dataset(self):
        predictions = []
        for pred in self.preds.predictions:
            predictions.append(np.argmax(pred))

        indices = np.argwhere(self.preds.label_ids == predictions).squeeze().tolist()

        solved_dataset = []
        for idx in tqdm(indices):
            if idx in indices:
                solved_dataset.append(self.dataset[idx])

        return solved_dataset, indices

    def generate_new_samples(self):
        solved_dataset, indices = self.get_solved_dataset()

        for i in tqdm(range(len(solved_dataset))):
            idx = indices[i]
            example = {}
            values = solved_dataset[i]
            example["context"] = values["sentence1"]
            example["marker"] = LABELS[values["label"]]
            generated_options = self.generate_dataset.generate_synthetic_options(i)
            example.update(generated_options)
            self.dataset[idx] = example
