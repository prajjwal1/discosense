import re
from string import digits
import torch

import numpy as np
from tqdm import tqdm

from data.discovery_con import LABELS
from utils import convert_dataset_to_json


class DatasetGenerate:
    def __init__(self, dataset, model, tokenizer, decoding_options, required_col):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.decoding_options = decoding_options
        self.fixed_sequences = 0
        self.context_col, , self.marker_col = required_col.split(',')
        self.remove_digits = str.maketrans("", "", digits)

    def get_sentence_as_context(self, idx):
        context = self.dataset[idx]
        tokenized_context = self.tokenizer(
            self.labels[self.marker_col] + " " + context[self.context_col],
            return_tensors="pt",
        )
        original_text_length = len(
            self.labels[context[marker_col]] + " " + context[context_col]
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
            return True
        return False

    def generate_from_model(self, tokenized_context, original_text_length, option_id):
        """
        Generate options from Autoregressive LM
        Input:
            tokenized_context: Context in tokenized form
            original_text_length: Text length to check empty output
            option_id: Whether to generate one or all options. Usually kept to True during AF.
        Output:
            output: Generated output from LM
        """
        input_ids = tokenized_context["input_ids"].to(self.device)

        output = []
        if option_id is not None:
            # Use greedy for AF (replace one) option
            output.append(self.model.generate(
                input_ids=input_ids, **self.decoding_options[0]
            ))
        else:
            output = []
            for i in range(3):
                output.append(self.model.generate(
                    input_ids=input_ids, **self.decoding_options[i]
                ))
            # Sometimes the greedy output is empty, so replace it with top-p-k
        if self.check_model_output(output[0], original_text_length):
            output[0] = self.model.generate(
                input_ids=input_ids, **self.decoding_options[-1]
            )
        return output

    def cleanup_generated_examples(
        self, outputs, idx, len_context, marker, option_id=None
    ):
        example = {}
        for i, sample_output in enumerate(outputs):
            text = self.tokenizer.decode(              #[1,max_len]
                sample_output.squeeze(0).tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )[len_context:]

            if "." in text:
                prev_input_end_index = text.index(
                    "."
                )  # remove context in generated output
                text = text[prev_input_end_index:]
            text = re.sub("[^A-Za-z0-9\. ]+", "", text)  # remove special characters, preserves .
            text = text.replace(marker, "")  # 2. remove marker
            if text[0] == ".":
                text = text[1:]
                #  text = text.replace(".", "").strip()

            # check for numbers that model generates, only 2 nums allowed
            if sum(c.isdigit() for c in text) > 2:
                text = text.translate(self.remove_digits)

            if option_id is not None:
                example["option_" + str(option_id)] = text
                break
            else:
                example["ground_truth"] = self.dataset[idx][self.to_predict_col]
                example["option_" + str(i)] = text

        return example

    def generate_synthetic_options(self, idx, option_id):
        tokenized_context, marker, original_text_length = self.get_sentence_as_context(idx)
        len_context = len(self.dataset[idx][self.context_col])
        output = self.generate_from_model(
            tokenized_context, original_text_length, option_id=option_id
        )
        clean_examples = self.cleanup_generated_examples(
            output, idx, len_context, marker, option_id=option_id
        )
        return clean_examples


class AdversarialFiltering:
    def __init__(self, generate_dataset_func, model, generated_dataset, preds):
        self.generate_dataset_func = generate_dataset_func
        self.model = model
        self.preds = preds  # PredictionOutput, contains predictions, label_ids
        self.original_dataset = convert_dataset_to_json(
            self.generate_dataset_func.dataset
        )
        self.generated_dataset = convert_dataset_to_json(generated_dataset)

    def get_solved_dataset(self, return_dict):
        """
        Checks which indices have been correctly classified

        Input:
            return_dict: Whether to get option_ids with low confidence scores

        Returns:
            solved_dataset: subsample of original dataset which was classified correctly
            indices: absolute indices in original dataset along with options with low confidence score
        """
        predictions = []
        for pred in self.preds.predictions:
            predictions.append(np.argmax(pred))

        indices = np.argwhere(self.preds.label_ids == predictions).squeeze().tolist()

        # It should be compatible with training set (can contain shuffled options),
        # If the model's certaintly on GT is least, remove the second least confident option

        if return_dict:
            indices_dict = {}

            #  predictions_logit, label_ids = self.preds.predictions, self.preds.label_ids

            for idx, pred in enumerate(self.preds.predictions):
                if idx in indices:
                    # Remove ground truth from pred
                    # GT is located at 0th index
                    indices_dict[idx] = np.argmin(pred[1:])

        solved_dataset = []
        for idx in tqdm(indices):
            if idx in indices:
                solved_dataset.append(self.original_dataset[idx])

        assert len(solved_dataset) == len(indices_dict)
        print(len(indices), " were classified correctly ")

        return solved_dataset, indices_dict if return_dict else indices

    def generate_new_samples(self, context_col, to_predict_next_col, replace_one):
        """
        Responsible for replacing correctly classified options

        Input:
            context: key from dataset to be used as context
            to_predict_next: key from dataset to be predicted
            replace_one: turn on one replacement mode where the options with low confidence scores are replaced
        Output:
            generated_dataset: return new dataset created from AF
        """
        print("Replace One Mode set to: ", replace_one)
        solved_dataset, indices_val = self.get_solved_dataset(return_dict=replace_one)
        if replace_one:
            indices, option_ids = list(indices_val.keys()), list(indices_val.values())
        else:
            indices, option_ids = indices_val, [None] * len(indices_val)

        for i in tqdm(range(len(solved_dataset))):
            idx, option_id = indices[i], option_ids[i]
            example = {}
            values = solved_dataset[i]
            example["context"] = values[context_col]
            example["marker"] = LABELS[values["label"]]

            generated_output = self.generate_dataset_func.generate_synthetic_options(
                idx, option_id=option_id, context_col=context_col, to_predict_next_col=to_predict_next_col
            )

            if replace_one:
                assert len(generated_output) == 1 # returns ground_truth and option_id example

                for k, v in generated_output.items():
                    assert k in self.generated_dataset[i].keys()
                    self.generated_dataset[idx][k] = v
            else:
                example.update(generated_output)
                self.generated_dataset[idx] = example

        print("Found ", self.generate_dataset_func.fixed_sequences, " empty greedy output(s)")
        return self.generated_dataset
