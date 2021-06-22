import re
from string import digits
import torch

import numpy as np
from tqdm import tqdm

from utils import convert_dataset_to_json, tokens_to_remove, fix_text


class DatasetGenerate:
    def __init__(
        self, dataset, model, tokenizer, decoding_options, required_cols, replace_one
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.decoding_options = decoding_options
        self.context_col, self.to_predict_col, self.marker_col = required_cols
        self.replace_one = replace_one
        self.init_common_attrs()

    def init_common_attrs(self):
        self.remove_digits = str.maketrans("", "", digits)
        self.filter_tokens_list = tokens_to_remove

    def get_sentence_as_context(self, idx):
        """
        This function performs tokenization of the requested example

        Args:
            idx
        Returns:
            tokenized_context
        """
        example = self.dataset[idx]
        input_text = example[self.marker_col] + " " + example[self.context_col]
        tokenized_context = self.tokenizer(input_text, return_tensors="pt")
        return input_text, tokenized_context

    def generate_from_model(self, tokenized_context, text_length, option_id):
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

        for i in range(len(self.decoding_options)):
            self.decoding_options[i]["max_length"] = text_length+10

        output = []
        if option_id is not None:
            output.append(
                self.model.generate(input_ids=input_ids, **self.decoding_options[-1])
            )
        else:
            output = []
            for i in range(3):
                output.append(
                    self.model.generate(input_ids=input_ids, **self.decoding_options[i])
                )
        return output

    def cleanup_generated_examples(self, outputs, input_text, idx, option_id=None):
        if option_id is None:
            example = {}
            example["ground_truth"] = self.dataset[idx][self.to_predict_col][:-1]

        for i, sample_output in enumerate(outputs):
            text = self.tokenizer.decode(  # [1, max_len]
                sample_output.squeeze(0).tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            # Remove input text from the generation
            text = text.replace(input_text, "")

            if "?" in text:
                text = text[: text.index("?")]
            if "." in text:
                text = text[: text.index(".")]

            # Remove (), [] and text within it and then apply text
            text = fix_text(text)

            if option_id is not None:
                example = {}
                if text and text[-1] == ".":
                    text = text[:-1]
                text += '.'
                example["option_" + str(option_id)] = text.strip()
                return example

            example["option_" + str(i)] = text.strip()

        return example

    def generate_synthetic_options(self, idx, option_id):
        input_text, tokenized_context = self.get_sentence_as_context(idx)
        output = self.generate_from_model(
            tokenized_context, len(input_text), option_id=option_id
        )
        clean_examples = self.cleanup_generated_examples(
            output, input_text, idx, option_id=option_id
        )
        return clean_examples


class AdversarialFiltering:
    def __init__(self, generate_dataset_func, model, generated_dataset, preds):
        self.generate_dataset_func = generate_dataset_func
        self.model = model
        self.preds = preds  # PredictionOutput, contains predictions, label_ids
        self.raw_dataset = convert_dataset_to_json(
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

        # Shuffling is not supported for Validation set. GT is expected to be first option
        if return_dict:
            indices_dict = {}

            #  predictions_logit, label_ids = self.preds.predictions, self.preds.label_ids

            for idx, pred in enumerate(self.preds.predictions):
                if idx in indices:
                    # Remove ground truth from pred
                    # GT is located at 0th index
                    indices_dict[idx] = np.argmin(pred[1:])

        solved_dataset = []
        for idx in tqdm(range(len(self.raw_dataset))):
            if idx in indices:
                solved_dataset.append(self.raw_dataset[idx])

        if return_dict:
            assert len(solved_dataset) == len(indices_dict)
        print(len(indices), " were classified correctly ")

        return solved_dataset, indices_dict if return_dict else indices

    def generate_new_samples(self):
        """
        Responsible for replacing correctly classified options

        Input:
            context: key from dataset to be used as context
            to_predict_next: key from dataset to be predicted
            replace_one: turn on one replacement mode where the options with low confidence scores are replaced
        Output:
            generated_dataset: return new dataset created from AF
        """
        replace_one = self.generate_dataset_func.replace_one
        print("Replace One mode set to: ", replace_one)

        solved_dataset, indices_val = self.get_solved_dataset(return_dict=replace_one)

        if replace_one:
            indices, option_ids = list(indices_val.keys()), list(indices_val.values())
        else:
            indices, option_ids = indices_val, [None] * len(indices_val)

        for i in tqdm(range(len(solved_dataset))):
            idx, option_id = indices[i], option_ids[i]
            generated_output = self.generate_dataset_func.generate_synthetic_options(
                idx, option_id=option_id
            )

            if self.generate_dataset_func.replace_one:
                assert (
                    len(generated_output) == 1
                )  # returns option_id example

                for k, v in generated_output.items():
                    assert k in self.generated_dataset[idx].keys()
                    #  print(idx, self.generated_dataset[idx]['context'])
                    #  print(self.generated_dataset[idx][k])
                    #  print(v)
                    self.generated_dataset[idx][k] = v
                    #  print()
            else:
                example = {}
                values = solved_dataset[i]
                example[self.generate_dataset_func.context_col] = values[
                    self.generate_dataset_func.context_col
                ]
                example[self.generate_dataset_func.marker_col] = values[
                    self.generate_dataset_func.marker_col
                ]
                example.update(generated_output)
                self.generated_dataset[idx] = example

        return self.generated_dataset
