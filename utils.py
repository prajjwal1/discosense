import random
import numpy as np

def convert_dataset_to_json(dataset):
    dataset_list = []
    for dict_val in dataset:
        val = {}
        for k, v in dict_val.items():
            val[k] = v
        dataset_list.append(dict_val)
    return dataset_list

def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


