import numpy as np

tokens_to_remove = [
    "@@",
    "@",
    "``",
    "$",
    ":",
    '"',
    "``",
    "(",
    ")",
    ";",
    "\r",
    "\n",
    "\\",
    "'"
]

def fix_text(text):
    text = text.strip()
    for i in tokens_to_remove:
        if i != "``":
            text = text.replace(i, "")
        else:
            text = text.replace(i, " ")
    text = text.replace(" .", ".")
    text = text.replace(" , ",", ")
    text = text.replace(", , ", ", ")
    text = text.replace("--", " ")
    text = text.replace(" %", "%")
    text = text.replace("? ?", "?")
    text = text.replace(" :", ":")
    text = text.replace(" ,", ", ")
    text = text.capitalize()
    return text


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







#      "\u00a3",
    #  "\u0394",
    #  "\u03b1",
    #  "\u2010",
    #  "\u03b3",
    #  "\u03b2",
    #  "\u0391",
    #  "\u00a7",
    #  "\u2010",
    #  "\u00f3",
    #  "\u00b0",
    #  "\u00ef",
    #  "\u03ba",
    #  "\u03b1",
    #  "\u0394",
    #  "\u20ac",
    #  "\u00e9",
    #  "\u00b0",
    #  "\u2122",
    #  "\u00ef",
    #  "\u00e9",

