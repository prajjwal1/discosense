import re
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
    "'",
]

token_fix_set = [
    (" .", "."),
    (" , ", ", "),
    (", , ", ", "),
    ("--", " "),
    (",.", "."),
    (" s ", "s "),
    ("_", " "),
    (" %", "%"),
    ("? ?", "?"),
    (" :", ":"),
    (" ,", ", "),
    ("Jump up ^ ", ""),
]

fix_re = re.compile(r'[a-zA-Z]+[\.,]?|(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:%|st|nd|rd|th)?[\.,]?')
#  fix_re = re.compile(r'[a-zA-Z]+[\.,]?|(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[\.,%]|st|nd|rd|th)?')

def fix_text(text):
    #  text = text.strip()
    for i in tokens_to_remove:
        if i != "``":
            text = text.replace(i, "")
        else:
            text = text.replace(i, " ")

    for k, v in token_fix_set:
        text = text.replace(k, v)

    # Remove space before decimal and exclude all other cases
    text = re.sub(r'(\d)\s*([.])\s*(\d)', '\\1\\2\\3', text)
    #  text = re.sub(r'(\d)\s+([.]\d)', '\\1\\2', text)

    # Add space after comma, but exclude digits
    #  text = re.sub(r"(?<=[0-9])(?=[.^[a-z])", r" ", text)
    text = ' '.join(re.findall(pattern=fix_re, string=text))


    temp = text.split()

    # Remove instance like 2,
    #  if temp[0].isdigit() and text[1] == ',':
        #  text = ' '.join(temp[1:])
    # Remove examples like 6 d, ....
    #  if temp[0].isdigit() and len(temp[1]) == 1:
        #  text = ' '.join(temp[1:])

    text = text.capitalize()
    text = text.strip()
    #  if text:
        #  if text[-2] == ' ' and text[-1] == '.':
            #  text = text.replace(' .', '.')
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
