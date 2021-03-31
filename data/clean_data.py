import re
import sys
import json
from tqdm import tqdm
sys.path.append('..')
from discosense.utils import fix_text

bad_tokens_list = ["\\", "\"", "``", "0\\", "``0", "...", "''"]#    "\u00a7", "\u00b0", "\u00a7", "\u00b0", ]
                   # "\u00e9", "\\\\", "\u00e3", "''", "\u03ba", "\u03b1", "\u0394", "\u00b0", "\u2010", "\u00b4", "\u00d4", "\u00e9", "\u00b5",
                   #  "\u0398", "\u20ac", "\u03b3", "\u00a3", "\u00b1", "\u03c0", "\u00d7", "\u03b2", "\u03b8", "\u0101", "\u00a7", "\u0160", "\u00e2", "\u00c2", "\u011f", "\u00dc", "\u00e7", "\u0131",
                   #  "\u00d5s", "\u00fc", "\u0141", "\u00b1", "\u02da", "\u00b1", "\u00e1", "\u015f", "\u0131", "\u00e0", "\u00e7", "\u1e5b", "\u1e63", "\u1e47", "\u2011", "\u00b64", "\u00c2", "\u00d7", "\u00e1", "\u03b3", "\u03b2", "\u00a7", "\u00fb", "\u00a3", "\u00bf", "\u00a7", "\u00f6", "\u00a3", "\u00ad", "\u03bc", "\u00f3", "\u017c", "\u00e9"]

with open("old/raw_train.json") as f:
    train_data = json.load(f)
with open("old/raw_valid.json") as f:
    valid_data = json.load(f)

def fix_text_init(text):
    for b in bad_tokens_list:
        text = text.replace(b, '')
    if not "(" in text[0:3]:
        text = re.sub("[\(\[].*?[\)\]]", "", text)
    text = (text.encode('ascii', 'ignore')).decode("utf-8")
    text = text.capitalize()
    text = fix_text(text)
    # ensure only one space
    text = re.sub(' +', ' ', text)
    return text

def remove_brackets(text):
    brackets = ["(", ")"]
    for b in brackets:
        text = text.replace(b, "")
    return text

def cleanup(data):
    for idx, d in enumerate(tqdm(data)):
        if len(d["sentence1"]) < 5 or d["sentence2"]) < 5:
            continue
        for b in bad_tokens_list:

            d["sentence1"] = fix_text_init(d["sentence1"]).strip().capitalize()
            d["sentence2"] = fix_text_init(d["sentence2"]).strip().capitalize()

        # After we have brackets that are in between, we remove the front ones
        d["idx"] = idx
        d["sentence1"] = remove_brackets(d["sentence1"])
        d["sentence2"] = remove_brackets(d["sentence2"])

    return data

train_data = cleanup(train_data)
valid_data = cleanup(valid_data)

with open("raw_train.json", "w") as f:
    json.dump(train_data, f, indent=4)
with open("raw_valid.json", "w") as f:
    json.dump(valid_data, f, indent=4)


