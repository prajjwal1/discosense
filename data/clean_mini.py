import re
import sys
import json
from tqdm import tqdm
sys.path.append('..')
#  from wordninja import split
from wordsegment import load, segment
from discosense.utils import fix_text

bad_tokens_list = ["\\", "\"", "``", "0\\", "``0", "...", "''"]#    "\u00a7", "\u00b0", "\u00a7", "\u00b0", ]
                   # "\u00e9", "\\\\", "\u00e3", "''", "\u03ba", "\u03b1", "\u0394", "\u00b0", "\u2010", "\u00b4", "\u00d4", "\u00e9", "\u00b5",
                   #  "\u0398", "\u20ac", "\u03b3", "\u00a3", "\u00b1", "\u03c0", "\u00d7", "\u03b2", "\u03b8", "\u0101", "\u00a7", "\u0160", "\u00e2", "\u00c2", "\u011f", "\u00dc", "\u00e7", "\u0131",
                   #  "\u00d5s", "\u00fc", "\u0141", "\u00b1", "\u02da", "\u00b1", "\u00e1", "\u015f", "\u0131", "\u00e0", "\u00e7", "\u1e5b", "\u1e63", "\u1e47", "\u2011", "\u00b64", "\u00c2", "\u00d7", "\u00e1", "\u03b3", "\u03b2", "\u00a7", "\u00fb", "\u00a3", "\u00bf", "\u00a7", "\u00f6", "\u00a3", "\u00ad", "\u03bc", "\u00f3", "\u017c", "\u00e9"]

with open("raw_train.json") as f:
    train_data = json.load(f)
with open("raw_valid.json") as f:
    valid_data = json.load(f)

load()
def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def cleanup(data):
    cnt = 0
    remove_examples = []
    for idx, d in enumerate(tqdm(data)):
#          if len(d["sentence1"]) < 10 or  len(d["sentence2"]) < 10:
            #  remove_examples.append(idx)
            #  print(idx)
        #  else:
            #  d["sentence1"] = fix_text(d["sentence1"])
            #  d["sentence2"] = fix_text(d["sentence2"])
#              sentence1 = d["sentence1"]
            #  sentence2 = d["sentence2"]

            #  if not hasNumbers(sentence1) and ', ' not in sentence1:
                #  text1 = ' '.join(segment(sentence1))+'.'
                #  text1 = text1.capitalize()
                #  if text1 != sentence1:
                    #  print("original: ", sentence1)
                    #  print("new: ", text1)
                    #  value = input("Change? ")
                    #  if value == "y":
                        #  d["sentence1"] = text1

            #  if not hasNumbers(sentence2) and ', ' not in sentence2:
                #  text2 = ' '.join(segment(sentence2))+'.'
                #  text2 = text2.capitalize()
                #  if text2 != sentence2:
                    #  print("original: ", sentence2)
                    #  print("new: ", text2)
                    #  value = input("Change? ")
                    #  if value == "y":
                        #  d["sentence2"] = text2

            d["idx"] = cnt
            cnt += 1

    return data, remove_examples

#  train_data, remove_examples = cleanup(train_data)
#  for i in remove_examples:
#      del train_data[i]
valid_data, remove_examples = cleanup(valid_data)
#  for i in remove_examples:
   #   del valid_data[i]


#  with open("raw_train.json", "w") as f:
    #  json.dump(train_data, f, indent=4)
with open("raw_valid_ninja.json", "w") as f:
    json.dump(valid_data, f, indent=4)


