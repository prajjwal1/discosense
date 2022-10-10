"""
This script is to be used if some options are missing from one Json file and the other Json file has some options which the first one doesnt have
"""

import fire
import json
from tqdm import tqdm

def merge_files(*args, fname):
    data = []
    assert len(args) == 3

    with open(args[0], "r") as f:
        data_0 = json.load(f)

    with open(args[1], "r") as f:
        data_1 = json.load(f)

    with open(args[2], "r") as f:
        data_2 = json.load(f)

    data = []
    for idx in range(len(data_0)):
        example = data_0[idx]
        example.update(data_1[idx])
        example.update(data_2[idx])
        data.append(example)

    with open(fname, "w") as f:
        json.dump(data, f, indent=4)
    print(fname, " written successfully")

if __name__ == '__main__':
    fire.Fire(merge_files)
