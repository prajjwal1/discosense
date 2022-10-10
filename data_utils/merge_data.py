import os
import fire
from tqdm import tqdm
import json


def merge_files(*args, fname):
    data = []
    for fp in tqdm(args):
        with open(fp) as f:
            data.extend(json.load(f))
    print("File merged")

    for idx, d in enumerate(data):
        for k, v in d.items():
            if k not in ['idx', 'marker', 'context']:
                data[idx][k] = v.capitalize()
                if v[-1] != '.':
                    if v[-1] != '?':
                        if v[-1] != '!':
                            v += '.'
                            data[idx][k] = v

    print('Length of Dataset: ', len(data))
    if not os.path.isfile(fname):
        with open(fname, 'w') as f:
            json.dump(data, f, indent=4)
        print("File written successfully")
    else:
        print("File already exists")


if __name__ == '__main__':
    fire.Fire(merge_files)
