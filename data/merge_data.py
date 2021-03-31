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

    #  for d in tqdm(data, desc='Fixing inputs'):
        #  for k, v in d.items():
            #  d[k] = v.capitalize()
            #  if v[0] == '.':
             #     d[k] = v.replace('.', '')

    if not os.path.isfile(fname):
        with open(fname, 'w') as f:
            json.dump(data, f, indent=4)
        print("File written successfully")
    else:
        print("File already exists")


if __name__ == '__main__':
    fire.Fire(merge_files)
