import fire
from tqdm import tqdm
import json

def merge_files(f1, f2, f3, fname):
    with open(f1) as f:
        data = json.load(f)
    with open(f2) as f:
        data1 = json.load(f)
    with open(f3) as f:
        data2 = json.load(f)
    data.extend(data1)
    data.extend(data2)
    print("File merged")

    for el in tqdm(data):
        for k, v in el.items():
            if v and v[-1] == '.':
                el[k] = v[:-1]

    with open(fname, 'w') as f:
        json.dump(data, f, indent=4)
    print("File written successfully")

if __name__ == '__main__':
  fire.Fire(merge_files)
