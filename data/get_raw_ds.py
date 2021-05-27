import os
import json
import fire

def get_raw_dataset(gen_ds, fname):
    data = []
    with open(gen_ds, "r") as f:
        gen_data = json.load(f)
    for item in gen_data:
        data.append(
            {'idx': item['idx'],
             'marker': item['marker'],
             'sentence1': item['context'],
             'sentence2': item['ground_truth']
            }
        )

    if not os.path.isfile(fname):
        with open(fname, "w") as f:
            json.dump(data, f, indent=4)
        print("File ", fname, "created successfully")
    else:
        print("File exists")

if __name__ == '__main__':
    fire.Fire(get_raw_dataset)
