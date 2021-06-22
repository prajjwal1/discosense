import json

raw_data = 'raw_valid.json'
gen_data = 'gen_valid_M7_af1.json'


def split_data(fname):
    data_1, data_2, data_3 = [], [], []
    with open(fname, "r") as f:
        data = json.load(f)

    for idx in range(0, 3000):
        data_1.append(data[idx])

    for idx in range(3000, 6000):
        data_2.append(data[idx])

    for idx in range(6000, len(data)):
        data_3.append(data[idx])

    with open(fname + "_split_1.json", "w") as f:
        json.dump(data_1, f, indent=4)

    with open(fname + "_split_2.json", "w") as f:
        json.dump(data_2, f, indent=4)

    with open(fname + "_split_3.json", "w") as f:
        json.dump(data_3, f, indent=4)

#  split_data(raw_data)
split_data(gen_data)
