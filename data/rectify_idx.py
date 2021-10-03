import json
import fire


def rectify_idx(fname):
    with open(fname, "r") as f:
        data = json.load(f)
    for idx in range(len(data)):
        data[idx]['idx'] = idx

    original_data = []
    for idx, val in enumerate(data):
        options_list = [val['option_0'], val['option_1'], val['option_2'], val['option_3']]
        seen_set = set(options_list)
        if len(seen_set) == 4:
            original_data.append(val)

    print(len(data), len(original_data))

    with open(fname, "w") as f:
        json.dump(original_data, f, indent=4)
    print("Index Rectified successsfully")


if __name__ == '__main__':
    fire.Fire(rectify_idx)
