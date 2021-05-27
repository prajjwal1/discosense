import json
import fire


def rectify_idx(fname):
    with open(fname, "r") as f:
        data = json.load(f)
    for idx in range(len(data)):
        data[idx]['idx'] = idx

    with open(fname, "w") as f:
        json.dump(data, f, indent=4)
    print("Index Rectified successsfully")


if __name__ == '__main__':
    fire.Fire(rectify_idx)
