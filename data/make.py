import os
import json
import fire

def get_stats(fname):
    with open(fname, "r") as f:
        data = json.load(f)

    for val in data:
        f_idx = val['idx']
        context = val['context'].lower()
        marker = val['marker'].lower()

        all_options = [val['option_0'].lower(), val['option_1'].lower(), val['option_2'].lower(), val['option_3'].lower()]
        ground_truth = all_options[val['label']]

        sentence = context + ' ' + marker + ' ' + ground_truth
        os.mkdir("make/" + str(f_idx))

        with open("make/" + str(f_idx) + "/" + str(f_idx) + ".txt", 'w') as f:
            f.write(sentence)



if __name__ == '__main__':
    fire.Fire(get_stats)
