import json
import fire
from transformers import AutoTokenizer


def get_stats(fname):
    with open(fname, "r") as f:
        data = json.load(f)

    print("Length of data: ", len(data))

    print("Getting stats for context")

    context, correct_options, incorrect_options = [], [], []

    for val in data:
        context.append(val['context'])
        all_options = [val['option_0'], val['option_1'], val['option_2'], val['option_3']]
        correct_options.append(all_options[val['label']])
        del all_options[val['label']]
        incorrect_options.append(all_options[0] + ' ' + all_options[1] + ' ' + all_options[2])

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    context_lengths, correct_options_length, incorrect_options_length = [], [], []

    for c in context:
        context_lengths.append(len(tokenizer.tokenize(c)))
    for idx, co in enumerate(correct_options):
        try:
            correct_options.append(len(tokenizer.tokenize(co)))
        except:
            print(idx, co)
    for inco in incorrect_options:
        incorrect_options.append(len(tokenizer.tokenize(inco)))

    print('Average # of tokens in context: ', sum(context_lengths)/len(context_lengths))
    print('Average # of tokens in correct options: ', sum(correct_options_lengths)/len(correct_options_length))
    print('Average # of tokens in incorrect options: ', sum(correct_options_lengths)/len(incorrect_options_length))







if __name__ == '__main__':
    fire.Fire(get_stats)
