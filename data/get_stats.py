import json
import fire
from collections import defaultdict
from transformers import AutoTokenizer

def get_stats(fname):
    with open(fname, "r") as f:
        data = json.load(f)

    print("Length of data: ", len(data))

    print("Getting stats for context")

    context, correct_options, incorrect_options = [], [], []
    marker = defaultdict(int)

    for val in data:
        context.append(val['context'].lower())
        all_options = [val['option_0'].lower(), val['option_1'].lower(), val['option_2'].lower(), val['option_3'].lower()]
        correct_options.append(all_options[val['label']])
        incorrect_options.append(all_options[0] + ' ' + all_options[1] + ' ' + all_options[2])
        marker[val['marker']] += 1

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    context_lengths, correct_options_length, incorrect_options_length = [], [], []
    all_options_length = []

    for c in context:
        context_lengths.append(len(tokenizer.tokenize(c)))

    for idx in range(len(data)):
        correct_options_length.append(len(tokenizer.tokenize(correct_options[idx]))-2)
        incorrect_options_length.append(len(tokenizer.tokenize(incorrect_options[idx]))-2)
        all_options_length.append(len(tokenizer.tokenize(correct_options[idx] + ' ' + incorrect_options[idx]))-2)

    print('Average # of tokens in context: ', sum(context_lengths)/len(context_lengths))
    print('Average # of tokens in correct options: ', sum(correct_options_length)/len(correct_options_length))
    print('Average # of tokens in incorrect options: ', sum(incorrect_options_length)/(len(incorrect_options_length)*3))
    print('Average # of tokens in all options: ', sum(all_options_length)/(len(all_options_length)*4))

    all_context_string, all_correct_options_string, all_incorrect_options_string = '', '', ''

    for idx in range(len(data)):
        all_context_string += context[idx]
        all_correct_options_string += ' ' + correct_options[idx]
        all_incorrect_options_string += ' ' + incorrect_options[idx]

    all_options_string = all_correct_options_string + all_incorrect_options_string

    print('# of unique tokens in All string: ', len(set(all_options_string.split())))
    print('# of unique tokens in Context string: ', len(set(all_context_string.split())))
    print('# of unique tokens in Correct string: ', len(set(all_correct_options_string.split())))
    print('# of unique tokens in Incorrect string: ', len(set(all_incorrect_options_string.split())))

    with open("marker_stats.json", "w") as f:
        json.dump(marker, f)


if __name__ == '__main__':
    fire.Fire(get_stats)
