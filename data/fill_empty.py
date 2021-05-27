import json
from tqdm import tqdm
import fire
#  from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
#  from config import token_max_length, decoding_options

#  model_id = "prajjwal1/ctrl_discovery_7"

def remove_empty(fname, new_fname):
    with open(fname, "r") as f:
        data = json.load(f)

    res = []

    for item in tqdm(data):
        for k, v in item.items():
            if len(item['option_0']) > 7 and len(item['option_1']) > 7 and len(item['option_2']) > 7:
                res.append(item)
                break

    print(len(data))
    print(len(res))

    with open(new_fname, "w") as f:
        json.dump(res, f, indent=4)


if __name__=='__main__':
    fire.Fire(remove_empty)


#  model = AutoModelForCausalLM.from_pretrained(model_id)

#  tokenizer = AutoTokenizer.from_pretrained(
    #  model_id,
    #  max_length=token_max_length,
    #  padding="max_length",
    #  add_special_tokens=True
#  )

#  nlp = pipeline("text-generation", model=model,
               #  tokenizer=tokenizer, device=0
              #  )

#  for item in tqdm(data):
    #  for k, v in item.items():
        #  if k not in ['idx', 'marker', 'context', 'ground_truth']:
            #  if len(v) < 5:
                #  print('Before')
                #  print(item)
                #  print()
                #  text = item['marker'] + ' ' + item['context']
                #  output = nlp(text, max_length=token_max_length,
                             #  return_full_text=False,
                             #  clean_up_tokenization_spaces=True,
                             #  generate_kwargs=decoding_options)[0]['generated_text'].strip()
                #  if "." in output:
                    #  output = output[:output.index(".")]
                #  if "?" in output:
                    #  output = output[:output.index("?")]
                #  item[k] = output
                #  print('After')
                #  print(item)
                #  print()
#                  #  break
