# Discosense: Commonsense Reasoning with Discourse Connectives
Official code for the EMNLP 2022 [paper](https://arxiv.org/pdf/2210.12478.pdf).

This repo is a work in progress. 

Data can be found in `/data` directory. The directory contains the training and test set.

### Huggingface Datasets
You can use `discosense` in two lines of code
```
from datasets import load_dataset
train_dataset = load_dataset("prajjwal1/discosense", split="train")
test_dataset = load_dataset("prajjwal1/discosense", split="test")
```


Generative Models can be found here:

These models were trained as follows:
Input: [control code] Sentence 1
Output: Sentence 2 (ground truth)

- [ctrl_discovery_1](https://huggingface.co/prajjwal1/ctrl_discovery_1)
- [ctrl_discovery_2](https://huggingface.co/prajjwal1/ctrl_discovery_2)
- [ctrl_discovery_3](https://huggingface.co/prajjwal1/ctrl_discovery_3)
- [ctrl_discovery_4](https://huggingface.co/prajjwal1/ctrl_discovery_4)
- [ctrl_discovery_5](https://huggingface.co/prajjwal1/ctrl_discovery_5)
- [ctrl_discovery_6](https://huggingface.co/prajjwal1/ctrl_discovery_6)
- [ctrl_discovery_7](https://huggingface.co/prajjwal1/ctrl_discovery_7)
- [ctrl_discovery_8](https://huggingface.co/prajjwal1/ctrl_discovery_8)
- [ctrl_discovery_9](https://huggingface.co/prajjwal1/ctrl_discovery_9)
- [ctrl_discovery_10](https://huggingface.co/prajjwal1/ctrl_discovery_10)
- [ctrl_discovery_11](https://huggingface.co/prajjwal1/ctrl_discovery_11)
- [ctrl_discovery_12](https://huggingface.co/prajjwal1/ctrl_discovery_12)
- [ctrl_discovery_13](https://huggingface.co/prajjwal1/ctrl_discovery_13)
- [ctrl_discovery_14](https://huggingface.co/prajjwal1/ctrl_discovery_14)

We also provide these generative models also.

These models were trained as follows:
Input: [control code] Sentence 2
Output: Sentence 1 (ground truth)

- [ctrl_discovery_flipped_1](https://huggingface.co/prajjwal1/ctrl_discovery_flipped_1)
- [ctrl_discovery_flipped_2](https://huggingface.co/prajjwal1/ctrl_discovery_flipped_2)
- [ctrl_discovery_flipped_3](https://huggingface.co/prajjwal1/ctrl_discovery_flipped_3)
- [ctrl_discovery_flipped_4](https://huggingface.co/prajjwal1/ctrl_discovery_flipped_4)
- [ctrl_discovery_flipped_5](https://huggingface.co/prajjwal1/ctrl_discovery_flipped_5)
- [ctrl_discovery_flipped_6](https://huggingface.co/prajjwal1/ctrl_discovery_flipped_6)


`[control code] can be replaced by these discourse markers:

| -------------------------------------------------------------------------------------- | ----------------------------
| `although` | `in other words` | `particularly`| 
| `as a result` |  `in particular` | `rather`|
| `by contrast` | `in short` | `similarly` |
| `because of this` | `in sum` | `specifically` |
| `because of that` | `interestingly` | `subsequently` |
| `but` | `instead` | `thereafter` |
| `consequently` | `likewise` | `thereby` |
| `conversely` | `nevertheless` | `therefore` |
| `for example` | `nonetheless` | `though`  |
| `for instance` | `on the contrary` | `thus` |
| `hence` | `on the other hand` | `yet` |
| `however` | `otherwise` |
| `in contrast` | `overall` | 

