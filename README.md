# Discosense: Commonsense Reasoning with Discourse Connectives


<h4>
EMNLP 2022
</br>
Prajjwal Bhargava. Vincent Ng
</h4>
<hr>

**Paper:** [arXiv](https://arxiv.org/pdf/2210.12478.pdf)

Data can be found in `/data` directory. The directory contains the training and test set.

### Usage (requires Huggingface Datasets)
Install `datasets`:
```
$ pip3 install datasets
```
You can now use `discosense` in two lines of code
```
from datasets import load_dataset
train_dataset = load_dataset("prajjwal1/discosense", split="train")
test_dataset = load_dataset("prajjwal1/discosense", split="test")
```

Data is also stored in `/data` directory.

### Models 
Generative Models can be found here:

These models were trained as follows:
Input: [control code] Sentence 1
Output: Sentence 2 (ground truth)

|  Model name           |  Model Links
| -------------------------------------------------------------------------------------- | ----------------------------
| `ctrl_discovery_1` | [Model Link](https://huggingface.co/prajjwal1/ctrl_discovery_1)         |
| `ctrl_discovery_2` | [Model Link](https://huggingface.co/prajjwal1/ctrl_discovery_2)         |
| `ctrl_discovery_3` | [Model Link](https://huggingface.co/prajjwal1/ctrl_discovery_3)         |
| `ctrl_discovery_4` | [Model Link](https://huggingface.co/prajjwal1/ctrl_discovery_4)         |
| `ctrl_discovery_5` | [Model Link](https://huggingface.co/prajjwal1/ctrl_discovery_5)         |
| `ctrl_discovery_6` | [Model Link](https://huggingface.co/prajjwal1/ctrl_discovery_6)         |
| `ctrl_discovery_7` | [Model Link](https://huggingface.co/prajjwal1/ctrl_discovery_7)         |
| `ctrl_discovery_8` | [Model Link](https://huggingface.co/prajjwal1/ctrl_discovery_8)         |
| `ctrl_discovery_9` | [Model Link](https://huggingface.co/prajjwal1/ctrl_discovery_9)         |
| `ctrl_discovery_10` | [Model Link](https://huggingface.co/prajjwal1/ctrl_discovery_10)         |
| `ctrl_discovery_11` | [Model Link](https://huggingface.co/prajjwal1/ctrl_discovery_11)         |
| `ctrl_discovery_12` | [Model Link](https://huggingface.co/prajjwal1/ctrl_discovery_12)         |
| `ctrl_discovery_13` | [Model Link](https://huggingface.co/prajjwal1/ctrl_discovery_13)         |
| `ctrl_discovery_14` | [Model Link](https://huggingface.co/prajjwal1/ctrl_discovery_14)         |



We also provide these generative models also.

These models were trained as follows:
Input: [control code] Sentence 2
Output: Sentence 1 (ground truth)

|  Model Name           |  Model Links
| -------------------------------------------------------------------------------------- | ----------------------------
| `ctrl_discovery_flipped_1` | [Model Link](https://huggingface.co/prajjwal1/ctrl_discovery_flipped_1)         |
| `ctrl_discovery_flipped_2` | [Model Link](https://huggingface.co/prajjwal1/ctrl_discovery_flipped_2)         |
| `ctrl_discovery_flipped_3` | [Model Link](https://huggingface.co/prajjwal1/ctrl_discovery_flipped_3)         |
| `ctrl_discovery_flipped_4` | [Model Link](https://huggingface.co/prajjwal1/ctrl_discovery_flipped_4)         |
| `ctrl_discovery_flipped_5` | [Model Link](https://huggingface.co/prajjwal1/ctrl_discovery_flipped_5)         |
| `ctrl_discovery_flipped_6` | [Model Link](https://huggingface.co/prajjwal1/ctrl_discovery_flipped_6)         |


`[control code]` can be replaced by these discourse markers:

| `although` | `in other words` | `particularly`|  `rather`| <br />
| `as a result` |  `in particular` | `similarly` | <br />
| `by contrast` | `in short` | `in sum` | `specifically` |  <br />
| `because of this` |  `interestingly` | `subsequently` | `because of that` | <br />
| `but` | `instead` | `thereafter` | `thereby` | `likewise` | <br />
| `consequently` | `conversely` | `nevertheless` | `therefore` | <br />
| `for example` | `nonetheless` | `though`  | `for instance` | <br />
| `on the contrary` | `thus` | `hence` | `on the other hand` | `yet` | <br />
| `however` | `otherwise` | `in contrast` | `overall` | <br />


### Conditional Adversarial Filtering
`run_af.py` can fine-tune, run CAF, run inference. These functionalities are acheived by passing different flags.

To run Conditional or non Conditional Adversarial Filtering, 
```
export OUTPUT_DIR='../../experiments/albert_large_meh' # This is the directory where the model will be saved
export RAW_DATA='' 
# This is the input data (option need to be generated for this JSON). This just has the context, discourse marker and ending

export TRAIN_DATA='' # This file will be created once CTRL has generated the training data.
# Do not use `inference_only` flag, remove `replace_one` if you want all 3 options to be generated
# Add `replace_one` if you want one option to be generated. This is useful when you're doing CAF, because
# when CAF is being run, we only want to replace the most redundant option.

export BS=16                                    
export CONTEXT_COL='sentence1'            # Key to use for getting context
export TO_PREDICT_COL='sentence2'         # Sentence which requires to be generated by the model
export MARKER_COL='marker' 
export EPOCHS=4
export WARMUP_STEPS=4000

export CLASSIFICATION_MODEL='' # Discriminator LM, can be 'roberta-large`
export AUTOREGRESSIVE_MODEL='prajjwal1/ctrl_discovery_5'                        # Generator LM
export VALIDATION_DATA=''                  
# File path for validation data, if you have validation data, this will be used by CAF to filter out examples.

export FILE_OUTPUT_PATH=''              # File that will be saved after AF has completed, this will be created

# `replace_one` will replace only one option during AF (to generate all 3 options, remove this flag)
# `run_inference_only` will perform inference (for training, remove this flag)


python3 run_af.py --replace_one --run_inference_only --classification_model_name_or_path $CLASSIFICATION_MODEL 
                  --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA 
                  --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR
                   --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8))
                   --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH --context_col $CONTEXT_COL 
                   --to_predict_col $TO_PREDICT_COL --marker_col $MARKER_COL --fp16 --save_total_limit 1 --save_strategy epoch
                   --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS
```

### Training CTRL on your own data
This is the script used for training CTRL, you can modify it as per your own usage.
```
python3 run_clm_discovery.py --model_name_or_path ctrl  --do_eval \
        --per_device_train_batch_size 24 --per_device_eval_batch_size 42 \
        --output_dir ~/apex/experiment/ctrl_discovery_flipped_6 \
        --preprocessing_num_workers 4 --evaluation_strategy no \
        --tokenizer_name ctrl --fp16 --dataset_name discovery \
        --dataset_config_name discovery --context_col sentence2 \
        --to_predict_next_col sentence1 --save_total_limit 1 --save_steps 20000
```

### Training and evaluation on HellaSWAG
```
$ cd hellaswag
```
Then run,
```
export BS=56
export EPOCH=4
export MAX_SEQ_LENGTH=96
export WARMUP_STEPS=1200
export LR=2e-5


export MODEL_PATH='google/electra-large-discriminator'
export OUTPUT_PATH='' # path where your discriminator will be stored

python3 run_hellaswag.py \
                      --model_name_or_path $MODEL_PATH\
                      --do_train \
                      --do_eval \
                      --num_train_epochs $EPOCH \
                      --output_dir $OUTPUT_PATH  \
                      --per_device_train_batch_size $BS \
                      --per_device_eval_batch_size $((BS*4)) \
                      --max_seq_length $MAX_SEQ_LENGTH \
                      --save_strategy epoch \
                      --evaluation_strategy epoch \
                      --warmup_steps $WARMUP_STEPS \
                      --fp16 \
                      --overwrite_output_dir
```
