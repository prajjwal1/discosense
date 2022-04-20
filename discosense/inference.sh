export MODEL_PATH='/shared/mlrdir3/disk1/pxb190028/models/electra-large-discriminator'

export DATA_PATH='/users/pxb190028/commonsense-discourse/data/discosense_validation.json'

python3 perform_inference.py  --model_name_or_path $MODEL_PATH \
                   --data_path $DATA_PATH --padding_length 96 --output_dir $MODEL_PATH
