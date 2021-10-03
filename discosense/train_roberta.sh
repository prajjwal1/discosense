export CLASSIFICATION_MODEL='roberta-large'
export OUTPUT_DIR='/shared/mlrdir1/disk1/pxb190028/roberta-large'

export TRAIN_DATA='../data/discosense_train.json'
export VALIDATION_DATA='../data/discosense_validation.json'
#####  Only required for AF
export FILE_OUTPUT_PATH='../data/_valid_meh.json'
export RAW_DATA='../data/raw_valid.json'
export AUTOREGRESSIVE_MODEL='prajjwal1/ctrl_discovery_1'
######################
export BS=16
export CONTEXT_COL='sentence1'
export TO_PREDICT_COL='sentence2'
export MARKER_COL='marker'
export EPOCHS=8
export WARMUP_STEPS=200

# 8, warmup: 8000 albert-xxlarge-v2
# 16, warmup: 4000  electra-large

# replace_one, run_inference_only --no_af
python3 run_af.py --replace_one --no_af --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA  --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH --context_col $CONTEXT_COL  --to_predict_col $TO_PREDICT_COL --marker_col $MARKER_COL --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS --random_seed
