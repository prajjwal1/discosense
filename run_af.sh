export CLASSIFICATION_MODEL='bert-base-uncased'
export AUTOREGRESSIVE_MODEL='prajjwal1/ctrl_discovery_2'
export RAW_DATA='data/raw_train.json'
export TRAIN_DATA='data/gen_train.json'
# export VALIDATION_DATA='data/gen_valid.json'
export VALIDATION_DATA='data/gen_valid.json'
export OUTPUT_DIR='../experiments/bert-base'
export BS=64
export FILE_OUTPUT_PATH='data/gen_valid_1_2af.json'
export CONTEXT_COL='sentence1'
export TO_PREDICT_COL='sentence2'
export MARKER_COL='marker'
export EPOCHS=3
export WARMUP_STEPS=10000

# replace_one, run_inference_only

CUDA_VISIBLE_DEVICES=0 python3 run_af.py \
        --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS --per_device_eval_batch_size $((BS*4)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH --context_col $CONTEXT_COL --to_predict_col $TO_PREDICT_COL --marker_col $MARKER_COL --save_total_limit 1 --save_steps 1000 --evaluation_strategy epoch --warmup $WARMUP_STEPS
