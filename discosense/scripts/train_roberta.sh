export CLASSIFICATION_MODEL='/shared/mlrdir2/disk1/pxb190028/albert-xxlarge'
export VALIDATION_DATA='../data/gen_valid_old.json'

export OUTPUT_DIR='/shared/mlrdir2/disk1/pxb190028/albert-xxlarge'
export FILE_OUTPUT_PATH='../data/gen_valid_M7_af1.json'
export AUTOREGRESSIVE_MODEL='prajjwal1/ctrl_discovery_7'
export RAW_DATA='../data/raw_valid.json'
export TRAIN_DATA='../data/gen_train.json'
export BS=16
export CONTEXT_COL='sentence1'
export TO_PREDICT_COL='sentence2'
export MARKER_COL='marker'
export EPOCHS=4
export WARMUP_STEPS=8000

# replace_one, run_inference_only --no_af
python3 run_af.py --replace_one --no_af --run_inference_only --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA  --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH --context_col $CONTEXT_COL  --to_predict_col $TO_PREDICT_COL --marker_col $MARKER_COL --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS
# CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus=2 ./run_af.py --replace_one --run_inference_only --no_af --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA  --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH --context_col $CONTEXT_COL  --to_predict_col $TO_PREDICT_COL --marker_col $MARKER_COL --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS --deepspeed ds_config.json
