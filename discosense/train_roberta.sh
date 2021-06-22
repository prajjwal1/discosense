export CLASSIFICATION_MODEL='allenai/longformer-large-4096'
export AUTOREGRESSIVE_MODEL='prajjwal1/ctrl_discovery_7'
# export VALIDATION_DATA='../data/gen_valid_split1_M7_M12_M4_M12_M1_M13_fM1_M1_M2.json'
export VALIDATION_DATA='../data/gen_valid.json'
export OUTPUT_DIR='../../experiments/roberta_large_10'
export FILE_OUTPUT_PATH='../data/gen_valid_M7_af1.json'

export RAW_DATA='../data/raw_valid.json'
export TRAIN_DATA='../data/gen_train.json'
export BS=2
export CONTEXT_COL='sentence1'
export TO_PREDICT_COL='sentence2'
export MARKER_COL='marker'
export EPOCHS=4
export WARMUP_STEPS=4000

# replace_one, run_inference_only --no_af
export OUTPUT_DIR='/shared/mlrdir2/disk1/pxb190028/longformer_large_4096'
# python3 run_af.py --replace_one --no_af --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA  --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH --context_col $CONTEXT_COL  --to_predict_col $TO_PREDICT_COL --marker_col $MARKER_COL --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS
CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus=2 ./run_af.py --replace_one --no_af --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA  --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH --context_col $CONTEXT_COL  --to_predict_col $TO_PREDICT_COL --marker_col $MARKER_COL --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS --deepspeed ds_config.json
