export MODEL_NAME='prajjwal1/ctrl_discovery_3'
export RAW_DATA='../data/raw_train.json'
export PCT_RANGE='0-50'
export OUTPUT_FILE_PATH='../data/gen_train_0_50_op_2.json'
export CONTEXT_COL='sentence1'
export TO_PREDICT_COL='sentence2'
export MARKER_COL='marker'
export OPTION_ID='2'
# Change the train/valid flag
# Check for flipping (Sentence order)

CUDA_VISIBLE_DEVICES=1 python3 generate_dataset.py --train_pct_range $PCT_RANGE --resume_gen_file ../data/gen_train_0_50_op_1.json \
                      --model_name_or_path $MODEL_NAME --raw_data $RAW_DATA --option_id $OPTION_ID \
                      --output_file_path $OUTPUT_FILE_PATH --context_col $CONTEXT_COL --to_predict_col $TO_PREDICT_COL \
                      --marker_col $MARKER_COL

