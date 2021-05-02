export MODEL_NAME='prajjwal1/ctrl_discovery_1'
export RAW_DATA='../data/raw_valid.json'
export PCT_RANGE='0-100'
export OUTPUT_FILE_PATH='../data/gen_valid.json'
export CONTEXT_COL='sentence1'
export TO_PREDICT_COL='sentence2'
export MARKER_COL='marker'
export OPTION_ID='0'

# Change the train/valid flag
# Check for flipping (Sentence order)

CUDA_VISIBLE_DEVICES=3 python3 generate_dataset.py --valid_pct_range $PCT_RANGE \
                      --model_name_or_path $MODEL_NAME --raw_data $RAW_DATA --option_id $OPTION_ID \
                      --output_file_path $OUTPUT_FILE_PATH --context_col $CONTEXT_COL --to_predict_col $TO_PREDICT_COL \
                      --marker_col $MARKER_COL
