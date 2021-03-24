export MODEL_NAME='prajjwal1/ctrl_discovery_1'
export RAW_DATA='../data/raw_train.json'
export PCT_RANGE='0-10'
export OUTPUT_FILE_PATH='../data/gen_train_0_10.json'
export CONTEXT_COL='sentence1'
export TO_PREDICT_COL='sentence2'
export MARKER_COL='marker'

# Change the train/valid flag
# Check for flipping (Sentence order)

CUDA_VISIBLE_DEVICES=0 python3 generate_dataset.py --train_pct_range $PCT_RANGE \
                      --model_name_or_path $MODEL_NAME --raw_data $RAW_DATA  --output_file_path $OUTPUT_FILE_PATH --context_col $CONTEXT_COL --to_predict_col $TO_PREDICT_COL --marker_col $MARKER_COL
