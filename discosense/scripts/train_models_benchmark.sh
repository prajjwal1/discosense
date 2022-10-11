export TRAIN_DATA='../data/discosense_train.json'
export VALIDATION_DATA='../data/discosense_validation.json'
#####  Only required for AF
export FILE_OUTPUT_PATH='../data/_valid_meh.json'
export RAW_DATA='../data/raw_valid.json'
export AUTOREGRESSIVE_MODEL='prajjwal1/ctrl_discovery_1'
######################
export BS=4
export LR=2e-5
export CONTEXT_COL='sentence1'
export TO_PREDICT_COL='sentence2'
export MARKER_COL='marker'
export EPOCHS=4
export WARMUP_STEPS=4000
export PADDING_LENGTH=96

# 8, warmup: 8000 albert-xxlarge-v2
# 16, warmup: 4000  electra-large

# replace_one, run_inference_only --no_af
# echo "ROBERTA BASE"

# for run in 1 2 3 4 5; do
    # echo "############################### Iteration $((run))  ####################################"
    # python3 run_af.py --replace_one --no_af --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA  --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH --context_col $CONTEXT_COL  --to_predict_col $TO_PREDICT_COL --marker_col $MARKER_COL --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS --random_seed
# # done
# echo "BERT-BASE"
# export CLASSIFICATION_MODEL='bert-base-uncased'
# export OUTPUT_DIR='/shared/mlrdir3/disk1/pxb190028/models/discosense/bert-base-uncased'
# # for run in 1 2 3 4 5; do
    # # echo "############################### Iteration $((run))  ####################################"
# python3 run_af.py --replace_one --no_af --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA  --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH --context_col $CONTEXT_COL  --to_predict_col $TO_PREDICT_COL --marker_col $MARKER_COL --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS --random_seed --padding_length 96 --learning_rate $LR
# # done


# echo "ROBERTA-BASE"
# export CLASSIFICATION_MODEL='roberta-base'
# export OUTPUT_DIR='/shared/mlrdir3/disk1/pxb190028/models/discosense/roberta-base'
# # for run in 1 2 3 4 5; do
    # # echo "############################### Iteration $((run))  ####################################"
# python3 run_af.py --replace_one --no_af --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA  --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH --context_col $CONTEXT_COL  --to_predict_col $TO_PREDICT_COL --marker_col $MARKER_COL --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS --random_seed --padding_length 96 --learning_rate $LR
# # done

# # echo "GPT2"
# # export CLASSIFICATION_MODEL='gpt2'
# # export OUTPUT_DIR='/shared/mlrdir3/disk1/pxb190028/models/gpt2-xl'
# # export BS=16
# # export EPOCH=5
# # for run in 1 2 3 4 5; do
    # # echo "############################### Iteration $((run))  ####################################"
    # # python3 run_af.py --replace_one --no_af --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA  --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH --context_col $CONTEXT_COL  --to_predict_col $TO_PREDICT_COL --marker_col $MARKER_COL --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS --random_seed --padding_length 96
# # done


# # echo "ROBERTA LARGE"
# export CLASSIFICATION_MODEL='roberta-large'
# export OUTPUT_DIR='/shared/mlrdir3/disk1/pxb190028/models/discosense/roberta-large'
# # # for run in 1 2 3 4 5; do
    # # # echo "############################### Iteration $((run))  ####################################"
# python3 run_af.py --replace_one --no_af --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA  --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH --context_col $CONTEXT_COL  --to_predict_col $TO_PREDICT_COL --marker_col $MARKER_COL --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS --random_seed --padding_length 96
# # done

# echo "XLNET Large"
# export CLASSIFICATION_MODEL='xlnet-large-cased'
# export BS=32
# export OUTPUT_DIR='/shared/mlrdir3/disk1/pxb190028/models/discosense/xlnet-large'
# # # for run in 1 2 3 4 5; do
    # # # echo "############################### Iteration $((run))  ####################################"
# python3 run_af.py --replace_one --no_af --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA  --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH --context_col $CONTEXT_COL  --to_predict_col $TO_PREDICT_COL --marker_col $MARKER_COL --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS --random_seed --padding_length $PADDING_LENGTH --dataloader_drop_last
# # # done
# export OUTPUT_DIR='/shared/mlrdir3/disk1/pxb190028/models/discosense/xlnet-large_ws_200'
# export WARMUP_STEPS=200
# python3 run_af.py --replace_one --no_af --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA  --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH --context_col $CONTEXT_COL  --to_predict_col $TO_PREDICT_COL --marker_col $MARKER_COL --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS --random_seed --padding_length $PADDING_LENGTH --dataloader_drop_last

# export OUTPUT_DIR='/shared/mlrdir3/disk1/pxb190028/models/discosense/xlnet-large_ws_600'
# export WARMUP_STEPS=600
# python3 run_af.py --replace_one --no_af --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA  --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH --context_col $CONTEXT_COL  --to_predict_col $TO_PREDICT_COL --marker_col $MARKER_COL --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS --random_seed --padding_length $PADDING_LENGTH --dataloader_drop_last

# # echo "ALBERT XXLARGE"
# # export CLASSIFICATION_MODEL='albert-xxlarge-v2'
# # export OUTPUT_DIR='/shared/mlrdir3/disk1/pxb190028/'$CLASSIFICATION_MODEL
# # export EPOCH=3
# # export BS=16
# # for run in 1 2 3 4 5; do
    # # echo "############################### Iteration $((run))  ####################################"
    # # python3 run_af.py --replace_one --no_af --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA  --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH --context_col $CONTEXT_COL  --to_predict_col $TO_PREDICT_COL --marker_col $MARKER_COL --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS --random_seed --padding_length $PADDING_LENGTH
# # done

# echo "BERT LARGE"
# export CLASSIFICATION_MODEL='bert-large-uncased'
# export OUTPUT_DIR='/shared/mlrdir3/disk1/pxb190028/models/discosense/bert-large'
# # for run in 1 2 3 4 5; do
    # # echo "############################### Iteration $((run))  ####################################"
# python3 run_af.py --replace_one --no_af --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA  --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH --context_col $CONTEXT_COL  --to_predict_col $TO_PREDICT_COL --marker_col $MARKER_COL --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS --random_seed --padding_length $PADDING_LENGTH
# # done

# # echo "ELECTRA"
# # export CLASSIFICATION_MODEL='google/electra-large-discriminator'
# # export OUTPUT_DIR='/shared/mlrdir3/disk1/pxb190028/models/electra-large-discriminator'
# # for run in 1 2 3 4 5; do
    # # echo "############################### Iteration $((run))  ####################################"
    # # python3 run_af.py --replace_one --no_af --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA  --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH --context_col $CONTEXT_COL  --to_predict_col $TO_PREDICT_COL --marker_col $MARKER_COL --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS --random_seed --padding_length $PADDING_LENGTH
# # done


# # echo "longformer"
# export CLASSIFICATION_MODEL='allenai/longformer-base-4096'
# export OUTPUT_DIR='/shared/mlrdir3/disk1/pxb190028/models/discosense/longformer-base'
# export BS=2
# # for run in 1 2 3 4 5; do
    # # echo "############################### Iteration $((run))  ####################################"
# python3 run_af.py --replace_one --no_af --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA  --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH --context_col $CONTEXT_COL  --to_predict_col $TO_PREDICT_COL --marker_col $MARKER_COL --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS --random_seed --padding_length 512
# done

# echo "funnel-transformer"
export CLASSIFICATION_MODEL='funnel-transformer/xlarge'
# export OUTPUT_DIR='/shared/mlrdir3/disk1/pxb190028/models/discosense/funnel-transformer-xlarge_ws_200'
# export WARMUP_STEPS=200
# # for run in 1 2 3 4 5; do
    # # echo "############################### Iteration $((run))  ####################################"
# python3 run_af.py --replace_one --no_af --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA  --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH  --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS --random_seed --padding_length 512
# # done
# export OUTPUT_DIR='/shared/mlrdir3/disk1/pxb190028/models/discosense/funnel-transformer-xlarge_ws_600'
# export LR=1e-5
# export WARMUP_STEPS=600
# python3 run_af.py --replace_one --no_af --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA  --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH  --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS --random_seed --padding_length 512

# export OUTPUT_DIR='/shared/mlrdir3/disk1/pxb190028/models/discosense/funnel-transformer-xlarge_ws_1500'
# export LR=3e-5
# export WARMUP_STEPS=600
# python3 run_af.py --replace_one --no_af --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA  --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH  --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS --random_seed --padding_length 512

# export OUTPUT_DIR='/shared/mlrdir3/disk1/pxb190028/models/discosense/funnel-transformer-xlarge_ws'
# export LR=2e-5
# # export WARMUP_STEPS=600
# python3 run_af.py --replace_one --no_af --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA  --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH  --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS --random_seed --padding_length 512

export OUTPUT_DIR='/shared/mlrdir3/disk1/pxb190028/models/discosense/funnel-transformer-xlarge_ws_600_1e6'
export LR=1e-6
export WARMUP_STEPS=600
python3 run_af.py --replace_one --no_af --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA  --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH  --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS --random_seed --padding_length 512

export OUTPUT_DIR='/shared/mlrdir3/disk1/pxb190028/models/discosense/funnel-transformer-xlarge_ws_600_1e4'
export LR=1e-4
# export WARMUP_STEPS=600
python3 run_af.py --replace_one --no_af --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA  --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH  --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS --random_seed --padding_length 512



