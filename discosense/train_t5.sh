export OUTPUT_DIR='/shared/mlrdir3/disk1/pxb190028/t5/'
export TRAIN_DATA='../data/discosense_train.json'
export VALIDATION_DATA='../data/discosense_validation.json'

export BS=4
export EPOCHS=3
export WARMUP_STEPS=4000
export PADDING_LENGTH=128

export MODEL_NAME_OR_PATH='t5-small'
python3 train_t5.py --model_name_or_path $MODEL_NAME_OR_PATH  \
                --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA \
                --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS \
                --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS  --fp16 \
                --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch \
                --warmup_steps $WARMUP_STEPS  --padding_length $PADDING_LENGTH

