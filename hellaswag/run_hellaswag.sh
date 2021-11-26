export BS=56
export EPOCH=4
export MAX_SEQ_LENGTH=96
export WARMUP_STEPS=1200
export LR=2e-5
# best performing funnel ws:1500


export MODEL_PATH='/shared/mlrdir3/disk1/pxb190028/models/discosense/funnel-transformer-xlarge_ws_1500/'
export OUTPUT_PATH='/shared/mlrdir3/disk1/pxb190028/models/seq_ft/funnel-transformer-xlarge_ws_1500/'
python3 run_hellaswag.py \
                      --model_name_or_path $MODEL_PATH\
                      --do_train \
                      --do_eval \
                      --num_train_epochs $EPOCH \
                      --output_dir $OUTPUT_PATH  \
                      --per_device_train_batch_size $BS \
                      --per_device_eval_batch_size $((BS*4)) \
                      --max_seq_length $MAX_SEQ_LENGTH \
                      --save_strategy epoch \
                      --evaluation_strategy epoch \
                      --warmup_steps $WARMUP_STEPS \
                      --fp16 \
                      --overwrite_output_dir


