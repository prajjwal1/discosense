export MODEL_PATH='/shared/mlrdir3/disk1/pxb190028/models/hellaswag/electra-large-discriminator'
export OUTPUT_PATH='/shared/mlrdir3/disk1/pxb190028/models/hellaswag/electra-large-discriminator'
export BS=64
export EPOCH=6
export MAX_SEQ_LENGTH=96

CUDA_VISIBLE_DEVICES=1 python3 run_hellaswag.py \
                      --model_name_or_path $MODEL_PATH\
                      --do_predict \
                      --num_train_epochs $EPOCH \
                      --output_dir $OUTPUT_PATH  \
                      --per_device_train_batch_size $BS \
                      --per_device_eval_batch_size $((BS*4)) \
                      --max_seq_length $MAX_SEQ_LENGTH \
                      --save_strategy epoch \
                      --evaluation_strategy epoch \
                      --warmup_steps 4000 \
                      --fp16 \
                      --overwrite_output_dir
