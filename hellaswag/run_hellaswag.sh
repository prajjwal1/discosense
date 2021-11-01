export MODEL_PATH='albert-xxlarge-v2'
export BS=16
export EPOCH=6
export MAX_SEQ_LENGTH=96

CUDA_VISIBLE_DEVICES=1 python3 run_hellaswag.py \
                      --model_name_or_path $MODEL_PATH\
                      --do_eval \
                      --num_train_epochs $EPOCH \
                      --output_dir /shared/mlrdir3/disk1/pxb190028/models/seq_ft_wo_marker \
                      --per_device_train_batch_size $BS \
                      --per_device_eval_batch_size $((BS*4)) \
                      --max_seq_length $MAX_SEQ_LENGTH \
                      --do_train \
                      --evaluation_strategy epoch \
                      --warmup_steps 4000 \
                      --fp16 \
                      --overwrite_output_dir
