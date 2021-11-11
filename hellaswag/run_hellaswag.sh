export MODEL_PATH='bert-base-uncased'
export OUTPUT_PATH='/shared/mlrdir3/disk1/pxb190028/models/hellaswag/bert-base-uncased'
export BS=32
export EPOCH=4
export MAX_SEQ_LENGTH=96

# python3 run_hellaswag.py \
                      # --model_name_or_path $MODEL_PATH\
                      # --do_train \
                      # --do_eval \
                      # --num_train_epochs $EPOCH \
                      # --output_dir $OUTPUT_PATH  \
                      # --per_device_train_batch_size $BS \
                      # --per_device_eval_batch_size $((BS*4)) \
                      # --max_seq_length $MAX_SEQ_LENGTH \
                      # --save_strategy epoch \
                      # --evaluation_strategy epoch \
                      # --warmup_steps 4000 \
                      # --fp16 \
                      # --overwrite_output_dir

export MODEL_PATH='bert-large-uncased'
export OUTPUT_PATH='/shared/mlrdir3/disk1/pxb190028/models/hellaswag/bert-large-uncased'
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
                      --warmup_steps 4000 \
                      --fp16 \
                      --overwrite_output_dir

# export MODEL_PATH='roberta-base'
# export OUTPUT_PATH='/shared/mlrdir3/disk1/pxb190028/models/hellaswag/roberta-base'
# python3 run_hellaswag.py \
                      # --model_name_or_path $MODEL_PATH\
                      # --do_train \
                      # --do_eval \
                      # --num_train_epochs $EPOCH \
                      # --output_dir $OUTPUT_PATH  \
                      # --per_device_train_batch_size $BS \
                      # --per_device_eval_batch_size $((BS*4)) \
                      # --max_seq_length $MAX_SEQ_LENGTH \
                      # --save_strategy epoch \
                      # --evaluation_strategy epoch \
                      # --warmup_steps 4000 \
                      # --fp16 \
                      # --overwrite_output_dir

# export MODEL_PATH='roberta-base'
# export OUTPUT_PATH='/shared/mlrdir3/disk1/pxb190028/models/hellaswag/roberta-base'
# python3 run_hellaswag.py \
                      # --model_name_or_path $MODEL_PATH\
                      # --do_train \
                      # --do_eval \
                      # --num_train_epochs $EPOCH \
                      # --output_dir $OUTPUT_PATH  \
                      # --per_device_train_batch_size $BS \
                      # --per_device_eval_batch_size $((BS*4)) \
                      # --max_seq_length $MAX_SEQ_LENGTH \
                      # --save_strategy epoch \
                      # --evaluation_strategy epoch \
                      # --warmup_steps 4000 \
                      # --fp16 \
                      # --overwrite_output_dir

export MODEL_PATH='allenai/longformer-base-4096'
export OUTPUT_PATH='/shared/mlrdir3/disk1/pxb190028/models/hellaswag/longformer-base-4096'
export BS=4
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
                      --warmup_steps 4000 \
                      --fp16 \
                      --overwrite_output_dir

export MODEL_PATH='xlnet-large-cased'
export OUTPUT_PATH='/shared/mlrdir3/disk1/pxb190028/models/hellaswag/xlnet-large-cased'
export BS=128
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
                      --warmup_steps 4000 \
                      --fp16 \
                      --overwrite_output_dir



# export MODEL_PATH='funnel-transformer/xlarge'
# export OUTPUT_PATH='/shared/mlrdir3/disk1/pxb190028/models/hellaswag/funnel-transformer'
# export BS=16
# python3 run_hellaswag.py \
                      # --model_name_or_path $MODEL_PATH\
                      # --do_train \
                      # --do_eval \
                      # --num_train_epochs $EPOCH \
                      # --output_dir $OUTPUT_PATH  \
                      # --per_device_train_batch_size $BS \
                      # --per_device_eval_batch_size $((BS*4)) \
                      # --max_seq_length $MAX_SEQ_LENGTH \
                      # --save_strategy epoch \
                      # --evaluation_strategy epoch \
                      # --warmup_steps 4000 \
                      # --fp16 \
                      # --overwrite_output_dir



