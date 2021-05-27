export CONTEXT_COL=sentence2
export TO_PREDICT_COL=sentence1

CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus=2 ./run_clm_discovery.py --model_name_or_path ctrl --do_train --do_eval \
        --deepspeed ds_config.json --dataset_name discovery --dataset_config_name discovery --per_device_train_batch_size 12 \
        --per_device_eval_batch_size 10 --output_dir ~/experiments/ctrl_discovery_flipped_6 \
        --preprocessing_num_workers 4 --evaluation_strategy no --tokenizer_name ctrl --gradient_accumulation_steps 12 \
        --context_col $CONTEXT_COL --to_predict_next_col $TO_PREDICT_COL
