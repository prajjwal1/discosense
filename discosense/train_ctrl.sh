python3 run_clm_discovery.py --model_name_or_path prajjwal1/ctrl_discovery_1  --do_eval \
        --per_device_train_batch_size 24 --per_device_eval_batch_size 42 \
        --output_dir ~/apex/experiment/ctrl_discovery_flipped_6 \
        --preprocessing_num_workers 4 --evaluation_strategy no \
        --tokenizer_name ctrl --fp16 --dataset_name discovery \
        --dataset_config_name discovery --context_col sentence2 \
        --to_predict_next_col sentence1 --save_total_limit 1 --save_steps 20000
