python3 run_clm_discovery.py --model_name_or_path /home/nlp/apex/experiment/ctrl/checkpoint-2000 --do_train --do_eval --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --output_dir /home/nlp/apex/experiment/ctrl  --fp16 --overwrite_output_dir --preprocessing_num_workers 4 --block_size 64 --evaluation_strategy no
# python3 generate_dataset.py
