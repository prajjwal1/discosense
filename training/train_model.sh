CUDA_VISIBLE_DEVICES=0 python3 run_clm_discovery.py --model_name_or_path ctrl --do_train --do_eval --per_device_train_batch_size 16 --per_device_eval_batch_size 8 --output_dir /home/nlp/apex/experiment/ctrl_4  --fp16 --preprocessing_num_workers 4 --block_size 64 --evaluation_strategy no --tokenizer_name ctrl
