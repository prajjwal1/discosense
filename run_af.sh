CUDA_VISIBLE_DEVICES=1 python3 run_af.py --run_inference_only --classification_model_name_or_path /home/nlp/apex/experiment/roberta --autoregressive_model_name_or_path /home/nlp/apex/experiment/ctrl_2/  --train_data_path /home/nlp/apex/commonsense-discourse/data/ctrl_main.json --validation_data_path /home/nlp/apex/commonsense-discourse/data/ctrl_valid_1.json --output_dir /home/nlp/apex/experiment/roberta --per_device_train_batch_size 256 --per_device_eval_batch_size 256 --num_train_epochs 3 --file_output_path /home/nlp/apex/commonsense-discourse/data/ctrl_valid_2.json
# --freeze_encoder
