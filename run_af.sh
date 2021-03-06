python3 run_af.py --no_af --classification_model_name_or_path roberta-large --autoregressive_model_name_or_path prajjwal1/ctrl_discovery_1  --train_data_path data/ctrl_main.json --validation_data_path data/ctrl_valid.json --output_dir ~/experiments/roberta --per_device_train_batch_size 256 --per_device_eval_batch_size 256 --num_train_epochs 3 --file_output_path data/ctrl_valid_1_2afall.json --context_col sentence1 --to_predict_col sentence2 --num_train_epochs 6 --replace_one False 
# --freeze_encoder
