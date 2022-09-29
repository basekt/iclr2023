# iclr2023

# data preprocess
python data_preprocess.py --dataset_name="assist2015" --mode="concept"

# train
python wandb_simpleKT_train.py --use_wandb=0 --dataset_name=assist2015

# predict
python wandb_predict.py --use_wandb=0 --save_dir=path/to/your/model
