python classification_trainer.py --model_name="google-bert/bert-base-multilingual-cased" --dataset_name="liar" --experiment_name="mBERT_liar" --max_epochs=30 --learning_rate=2e-5;
python classification_trainer.py --model_name='xlm-roberta-base' --dataset_name="liar" --experiment_name="xlmRoBERTa_liar" --max_epochs=30 --learning_rate=2e-5;

