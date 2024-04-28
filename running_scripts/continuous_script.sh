python classification_trainer.py --model_name="jcblaise/electra-tagalog-base-cased-discriminator" --dataset_name="verafiles_balanced" --experiment_name="electra_verafiles_balanced" --max_epochs=60 --learning_rate=2e-5;
python classification_trainer.py --model_name="jcblaise/roberta-tagalog-base" --dataset_name="verafiles_balanced" --experiment_name="robertatl_verafiles_balanced" --max_epochs=60 --learning_rate=5e-6;
python classification_trainer.py --model_name="google-bert/bert-base-multilingual-cased" --dataset_name="liar" --experiment_name="mBERT_liar" --max_epochs=30 --learning_rate=2e-5;
python classification_trainer.py --model_name="xlm-roberta-base" --dataset_name="liar" --experiment_name="xlm_liar" --max_epochs=30 --learning_rate=2e-5;
python classification_trainer.py --model_name="google-bert/bert-base-multilingual-cased" --dataset_name="verafiles_unbalanced" --experiment_name="mBERT_verafiles_unbalanced" --max_epochs=30 --learning_rate=2e-5;
python classification_trainer.py --model_name='xlm-roberta-base' --dataset_name="verafiles_unbalanced" --experiment_name="xlm_verafiles_unbalanced" --max_epochs=30 --learning_rate=2e-5;
python classification_trainer.py --model_name="jcblaise/electra-tagalog-base-cased-discriminator" --dataset_name="verafiles_unbalanced" --experiment_name="electra_verafiles_unbalanced" --max_epochs=60 --learning_rate=2e-5;
