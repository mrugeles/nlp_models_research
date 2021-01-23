pip install -U mlflow
pip install -q tensorflow_addons
python train_tensorflow_lstm.py --algoritm lstm --sample_size 0.01
python train_tensorflow_lstm.py --algoritm lstm --sample_size 0.1
python train_tensorflow_lstm.py --algoritm lstm --sample_size 1

