pip install -q tensorflow-text
pip install -q tf-models-official
pip install -q tensorflow_addons
pip install -U mlflow
python train_tensorflow.py --algorithm lstm --sample_size 0.01
python train_tensorflow.py --algorithm lstm --sample_size 0.1
python train_tensorflow.py --algorithm lstm --sample_size 1

