mkdir datasets/bert
mkdir datasets/bert/train
mkdir datasets/bert/train/0
mkdir datasets/bert/train/1
mkdir datasets/bert/validation
mkdir datasets/bert/validation/0
mkdir datasets/bert/validation/1
mkdir datasets/bert/test
mkdir datasets/bert/test/0
mkdir datasets/bert/test/1

pip install -q tensorflow-text
pip install -q tf-models-official
pip install -q tensorflow_addons
pip install -U mlflow

python train_tensorflow.py --algorithm bert --sample_size 0.01
python train_tensorflow.py --algorithm bert --sample_size 0.1
python train_tensorflow.py --algorithm bert --sample_size 1
