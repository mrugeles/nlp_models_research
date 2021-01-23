pip install -U mlflow
pip install -q tensorflow_addons
python train_sklearn.py --algorithm RandomForestClassifier --sample_size 0.01 --tune_model false
python train_sklearn.py --algorithm GradientBoostingClassifier --sample_size 0.01 --tune_model false
python train_sklearn.py --algorithm GaussianNB --sample_size 0.01 --tune_model false
python train_sklearn.py --algorithm SVC --sample_size 0.01 --tune_model false
python train_sklearn.py --algorithm RandomForestClassifier --sample_size 0.1 --tune_model false
python train_sklearn.py --algorithm GradientBoostingClassifier --sample_size 0.1 --tune_model false
python train_sklearn.py --algorithm GaussianNB --sample_size 0.1 --tune_model false
python train_sklearn.py --algorithm SVC --sample_size 0.1 --tune_model false
python train_sklearn.py --algorithm RandomForestClassifier --sample_size 1 --tune_model false
python train_sklearn.py --algorithm GradientBoostingClassifier --sample_size 1 --tune_model false
python train_sklearn.py --algorithm GaussianNB --sample_size 1 --tune_model false
python train_sklearn.py --algorithm SVC --sample_size 1 --tune_model false
python train_tensorflow_lstm.py --algoritm lstm --sample_size 0.01
python train_tensorflow_lstm.py --algoritm lstm --sample_size 0.1
python train_tensorflow_lstm.py --algoritm lstm --sample_size 1
python train_tensorflow_cnn.py --algoritm cnn --sample_size 0.01
python train_tensorflow_cnn.py --algoritm cnn --sample_size 0.1
python train_tensorflow_cnn.py --algoritm cnn --sample_size 1
