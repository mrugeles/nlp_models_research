
import click
import warnings
import pandas as pd

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.metrics import f1_score

from data_utils import DataUtils
from model_utils import ModelUtils
from nlp_utils import NLPUtils

warnings.filterwarnings('ignore')

modelUtils = ModelUtils()
dataUtils = DataUtils()
nlpUtils = NLPUtils()

datasets_path = ''


def build_model(algorithm, sample_size):
    with mlflow.start_run():
        train_df = pd.read_csv(f'datasets/processed/aws/train_{sample_size}.csv')
        test_df = pd.read_csv(f'datasets/processed/aws/test_{sample_size}.csv')

        mlflow.log_param("algorithm", algorithm)
        mlflow.log_param("use_tokenizer", False)
        mlflow.log_param("remove_stop_words", False)
        mlflow.log_param("sample_size", sample_size)

        model = modelUtils.build_model(algorithm)
        model.fit(train_df['text'], train_df['label'])

        y_pred = model.predict(test_df['text'])

        signature = infer_signature(train_df['text'], y_pred)
        mlflow.sklearn.log_model(model, algorithm, signature=signature)

        mlflow.log_metric("f1_score", f1_score(test_df['label'], y_pred, average='micro'))


@click.command()
@click.option('--algorithm', help='ML Algorithm')
@click.option('--sample_size', help='Sample size for dataset', type=float)
def run_experiment(algorithm, sample_size):
    build_model(algorithm, sample_size)


if __name__ == '__main__':
    run_experiment()


