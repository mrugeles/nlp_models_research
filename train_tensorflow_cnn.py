import os
import click
import shutil
import warnings
import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.metrics import f1_score

warnings.filterwarnings('ignore')

vocab_size = 1000
mlflow.tensorflow.autolog()

def sequence_datasets(train_df, test_df, val_df):
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    trunc_type = 'post'
    oov_tok = '<OOV>'

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok, )
    tokenizer.fit_on_texts(train_df['text'])

    sequences = tokenizer.texts_to_sequences(train_df['text'])
    len_sequences = [len(sequence) for sequence in sequences]
    max_length = max(len_sequences)
    train_padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)

    val_sequences = tokenizer.texts_to_sequences(val_df['text'])
    val_padded = pad_sequences(val_sequences, maxlen=max_length)

    test_sequences = tokenizer.texts_to_sequences(test_df['text'])
    test_padded = pad_sequences(test_sequences, maxlen=max_length)

    return train_padded, val_padded, test_padded, max_length


def get_class_weigth(train_df):
    neg, pos = np.bincount(train_df['label'])
    total = neg + pos
    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))

    weight_for_0 = (1 / neg) * (total) / 2.0
    weight_for_1 = (1 / pos) * (total) / 2.0

    class_weight = {0: weight_for_0, 1: weight_for_1}
    return class_weight


def train_model(algorithm, train_df, test_df, val_df, sample_size):

    embedding_dim = 64

    train_padded, test_padded, val_padded, max_length = sequence_datasets(train_df, test_df, val_df)
    class_weight = get_class_weigth(train_df)

    EPOCHS = 1

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.Conv1D(64, 5, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    log_dir = f"logs/fit/cnn_sample_size_{sample_size}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tfa.metrics.F1Score(num_classes=2, average="micro", threshold=0.5)])

    model.fit(
        train_padded,
        train_df['label'],
        epochs=EPOCHS,
        batch_size=2000,
        validation_data=(test_padded, test_df['label']),
        class_weight=class_weight,
        callbacks=[tensorboard_callback])

    v_eval = np.vectorize(eval_prediction)

    predictions = model.predict(test_padded).flatten()
    predictions = v_eval(predictions)
    mlflow.log_metric("f1_score", f1_score(test_df['label'].values, predictions, average='micro'))

    return model


def eval_prediction(n):
    return 1 if n > 0.5 else 0


def build_model(algorithm, sample_size):
    with mlflow.start_run():

        train_df = pd.read_csv(f'datasets/processed/aws/train_{sample_size}.csv')
        test_df = pd.read_csv(f'datasets/processed/aws/test_{sample_size}.csv')
        val_df = pd.read_csv(f'datasets/processed/aws/val_{sample_size}.csv')

        mlflow.log_param("algorithm", algorithm)
        mlflow.log_param("sample_size", sample_size)

        train_model(algorithm, train_df, test_df, val_df, sample_size)


@click.command()
@click.option('--algorithm', help='ML Algorithm')
@click.option('--sample_size', help='Sample size for dataset', type=float)
def run_experiment(algorithm, sample_size):
    build_model(algorithm, sample_size)


if __name__ == '__main__':
    run_experiment()
