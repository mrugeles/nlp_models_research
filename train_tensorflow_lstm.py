import os
import click
import shutil
import warnings
import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn
from tqdm.autonotebook import tqdm


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.metrics import f1_score

import re
import nltk
from nltk.tokenize import TweetTokenizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
tweet_tokenizer = TweetTokenizer()

warnings.filterwarnings('ignore')

vocab_size = 1000
mlflow.tensorflow.autolog()

lemmas = pd.read_csv('datasets/lemmas_df.csv', index_col=0)['lemma']
tqdm.pandas()


def get_lemma(text):
    if text in lemmas:
        return lemmas[text]
    return text


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-záéíóúÁÉÓÚÑñüÜ]", " ", text)
    tokens = tweet_tokenizer.tokenize(text)
    word_list = [get_lemma(text) for text in tokens]
    return ' '.join(word_list)


def get_dataset(path, sample_size):
    df = pd.read_csv(path)
    df = df.loc[df['label']!='neu']
    df['label'] = df['label'].apply(lambda n: 0 if n == 'neg' else 1)
    df = df.drop_duplicates()
    return df.sample(frac=sample_size)


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

    weight_for_0 = (1 / neg) * (total) / 2.0
    weight_for_1 = (1 / pos) * (total) / 2.0

    class_weight = {0: weight_for_0, 1: weight_for_1}
    return class_weight


def train_model(train_df, test_df, val_df, sample_size):

    EPOCHS = 50
    embedding_dim = 64

    train_padded, test_padded, val_padded, max_length = sequence_datasets(train_df, test_df, val_df)
    class_weight = get_class_weigth(train_df)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, kernel_regularizer=tf.keras.regularizers.l2(), return_sequences=True),
            merge_mode='concat'),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, kernel_regularizer=tf.keras.regularizers.l2()),
                                      merge_mode='concat'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    log_dir = f"logs/fit/lstm_sample_size_{sample_size}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics=[tfa.metrics.F1Score(num_classes=2, average="micro", threshold=0.5)]
    )

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

        train_df = get_dataset('datasets/datasets_aws/aws_es_train.csv', sample_size)
        test_df = get_dataset('datasets/datasets_aws/aws_es_test.csv', sample_size)
        val_df = get_dataset('datasets/datasets_aws/aws_es_dev.csv', sample_size)

        train_df['text'] = train_df['text'].progress_apply(tokenize)
        test_df['text'] = test_df['text'].progress_apply(tokenize)
        val_df['text'] = val_df['text'].progress_apply(tokenize)

        mlflow.log_param("algorithm", algorithm)
        mlflow.log_param("sample_size", sample_size)

        train_model(train_df, test_df, val_df, sample_size)


@click.command()
@click.option('--algorithm', help='ML Algorithm')
@click.option('--sample_size', help='Sample size for dataset')
def run_experiment(algorithm, sample_size):
    build_model(algorithm, sample_size)


if __name__ == '__main__':
    run_experiment()
