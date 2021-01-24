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
import tensorflow_hub as hub
import tensorflow_text as text

from official.nlp import optimization

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
    word_list = tweet_tokenizer.tokenize(text)
    word_list = [get_lemma(text) for text in word_list]
    return ' '.join(word_list)


def get_dataset(path, sample_size):
    df = pd.read_csv(path)
    df = df.loc[df['label'] != 'neu']
    df['label'] = df['label'].apply(lambda n: 0 if n == 'neg' else 1)
    df = df.drop_duplicates()
    return df.sample(frac=sample_size)


def sequence_datasets(train_df, val_df, test_df):
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


def train_lstm_model(train_df, val_df, test_df, sample_size):
    EPOCHS = 50
    embedding_dim = 64

    train_padded, val_padded, test_padded, max_length = sequence_datasets(train_df, val_df, test_df)
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
        validation_data=(val_padded, val_df['label']),
        class_weight=class_weight,
        callbacks=[tensorboard_callback])

    v_eval = np.vectorize(eval_prediction)

    predictions = model.predict(test_padded).flatten()
    predictions = v_eval(predictions)
    mlflow.log_metric("f1_score", f1_score(test_df['label'].values, predictions, average='micro'))

    return model


def train_cnn_model(train_df, val_df, test_df, sample_size):
    embedding_dim = 64

    train_padded, val_padded, test_padded, max_length = sequence_datasets(train_df, val_df, test_df)
    class_weight = get_class_weigth(train_df)

    EPOCHS = 30

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
        validation_data=(val_padded, val_df['label']),
        class_weight=class_weight,
        callbacks=[tensorboard_callback])

    v_eval = np.vectorize(eval_prediction)

    predictions = model.predict(test_padded).flatten()
    predictions = v_eval(predictions)
    mlflow.log_metric("f1_score", f1_score(test_df['label'].values, predictions, average='micro'))

    return model


def write_document(row, folder):
    f = open(f"{folder}/{row['index']}.txt", "w")
    f.write(row['text'])
    f.close()


def sequence_bert_datasets():
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    batch_size = 32
    seed = 42

    raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
        'datasets/bert/train',
        batch_size=batch_size,
        seed=seed)

    train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

    val_ds = tf.keras.preprocessing.text_dataset_from_directory(
        'datasets/bert/validation',
        batch_size=batch_size,
        seed=seed)

    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    test_ds = tf.keras.preprocessing.text_dataset_from_directory(
        'datasets/bert/test',
        batch_size=batch_size)

    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds


def build_bert_model(tfhub_handle_encoder, tfhub_handle_preprocess):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation="sigmoid", name='classifier')(net)
    return tf.keras.Model(text_input, net)


def train_bert_model(train_df, test_df, sample_size):
    EPOCHS = 5
    train_ds, val_ds, test_ds = sequence_bert_datasets()
    tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1'
    tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2'

    classifier_model = build_bert_model(tfhub_handle_encoder, tfhub_handle_preprocess)

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = tfa.metrics.F1Score(num_classes=2, average="micro", threshold=0.5)

    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    num_train_steps = steps_per_epoch * EPOCHS
    num_warmup_steps = int(0.1 * num_train_steps)

    init_lr = 1e-4
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')
    log_dir = f"logs/fit/bert_sample_size_{sample_size}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    classifier_model.compile(optimizer=optimizer,
                             loss=loss,
                             metrics=metrics)

    neg, pos = np.bincount(train_df['label'])
    total = neg + pos
    weight_for_0 = (1 / neg) * (total) / 2.0
    weight_for_1 = (1 / pos) * (total) / 2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}
    classifier_model.fit(x=train_ds,
                         validation_data=val_ds,
                         class_weight=class_weight,
                         batch_size=20,
                         epochs=EPOCHS,
                         callbacks=[tensorboard_callback])
    predictions = tf.sigmoid(classifier_model.predict(test_ds))
    predictions = [eval_prediction(p) for p in predictions.numpy().flatten()]

    mlflow.log_metric("f1_score", f1_score(test_df['label'].values, predictions, average='micro'))
    saved_model_path = f'aws_reviews_bert_{sample_size}'
    classifier_model.save(saved_model_path, include_optimizer=True)


def eval_prediction(n):
    return 1 if n > 0.5 else 0


def build_model(algorithm, sample_size):
    with mlflow.start_run():

        train_df = get_dataset('datasets/datasets_aws/aws_es_train.csv', sample_size)
        val_df = get_dataset('datasets/datasets_aws/aws_es_dev.csv', sample_size)
        test_df = get_dataset('datasets/datasets_aws/aws_es_test.csv', sample_size)

        train_df['text'] = train_df['text'].progress_apply(tokenize)
        val_df['text'] = val_df['text'].progress_apply(tokenize)
        test_df['text'] = test_df['text'].progress_apply(tokenize)

        mlflow.log_param("algorithm", algorithm)
        mlflow.log_param("sample_size", sample_size)
        if algorithm == 'lstm':
            train_lstm_model(train_df, val_df, test_df, sample_size)
        elif algorithm == 'cnn':
            train_cnn_model(train_df, val_df, test_df, sample_size)
        elif algorithm == 'bert':
            train_df.reset_index(inplace=True)
            val_df.reset_index(inplace=True)
            test_df.reset_index(inplace=True)

            train_df.loc[train_df['label'] == 0].apply(lambda row: write_document(row, 'datasets/bert/train/0'), axis=1)
            train_df.loc[train_df['label'] == 1].apply(lambda row: write_document(row, 'datasets/bert/train/1'), axis=1)

            val_df.loc[val_df['label'] == 0].apply(lambda row: write_document(row, 'datasets/bert/validation/0'), axis=1)
            val_df.loc[val_df['label'] == 1].apply(lambda row: write_document(row, 'datasets/bert/validation/1'), axis=1)

            test_df.loc[test_df['label'] == 0].apply(lambda row: write_document(row, 'datasets/bert/test/0'), axis=1)
            test_df.loc[test_df['label'] == 1].apply(lambda row: write_document(row, 'datasets/bert/test/1'), axis=1)

            train_bert_model(train_df, test_df, sample_size)


@click.command()
@click.option('--algorithm', help='ML Algorithm')
@click.option('--sample_size', help='Sample size for dataset', type=float)
def run_experiment(algorithm, sample_size):
    build_model(algorithm, sample_size)


if __name__ == '__main__':
    run_experiment()
