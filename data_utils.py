import os
import json
import pandas as pd
from tqdm import tqdm
from nlp_utils import NLPUtils
from sklearn.model_selection import train_test_split


class DataUtils:

    def __init__(self):
        self.nlpUtils = NLPUtils()
        tqdm.pandas()

    def get_dataset_aws(self, path, sample_size):
        df = pd.read_csv(path)
        df = df.loc[df['label'] != 'neu']
        df['label'] = df['label'].apply(lambda n: 0 if n == 'neg' else 1)
        df = df.drop_duplicates()
        return df.sample(frac=sample_size)

    def pre_process_aws(self, sample_size):
        train_df = self.get_dataset_aws('datasets/datasets_aws/aws_es_train.csv', sample_size)
        test_df = self.get_dataset_aws('datasets/datasets_aws/aws_es_test.csv', sample_size)
        val_df = self.get_dataset_aws('datasets/datasets_aws/aws_es_dev.csv', sample_size)

        train_df['text'] = train_df['text'].progress_apply(self.nlpUtils.tokenize)
        test_df['text'] = test_df['text'].progress_apply(self.nlpUtils.tokenize)
        val_df['text'] = val_df['text'].progress_apply(self.nlpUtils.tokenize)

        train_df.to_csv(f'datasets/processed/aws/train_{sample_size}.csv', index=False)
        test_df.to_csv(f'datasets/processed/aws/test_{sample_size}.csv', index=False)
        val_df.to_csv(f'datasets/processed/aws/val_{sample_size}.csv', index=False)

    def pre_process_airline(self):
        df = pd.read_csv('datasets/airline_tweets.csv')
        df['text'] = df['text'].progress_apply(self.nlpUtils.tokenize)

        df.to_csv(f'datasets/processed/spain_airline_tweets/airline_tweets.csv', index=False)

    def get_json_file(self, path):
        with open(path) as json_file:
            return json.load(json_file)

    def json_to_df(self, path):
        json_tweets = self.get_json_file(path)
        return pd.DataFrame.from_dict(json_tweets['data'])

    def pre_process_col_tweets(self):
        path_files = 'datasets/colombia_politics_dataset'
        df = pd.concat([self.json_to_df(f'{path_files}/{file}') for file in tqdm(os.listdir(path_files))])
        df.to_csv(f'datasets/col_politics.csv', index=False)



