import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
tweet_tokenizer = TweetTokenizer()


class NLPUtils:

    def __init__(self):
        self.lemmas = pd.read_csv('datasets/lemmas_df.csv', index_col=0)['lemma']

    def get_lemma(self, text):
        if text in self.lemmas:
            return self.lemmas[text]
        return text

    def tokenize(self, text, stop_words=False):
        text = text.lower()
        text = re.sub(r"[^a-záéíóúÁÉÓÚÑñüÜ]", " ", text)
        word_list = tweet_tokenizer.tokenize(text)
        if stop_words is False:
            word_list = [w for w in word_list if w not in stopwords.words("spanish")]
        word_list = [self.get_lemma(text) for text in word_list]
        return ' '.join(word_list)

