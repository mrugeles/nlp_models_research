
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from mlflow import log_metric

from nlp_utils import NLPUtils

class ModelUtils:
    models = {
        'RandomForestClassifier': RandomForestClassifier(),
        'GaussianNB': GaussianNB(),
        'SVC': SVC(),
        'ExtraTreesClassifier': ExtraTreesClassifier(class_weight='balanced'),
        'GradientBoostingClassifier': GradientBoostingClassifier(),
        'HistGradientBoostingClassifier': GradientBoostingClassifier()

    }

    models_parameters = {
        'RandomForestClassifier': {
            'clf__n_estimators': [100, 150, 200],
            'clf__class_weight': ['balanced', 'balanced_subsample']
        },
        'GaussianNB': {
            'clf__var_smoothing': [1e-9, 1e-5, 1e-3]
        },
        'SVC': {
            'clf__kernel': ['rbf', 'linear', 'poly'],
            'clf__degree': [3, 4]
        }
    }

    def __init__(self):
        self.nlpUtils = NLPUtils()

    def build_model(self, model_name, tune_model=False):
        """ Builds the pipeline and finds the best classification model with gridsearch.

        Returns
        -------

        cv: GridSearchCV
            GridSearchCV instance with the tuned model.
        """
        print("build_model")
        model = self.models[model_name]
        pipeline = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('dense_transformer', DenseTransformer()),
            ('clf', model)
        ])

        if tune_model:
            parameters = self.models_parameters[model_name]
        else:
            parameters = {}
        scorer = make_scorer(f1_score, average="micro")
        cv = GridSearchCV(pipeline, scoring=scorer, param_grid=parameters, verbose=50, n_jobs=8)

        return cv

    @staticmethod
    def evaluate_model(model, X_test, Y_test):
        """Model evaluation
        """
        print("evaluate_model")
        y_pred = model.predict(X_test)
        log_metric("f1_score", f1_score(Y_test, y_pred, average='micro'))


class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()
