import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.preprocessing import MultiLabelBinarizer

class MultiLabelTransformer(BaseEstimator, TransformerMixin):
    """Transformer for multi-label categorical features."""

    def __init__(self):
        self.mlb = MultiLabelBinarizer()

    def fit(self, X, y=None):
        X_series = pd.Series(X.squeeze())
        X_series = X_series.apply(lambda x: x.split(', ') if isinstance(x, str) else x)
        self.mlb.fit(X_series)
        return self

    def transform(self, X):
        X_series = pd.Series(X.squeeze())
        X_series = X_series.apply(lambda x: x.split(', ') if isinstance(x, str) else x)
        return self.mlb.transform(X_series)

    def get_feature_names_out(self, input_features=None):
        return self.mlb.classes_


class BoundedRegressor(BaseEstimator, RegressorMixin):
    """Wraps a regressor to clip predictions within a range."""

    def __init__(self, regressor, min_val=0, max_val=10):
        self.regressor = regressor
        self.min_val = min_val
        self.max_val = max_val

    def fit(self, X, y):
        self.regressor.fit(X, y)
        return self

    def predict(self, X):
        predictions = self.regressor.predict(X)
        return np.clip(predictions, self.min_val, self.max_val)