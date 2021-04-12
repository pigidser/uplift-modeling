import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ClusterTransform(BaseEstimator, TransformerMixin):

    def __init__(self, cat_feature=None):
        """Build a cluster tranformer"""
        self.cat_feature = cat_feature

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not 'cluster' in X.columns:
            if self.cat_feature == 'vf_category' or self.cat_feature is None:
                # vf_category means that no category defined
                X['cluster'] = 0.0
            elif isinstance(self.cat_feature, (unicode, str)):
                # category is a feature
                X['cluster'] = X[self.cat_feature]
        return X