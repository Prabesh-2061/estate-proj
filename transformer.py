from sklearn.base import BaseEstimator, TransformerMixin


class AreaPerBedroomTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['area_per_bedroom'] = X['area'] / X['bedroom'].replace(0, 1)
        return X
