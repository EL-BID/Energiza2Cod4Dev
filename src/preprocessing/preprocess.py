from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class ToDummy(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        self.dummy_names = pd.get_dummies(X[self.cols], prefix=['dummy_' + x for x in self.cols],
                                          columns=self.cols).columns
        return self

    def transform(self, X,y = None):
        X = pd.get_dummies(X, prefix=['dummy_' + x for x in self.cols], columns=self.cols)
        cols_dummy_transform = [x for x in X.columns if 'dummy_' in x]
        diff_dummy = list(set(self.dummy_names) - set(cols_dummy_transform))
        for d in diff_dummy:
            X[d] = 0
        
        # puede que existan valores en test y no en train
        diff_dummy = list(set(cols_dummy_transform) - set(self.dummy_names))
        X = X.drop(columns=diff_dummy)
        return X[self.dummy_names]
    
    def get_feature_names(self,params):
        return self.dummy_names
    
class TeEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols, w=20):
        self.cols = cols
        self.w = w
        self.te_var_name = "_".join(cols) + '_prob'

    def fit(self, X, y=None):
        feat = self.cols
        X['target'] = y.values
        self.mean_global = y.mean()
        te = X.groupby(feat)['target'].agg(['mean', 'count']).reset_index()
        te[self.te_var_name] = ((te['mean'] * te['count']) + (self.mean_global * self.w)) / (te['count'] + self.w)
        self.te = te
        return self

    def transform(self, X):
        X = X.merge(self.te[self.cols + [self.te_var_name]], on=self.cols, how='left')
        X[self.te_var_name].fillna(self.mean_global, inplace=True)
        
        for x in self.cols:
            if x in X.columns.tolist():
                X.drop(columns=[x],inplace=True)
        
        X[self.cols[0]] = X[self.te_var_name]
        return X[[self.cols[0]]]
    
    def get_feature_names(self,params):
        return self.te_var_name


class CardinalityReducer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=.1):
        self.threshold = threshold
        
    def find_top_categories(self, feature):
        proportions = feature.value_counts(normalize=True)
        categories = proportions[proportions>=self.threshold].index.values
        return categories
    
    def fit(self, X, y=None):
        self.columns = X.columns
        self.categories = {}
        for feature in self.columns:
            self.categories[feature] = self.find_top_categories(X[feature])
        return self
    
    def transform(self, X):
        X = X.copy()
        for feature in self.columns:
            X[feature] = np.where(X[feature].isin(self.categories[feature]), X[feature], 'otros')
        return X
    
class MinMaxScalerRow(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        scaler = MinMaxScaler()
        return  scaler.fit_transform(X.T).T