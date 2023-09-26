from itertools import groupby
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import tsfel
from sklearn.pipeline import Pipeline
logger = logging.getLogger()

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


class TsfelVars(BaseEstimator, TransformerMixin):

    def __init__(self, features_names_path=None,num_periodos=12):
        self.num_periodos = num_periodos
        self.features_names_path = features_names_path

    def obtener_cols_anterior(self, num_cols=12):
        return [f'{i}_anterior' for i in range(num_cols,0, -1)]

    def extra_cols(self, df, domain, cols, window=12):
        cfg = tsfel.get_features_by_domain(domain)
        df_result = tsfel.time_series_features_extractor(cfg, df[cols].values.tolist(),n_jobs=20)
        df_result['index'] = df.index
        return df_result
    
    def compute_by_json(self,df, cols, window=12):
        cfg = tsfel.get_features_by_domain(json_path=self.features_names_path)
        df_result = tsfel.time_series_features_extractor(cfg, df[cols].values,n_jobs=20)
        df_result['index'] = df.index
        return df_result

    def crear_all_tsfel(self, df):
        cols_anterior = self.obtener_cols_anterior(self.num_periodos)
        df_result_stat = self.extra_cols(df, "statistical", cols_anterior, window=self.num_periodos)
        df_result_temporal = self.extra_cols(df, "temporal", cols_anterior, window=self.num_periodos)
        # df_result_spectral = self.extra_cols(df, "spectral", cols_anterior, window=self.num_periodos)
        self.temp_vars = df_result_temporal.columns.tolist()
        self.temp_vars.remove('index')
        self.stat_vars = df_result_stat.columns.tolist()
        self.stat_vars.remove('index')
        # self.spec_vars = df_result_spectral.columns.tolist()
        # self.spec_vars.remove('index')
        return df_result_stat, df_result_temporal #, df_result_spectral

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.features_names_path != None:
            cols_anterior = self.obtener_cols_anterior(self.num_periodos)
            df_tsfel = self.compute_by_json(X, cols_anterior, window=self.num_periodos)
            X = X.merge(df_tsfel, on='index', how='left')
            
        else:
            # df_result_stat, df_result_temporal, df_result_spectral = self.crear_all_tsfel(X)
            df_result_stat, df_result_temporal = self.crear_all_tsfel(X)
            df_tsfel = pd.merge(df_result_stat, df_result_temporal, how='inner', on='index')
            # df_tsfel = pd.merge(df_tsfel, df_result_spectral, how='inner', on='index')
            X = X.merge(df_tsfel, on='index', how='left')

        return X
    

class ExtraVars(BaseEstimator, TransformerMixin):
    def __init__(self,num_periodos=3):
        self.num_periodos = num_periodos

    def fit(self, X, y=None):
        return self

    def obtener_cols_anterior(self, num_cols=12):
        return [f'{i}_anterior' for i in range(num_cols, 0, -1)]

    def transform(self, X):
        return self.create_vbles(X)

    def count_cero(self, x):
        return (x == 0.0).sum()

    def count_cero_seguidos(self, x):
        ceros_seguidos = 2
        consumo = x.values
        g = [[k, len(list(v))] for k, v in groupby(consumo)]
        g = [x for x in g if (x[0] == 0.0) & (x[1] >= ceros_seguidos)]
        if any(g):
            return sorted(g, reverse=True, key=lambda x: x[-1])[0][1]
        else:
            return 0

    def calc_slope(self, x):
        consumo = list(x.values)
        slope = np.polyfit(range(len(consumo)), consumo, 1)[0]
        return slope

    def create_vbles(self, df_total_super):
        # generar listado de cols de atras hacia delante i.e: ['3_anterior', '2_anterior', '1_anterior'], etc.
        cols_3_anterior = self.obtener_cols_anterior(num_cols=self.num_periodos)
        num_periodos_str = str(self.num_periodos)
        ## promedios
        df_total_super.loc[:, 'mean_' + num_periodos_str] = df_total_super[cols_3_anterior].mean(axis=1)
        ## Cantidad de ceros
        df_total_super.loc[:, 'cant_ceros_' + num_periodos_str] = df_total_super[cols_3_anterior].apply(self.count_cero,
                                                                                                        axis=1)
        df_total_super.loc[:, 'max_cant_ceros_seg_' + num_periodos_str] = df_total_super[cols_3_anterior].apply(
            self.count_cero_seguidos, axis=1)
        ## Slope
        df_total_super.loc[:, 'slope_' + num_periodos_str] = df_total_super[cols_3_anterior].apply(self.calc_slope,
                                                                                                   axis=1)
        ## Min, Max, STD, Varianza 3 periodos
        df_total_super.loc[:, 'min_cons' + num_periodos_str] = df_total_super[cols_3_anterior].min(axis=1)
        df_total_super.loc[:, 'max_cons' + num_periodos_str] = df_total_super[cols_3_anterior].max(axis=1)
        df_total_super.loc[:, 'std_cons' + num_periodos_str] = df_total_super[cols_3_anterior].std(axis=1)
        df_total_super.loc[:, 'var_cons' + num_periodos_str] = df_total_super[cols_3_anterior].var(axis=1)
        ## skewness y kurtosis 3 periodos
        df_total_super.loc[:, 'skew_cons' + num_periodos_str] = df_total_super[cols_3_anterior].skew(axis=1)
        if self.num_periodos > 3:
            df_total_super.loc[:, 'kurt_cons' + num_periodos_str] = df_total_super[cols_3_anterior].kurt(axis=1)

        return df_total_super   
    
    
def borrar_uc_nan(df):
    df = df[~(df.uc == 'nan')]
    df = df[~(df.uc.isnull())]
    return df

def llenar_val_vacios_ciclo(df, cant_ciclos_validos):
    cols_consumo = [f'{i}_anterior' for i in range(cant_ciclos_validos, 0, -1)]
    
    df.loc[:, cols_consumo] = df.loc[:, cols_consumo].fillna(method='ffill', axis=1)
    df.loc[:, cols_consumo] = df.loc[:, cols_consumo].fillna(method='bfill', axis=1)
    return df

def llenar_val_vacios_str(df, cols,  str_value):
    for x in cols:
        df.loc[:, x] = df[x].fillna(str_value)
    return df

def llenar_val_vacios_numeric(df, cols,  numeric_value):
    for x in cols:
        df.loc[:, x] = df[x].fillna(numeric_value)
    return df

def llenar_val_vacios_datetime(df, cols,  dt_value):
    for x in cols:
        df.loc[:, x] = df[x].fillna(dt_value)
    return df

def build_feature_engeniering_pipeline(f_names_path,num_periodos):
    pipe_feature_eng_train = Pipeline(
    [
        ("tsfel vars", TsfelVars('all', features_names_path=f_names_path, read=False,num_periodos= num_periodos)),
        ("add vars3", ExtraVars(None, read=False, num_periodos=3)),
        ("add vars6", ExtraVars(None, read=False, num_periodos=6)),
        ("add vars12", ExtraVars(None, read=False, num_periodos=12)),

    ]
        )
    return pipe_feature_eng_train