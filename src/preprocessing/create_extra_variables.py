import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from itertools import groupby
import warnings
warnings.filterwarnings('ignore')


class ExtraVars(BaseEstimator, TransformerMixin):

    def __init__(self, dao, read=True, num_periodos=3, meses_vacaciones=[]):
        self.num_periodos = num_periodos
        self.read = read
        self.dao = dao
        self.meses_vacaciones = meses_vacaciones

    def fit(self, X, y=None):
        return self

    def obtener_cols_anterior(self, num_cols=12):
        return [f'{i}_anterior_consumo' for i in range(num_cols, 0, -1)]

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
        print("[INFO]...Creating variables")
        # generar listado de cols de atras hacia delante i.e: ['3_anterior', '2_anterior', '1_anterior'], etc.
        cols_3_anterior = self.obtener_cols_anterior(num_cols=self.num_periodos)
        print('cols_3_anterior', cols_3_anterior)
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
        print("[INFO]... Calculating Min, Max, STD, Varianza")
        df_total_super.loc[:, 'min_cons' + num_periodos_str] = df_total_super[cols_3_anterior].min(axis=1)
        df_total_super.loc[:, 'max_cons' + num_periodos_str] = df_total_super[cols_3_anterior].max(axis=1)
        df_total_super.loc[:, 'std_cons' + num_periodos_str] = df_total_super[cols_3_anterior].std(axis=1)
        df_total_super.loc[:, 'var_cons' + num_periodos_str] = df_total_super[cols_3_anterior].var(axis=1)
        ## skewness y kurtosis 3 periodos
        df_total_super.loc[:, 'skew_cons' + num_periodos_str] = df_total_super[cols_3_anterior].skew(axis=1)
        if self.num_periodos > 3:
            df_total_super.loc[:, 'kurt_cons' + num_periodos_str] = df_total_super[cols_3_anterior].kurt(axis=1)

        ##

        df_total_super_avg = df_total_super[cols_3_anterior].expanding(axis=1).mean()
        df_total_super_avg.columns = [x + '_avg_'+ num_periodos_str for x in cols_3_anterior]
        df_total_super_mavg = df_total_super[cols_3_anterior].rolling(3, axis=1).mean()
        df_total_super_mavg.columns = [x + '_mavg_'+ num_periodos_str for x in cols_3_anterior]

        df_total_super = pd.concat([df_total_super, df_total_super_avg], axis=1)
        df_total_super = pd.concat([df_total_super, df_total_super_mavg], axis=1)

        return df_total_super
    
    
class ExtraVarsVacaciones(BaseEstimator, TransformerMixin):

    def __init__(self, dao, read=True, meses_vacaciones=[]):
        self.dao = dao
        self.meses_vacaciones = meses_vacaciones
        
    def fit(self, X, y=None):
        return self
    
    def calc_mes_pred_and_vacaciones(self, df_total_super, meses_vacaciones=[]):
        df_total_super['num_mes_predecir'] = (df_total_super['1_anterior_mes'] + 1).astype(int)
        df_total_super.loc[df_total_super.num_mes_predecir==13,'num_mes_predecir'] = 1
        df_total_super['is_vaca_mes_predecir'] = (df_total_super['num_mes_predecir'].isin(meses_vacaciones))
        df_total_super['is_vaca_mes_predecir'] = df_total_super['is_vaca_mes_predecir'].astype(int)
        return df_total_super
    
    def transform(self, X):
        return self.calc_mes_pred_and_vacaciones(X)


