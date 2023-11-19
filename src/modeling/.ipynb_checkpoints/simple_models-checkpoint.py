"""
Módulo modelos_simples.py

Este módulo contiene implementaciones de modelos simples utilizados en análisis de datos.

Clases:
- ChangeTrendPercentajeIdentifierWide: Un clasificador para identificar cambios en el porcentaje de tendencia en datos en formato amplio (wide).
- ConstantConsumptionClassifierWide: Un clasificador para identificar consumos constantes en datos en formato amplio (wide).
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from itertools import groupby



class ChangeTrendPercentajeIdentifierWide(BaseEstimator, ClassifierMixin):
    

    def __init__(self, last_base_value, last_eval_value, threshold, is_wide = True):
        """
        Inicializa la clase ChangeTrendPercentajeIdentifierWide.

        Parámetros:
        - last_base_value: int, el número de valores base utilizados para el cálculo del porcentaje de tendencia.
        - last_eval_value: int, el número de valores de evaluación utilizados para el cálculo del porcentaje de tendencia.
        - threshold: float, el umbral utilizado para determinar si el porcentaje de tendencia indica un fraude.
        - is_wide: bool, indica si los datos de entrada están en formato ancho (wide) o no.

        """
        self.last_base_value = last_base_value
        self.last_eval_value = last_eval_value
        self.threshold = threshold
        self.is_wide = is_wide
        
    def convert_wide(self, df):
        df_wide=pd.pivot(df, index=['index'], columns=['date'], values=['consumo']).reset_index()
        # organizar las columnas con nombres apropiados
        df_wide.columns = ['index']+[str(i)+'_anterior' for i in range(self.last_eval_value + self.last_base_value)][::-1]
        return df_wide
    
    def get_cant_cols(self):
        #obtener columnas base y columnas usadas para evaluar
        cols_base = [str(i)+'_anterior' for i in range(self.last_eval_value+1,self.last_base_value+self.last_eval_value+1)][::-1]#last_base_value
        cols_eval = [str(i)+'_anterior' for i in range(1,self.last_eval_value+1)][::-1]#last_eval_value
#         print('[INFO]...cols base:', cols_base)
#         print('[INFO]...cols eval:', cols_eval)
        return cols_base, cols_eval
        
    def compute_trend_percentage_wide(self, X):
        if self.is_wide==False:
            X = self.convert_wide(X)
        
        cols_base, cols_eval = self.get_cant_cols()
        X['trend_perc'] = 100 * X[cols_eval].mean(axis=1)/(X[cols_base].mean(axis=1)+0.000001)
        return X

    def fit(self, X, y=None):
        
        return self

    def predict(self, X):
        X_copy = X.copy()
        X_copy = self.compute_trend_percentage_wide(X_copy)
        X_copy['is_fraud_trend_perc'] = (100-X_copy['trend_perc']>self.threshold).astype(int)
        return X_copy[['trend_perc','is_fraud_trend_perc']]

    
class ConstantConsumptionClassifierWide(BaseEstimator, ClassifierMixin):
    
    def __init__(self, min_count_constante):
        """
        Inicializa la clase ConstantConsumptionClassifierWide.

        Parámetros:
        - min_count_constante: int, el número mínimo de ocurrencias consecutivas de un valor para ser considerado constante.

        """
        self.min_count_constante = min_count_constante
        
    def fit(self, X, y=None):
        return self
    
    def len_max_consumo_constante_seg(self,consumo):
#         print(consumo)
        g = [[k, len(list(v))] for k, v in groupby(consumo)]
        g = [x for x in g if (x[1] >= self.min_count_constante)]
        if any(g):
            return 1
#             return sorted(g, reverse=True, key=lambda x: x[-1])[0][1]
        else:
            return 0

    def predict(self, X):
        pred = X.apply(lambda x : self.len_max_consumo_constante_seg(x.values),axis=1)
        return pred