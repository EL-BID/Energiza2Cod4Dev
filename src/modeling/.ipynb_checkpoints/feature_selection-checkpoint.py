"""
Módulo para realizar selección de características en conjuntos de datos.
Contiene funciones que implementan diferentes métodos de selección de características.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

def feature_selection_by_correlation(x_train, y_train, variables,method='pearson', th=0.9):
    """
    Realiza la selección de características basada en la correlación entre variables y la variable objetivo.
    
    Args:
        x_train (pd.DataFrame): Conjunto de datos de entrenamiento (variables independientes).
        y_train (pd.Series): Variable objetivo.
        variables (list): Lista de variables a considerar.
        method (str): Método de correlación a utilizar ('pearson' por defecto).
        th (float): Umbral de correlación para considerar variables como altamente correlacionadas (0.9 por defecto).
    
    Returns:
        list: Lista de variables seleccionadas.
    """
    print('Calculando Correlación Entre Variables')
    x_train['target'] = y_train
    df_corr = x_train[variables + ['target']].corr(method=method)
    # Buscar variables mas correlacionadas
    vars_to_drop_corr = []
    for x in variables:
        for y in variables:
            if x != y:
                c_value = df_corr[x][y]
                if np.abs(c_value) > th:
                    corr_x_t = np.abs(df_corr[x]['target'])
                    corr_y_t = np.abs(df_corr[y]['target'])
                    if corr_x_t > corr_y_t:
                        vars_to_drop_corr.append(y)
    x_train.drop(columns=['target'],inplace=True)
    print('Eliminando Variables Altamente Correlacionadas')
    variables = [x for x in variables if x not in vars_to_drop_corr]
    return variables

def feature_selection_by_constant(x_train, y_train, variables, th=0.99):
    """
    Realiza la selección de características eliminando variables con valores constantes en el conjunto de entrenamiento.
    
    Args:
        x_train (pd.DataFrame): Conjunto de datos de entrenamiento (variables independientes).
        y_train (pd.Series): Variable objetivo.
        variables (list): Lista de variables a considerar.
        th (float): Umbral de variabilidad para considerar una variable como constante (0.99 por defecto).
    
    Returns:
        list: Lista de variables seleccionadas.
    """
    num_rows = x_train.shape[0]
    allLabels = variables
    constant_per_feature = {label: x_train[label].value_counts().iloc[0]/num_rows for label in allLabels}
    variables_to_drop = [label for label in allLabels if constant_per_feature [label] > th]
    variables = [x for x in variables if x not in variables_to_drop]
    return variables

def feature_selection_by_boruta(X_train, y_train, N=10):
    """
    Realiza la selección de características utilizando el algoritmo Boruta.

    Args:
        X_train (pd.DataFrame): Conjunto de datos de entrenamiento (variables independientes).
        y_train (pd.Series): Variable objetivo.
        N (int): Número de iteraciones para el algoritmo Boruta (10 por defecto).

    Returns:
        list: Lista de variables seleccionadas.
    """
    d = {}
    X_train = X_train.copy()
    y_train = y_train.copy()
    for i in tqdm(range(N),total=N):
        # Hacemos eliminacion de variables por Boruta.
        X_train['random'] = [np.random.randn() for i in range(len(X_train))]
        rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=8)
        feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=i, perc=100)
        feat_selector.fit(X_train.values, y_train.values)
        ranking = pd.DataFrame({'col': X_train.columns, 'ranking': feat_selector.ranking_}).sort_values(
            ['ranking']).reset_index(drop=True)
        variables = ranking.head(ranking.loc[ranking.col == 'random'].index.values[0]).col.values
        d[i] = variables

    E = {}
    for i in d.keys():
        for j in d[i]:
            if j not in E.keys():
                E[j] = 1
            else:
                E[j] += 1

    variables = [k for k in E.keys() if E[k] >= N // 2]
    return variables