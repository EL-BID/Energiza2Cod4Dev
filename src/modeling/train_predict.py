import sys
from datetime import date
import pandas as pd
import numpy as np
import os
import joblib
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from src.preprocessing.preprocessing  import llenar_val_vacios_str,llenar_val_vacios_ciclo,TsfelVars, ExtraVars,ToDummy, TeEncoder, CardinalityReducer
from src.modeling.feature_selection import feature_selection_by_constant, feature_selection_by_boruta, feature_selection_by_correlation
from src.modeling.supervised_models import LGBMModel
module_path = os.path.abspath(os.path.join('../../'))
if module_path not in sys.path:
    sys.path.append(module_path)

nombre_carpeta_datos = '../../data'
nombre_carpeta_reportes = '../../reports'
nombre_path_historico_features_selected = nombre_carpeta_datos+'/historico/features_selected.parquet'
nombre_path_historico_X = nombre_carpeta_datos+'/historico/X.parquet'
nombre_path_historico_Y = nombre_carpeta_datos+'/historico/Y.parquet'
nombre_path_historico_modelo_LGBM = nombre_carpeta_datos+'/historico/modelo_lgbm.pkl'
nombre_path_historico_series_features_mes = nombre_carpeta_datos+'/historico/df_series_features_mes.parquet'
nombre_path_historico_features_selected = nombre_carpeta_datos+'/historico/features_selected.parquet'
nombre_path_historico_predicciones_ultimo_mes = nombre_carpeta_datos+'/historico/df_predicciones_ultimo_mes.parquet'
nombre_path_historico_predicciones_ultimo_mes_excel = nombre_carpeta_reportes+f'/reporte_predicciones_{date.today()}.xlsx'
nombre_path_historico_instalacion_cuenta = nombre_carpeta_datos+'/historico/df_instalacion_cuenta.parquet'

def train_lgbm_model():
    print('---Entrenamiento modelo LGBM sobre Dataset---')
    X = pd.read_parquet(nombre_path_historico_X)
    y_aux = pd.read_parquet(nombre_path_historico_Y)
    y = y_aux['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    feat_selec = pd.read_parquet(nombre_path_historico_features_selected)
    feautures_selected = select_by_boruta = feat_selec['boruta'][0].tolist()
    variables_consumo = [x for x in X.columns if '_anterior' in x]# and x!='0_anterior'
    variables_categoricas = ['cant_consumo_est', 'cant_estado_0', 'cant_estado_1', 'cant_estado_2', 'cant_estado_3', 'cant_estado_4', 'mes', 'bimestre', 'trimestre', 'cuatrimestre', 'semestre', 'cant_categorias', 'ult_categoria', 'categ_mas_frecuente', 'cambios_categoria']
    cols_for_model = variables_categoricas+variables_consumo[:-1]+feautures_selected
    # Definimos el metodo de balanceo de clases con su correspondiente umbral y el pipeline de pre-procesamiento de variables categoricas.
    param_imb_method = 'over'#under
    sam_th = 1#1
    periodo = 12
    preprocesor = 1 # Pipeline de variables categoricas
    train_lgbm_model = LGBMModel(cols_for_model,
                             hyperparams=None,
                             search_hip=True,
                             sampling_th = sam_th,
                             preprocesor_num = preprocesor,
                             sampling_method=param_imb_method)
    lgbm_model = train_lgbm_model.train(X_train,y_train,X_test, y_test)
    joblib.dump(lgbm_model, nombre_path_historico_modelo_LGBM)
    print(f'Dimensiones X: {X.shape}, Dimensiones y: {y.shape}')
    print('---Finalizado: Entrenamiento modelo LGBM sobre Dataset---')
    return X_train, X_test, y_train, y_test, cols_for_model, lgbm_model

def predicciones_mes(series_features_mes, umbral_fraude = 0.90):
    variables_categoricas = ['cant_consumo_est','cant_estado_0','cant_estado_1','cant_estado_2','cant_estado_3','cant_estado_4','mes','bimestre','trimestre','cuatrimestre','semestre','cant_categorias','ult_categoria','categ_mas_frecuente','cambios_categoria']
    variables_consumo = [x for x in series_features_mes.columns if '_anterior' in x]
    features_selected = pd.read_parquet(nombre_path_historico_features_selected)
    features_selected = features_selected['boruta'].tolist()[0].tolist()
    cols_for_model = variables_categoricas+variables_consumo+features_selected
    #cols_for_model = [col for col in cols_for_model if col in series_features_mes.columns]
    print('Iniciando carga de modelo y predicción de valores')
    lgbm_model = joblib.load(nombre_path_historico_modelo_LGBM)
    predicciones = lgbm_model.predict_proba(series_features_mes[cols_for_model])[:,1]
    series_features_mes['indice_riesgo'] = predicciones
    
    print('Agregando datos de localización')
    df_instalacion_cuenta = pd.read_parquet(nombre_path_historico_instalacion_cuenta)
    cols_loc = ['ciclo', 'sector', 'ruta', 'manzana', 'secuencia']
    series_features_mes = series_features_mes.merge(df_instalacion_cuenta[['instalacion']+cols_loc], how='inner', on='instalacion')
    series_features_mes = series_features_mes.drop_duplicates(subset = ['instalacion'])
    
    print('Ordenando listado por índice de riesgo')
    series_features_mes.sort_values('indice_riesgo',ascending=False,inplace=True)
    series_features_mes.to_parquet(nombre_path_historico_predicciones_ultimo_mes)
    cols_export = ['instalacion'] + cols_loc + variables_categoricas + variables_consumo + ['indice_riesgo']
    series_excel = series_features_mes.query(f'indice_riesgo>{umbral_fraude}')
    series_excel[cols_export].to_excel(nombre_path_historico_predicciones_ultimo_mes_excel)
    print('--- Finalizada la predicción de índice de riesgo de fraude para el mes ---')
    return series_features_mes