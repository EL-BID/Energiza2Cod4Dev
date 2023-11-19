from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline
from itertools import groupby
from tqdm import tqdm
import pandas as pd
import numpy as np
import tsfel
from datetime import datetime, timedelta

class TsfelVars(BaseEstimator, TransformerMixin):

    def __init__(self, features_names_path=None, num_periodos=12):
        self.num_periodos = num_periodos
        self.features_names_path = features_names_path

    def obtener_cols_anterior(self, num_cols=12):
        return [f'{i}_anterior' for i in range(num_cols,0, -1)]

    def extra_cols(self, df, domain, cols, window=12):
        cfg = tsfel.get_features_by_domain(domain)
        df_result = tsfel.time_series_features_extractor(cfg, df[cols].values.tolist(),verbose=1,n_jobs=-1)
        df_result['index'] = df.index
        print('df_result index:',df_result['index'].shape)
        return df_result
    
    def compute_by_json(self,df, cols, window=12):
        cfg = tsfel.get_features_by_domain(json_path=self.features_names_path)
        df_result = tsfel.time_series_features_extractor(cfg, df[cols].values.tolist(),n_jobs=-1)
        df_result['index'] = df.index
        print('df_result index:',df_result['index'].shape)
        return df_result

    def crear_all_tsfel(self, df):
        print('ENTRI a CREAR TSFEL')
        cols_anterior = self.obtener_cols_anterior(self.num_periodos)
        # df_result_spectral = self.extra_cols(df, "spectral", cols_anterior, window=self.num_periodos)
        df_result_stat = self.extra_cols(df, "statistical", cols_anterior, window=self.num_periodos)
        df_result_temporal = self.extra_cols(df, "temporal", cols_anterior, window=self.num_periodos)
        df_result_spectral = self.extra_cols(df, "spectral", cols_anterior, window=self.num_periodos)
        self.temp_vars = df_result_temporal.columns.tolist()
        self.temp_vars.remove('index')
        self.stat_vars = df_result_stat.columns.tolist()
        self.stat_vars.remove('index')
        self.spec_vars = df_result_spectral.columns.tolist()
        self.spec_vars.remove('index')
        print('temp_vars', len(self.temp_vars))
        print('stat_vars', len(self.stat_vars))
        print('spec_vars', len(self.spec_vars))
        return df_result_stat, df_result_temporal, df_result_spectral

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.features_names_path != None:
            print("ENTRO PARA CREAR JSON")
            cols_anterior = self.obtener_cols_anterior(self.num_periodos)
            df_tsfel = self.compute_by_json(X, cols_anterior, window=self.num_periodos)
            print('df_tsfel', df_tsfel.shape)
            X = X.merge(df_tsfel, on='index', how='left')
            
        else:
            print("ENTRO PARA CREAR")
            df_result_stat, df_result_temporal, df_result_spectral = self.crear_all_tsfel(X)
            df_tsfel = pd.merge(df_result_stat, df_result_temporal, how='inner', on='index')
            df_tsfel = pd.merge(df_tsfel, df_result_spectral, how='inner', on='index')
            print('df_tsfel', df_tsfel.shape)
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
    
class ChangeTrendPercentajeIdentifierWideTransform(BaseEstimator, TransformerMixin):

    def __init__(self, last_base_value, last_eval_value, threshold, is_wide = True):
        self.last_base_value = last_base_value
        self.last_eval_value = last_eval_value
        self.threshold = threshold
        self.is_wide = is_wide
        
    def convert_wide(self, df):
        df_wide=pd.pivot(df, index=['index'], columns=['date'], values=['consumo']).reset_index()
        df_wide.columns = ['index']+[str(i)+'_anterior' for i in range(self.last_eval_value + self.last_base_value)][::-1]
        return df_wide
    
    def get_cant_cols(self):
        #obtener columnas base y columnas usadas para evaluar
        cols_base = [str(i)+'_anterior' for i in range(self.last_eval_value+1,self.last_base_value+self.last_eval_value+1)][::-1]#last_base_value
        cols_eval = [str(i)+'_anterior' for i in range(1,self.last_eval_value+1)][::-1]#last_eval_value
        return cols_base, cols_eval
        
    def compute_trend_percentage_wide(self, X):
        if self.is_wide==False:
            X = self.convert_wide(X)
        
        cols_base, cols_eval = self.get_cant_cols()
        X['trend_perc'] = 100 * X[cols_eval].mean(axis=1)/(X[cols_base].mean(axis=1)+0.000001)
        # X['trend_perc'] = 100*(X[cols_eval].mean(axis=1)-X[cols_base].mean(axis=1))/(X[cols_base].mean(axis=1)+1)
        # X['trend_perc'] = (X[cols_eval].mean(axis=1))/(X[cols_base].mean(axis=1)+1)
        return X

    def fit(self, X, y=None):
        return self
        
    def transform(self, X,y = None):
        X_copy = X.copy()
        X_copy = self.compute_trend_percentage_wide(X_copy)
        X_copy['is_fraud_trend_perc'] = (100-X_copy['trend_perc']>self.threshold).astype(int)
        return X_copy.is_fraud_trend_perc#X_copy[['trend_perc']]
    

class ConstantConsumptionClassifierWide(BaseEstimator, TransformerMixin):
    def __init__(self, min_count_constante):
        self.min_count_constante = min_count_constante
        
    def fit(self, X, y=None):
        return self
    
    def len_max_consumo_constante_seg(self,consumo):
        g = [[k, len(list(v))] for k, v in groupby(consumo)]
        g = [x for x in g if (x[1] >= self.min_count_constante)]
        if any(g):
            return 1
        else:
            return 0

    def transform(self, X,y = None):
        pred = X.apply(lambda x : self.len_max_consumo_constante_seg(x.values),axis=1)
        return pred
    
def remove_outliers(df):
    # Calcular los cuartiles y el rango intercuartil
    Q1 = np.percentile(df.consumo_medido.tolist(), 25)
    Q3 = np.percentile(df.consumo_medido.tolist(), 75)
    IQR = Q3 - Q1

    # Definir los límites para los valores atípicos
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    
    df = df[df.consumo_medido.between(limite_inferior,limite_superior)]
    return df

def remove_outliers_limite_inferior(consumo_medido):
    # Calcular los cuartiles y el rango intercuartil
    Q1 = np.percentile(consumo_medido, 25)
    Q3 = np.percentile(consumo_medido, 75)
    IQR = Q3 - Q1

    # Definir los límites para los valores atípicos
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    return limite_inferior

def remove_outliers_limite_superior(consumo_medido):
    # Calcular los cuartiles y el rango intercuartil
    Q1 = np.percentile(consumo_medido, 25)
    Q3 = np.percentile(consumo_medido, 75)
    IQR = Q3 - Q1
    # Definir los límites para los valores atípicos
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    return limite_superior

def compute_stat_consumo_by_group(df_consumo,fecha_fraud,cols_group,cant_periodos):
    list_df = []
    # for fecha_fraud in tqdm(fecha_fraud_list, total=len(fecha_fraud_list)):
    df_etiquetado_fraud = df_consumo[df_consumo.fecha_referencia < fecha_fraud].copy()
    date_inicial = str(pd.to_datetime(fecha_fraud) - pd.DateOffset(months=cant_periodos))
    df_etiquetado_fraud = df_etiquetado_fraud[df_etiquetado_fraud['fecha_referencia'] >= date_inicial]
    date_semestre = str(pd.to_datetime(fecha_fraud) - pd.DateOffset(months=6))
    df_etiquetado_fraud['is_semestre_1'] = df_etiquetado_fraud.fecha_referencia<date_semestre
    df_etiquetado_fraud = df_etiquetado_fraud[df_etiquetado_fraud.consumo_medido>10]
    df_etiquetado_fraud_s1 = df_etiquetado_fraud[df_etiquetado_fraud.is_semestre_1].copy()
    df_etiquetado_fraud_s2 = df_etiquetado_fraud[~df_etiquetado_fraud.is_semestre_1].copy()
    df_etiquetado_fraud_s1['l_inf'] = df_etiquetado_fraud_s1.groupby(cols_group)['consumo_medido'].transform(remove_outliers_limite_inferior)
    df_etiquetado_fraud_s1['l_sup'] = df_etiquetado_fraud_s1.groupby(cols_group)['consumo_medido'].transform(remove_outliers_limite_superior)
    df_etiquetado_fraud_s1 = df_etiquetado_fraud_s1[df_etiquetado_fraud_s1.consumo_medido.between(df_etiquetado_fraud_s1.l_inf,df_etiquetado_fraud_s1.l_sup)]
    df_etiquetado_fraud_s2['l_inf'] = df_etiquetado_fraud_s2.groupby(cols_group)['consumo_medido'].transform(remove_outliers_limite_inferior)
    df_etiquetado_fraud_s2['l_sup'] = df_etiquetado_fraud_s2.groupby(cols_group)['consumo_medido'].transform(remove_outliers_limite_superior)
    df_etiquetado_fraud_s2 = df_etiquetado_fraud_s2[df_etiquetado_fraud_s2.consumo_medido.between(df_etiquetado_fraud_s2.l_inf,df_etiquetado_fraud_s2.l_sup)]
    df_etiquetado_fraud_s1 = df_etiquetado_fraud_s1.groupby(cols_group).consumo_medido.describe().reset_index()
    df_etiquetado_fraud_s1['periodos'] = 'first_6'
    df_etiquetado_fraud_s2 = df_etiquetado_fraud_s2.groupby(cols_group).consumo_medido.describe().reset_index()
    df_etiquetado_fraud_s2['periodos'] = 'last_6'
    df_etiquetado_fraud_s1['date_fizcalizacion'] = fecha_fraud
    df_etiquetado_fraud_s2['date_fizcalizacion'] = fecha_fraud
    list_df.append(df_etiquetado_fraud_s1)
    list_df.append(df_etiquetado_fraud_s2)
    return pd.concat(list_df)

def consumo_below_mean_group(df,th, df_stat_consumo, cols_group):
    df = df.merge(df_stat_consumo, on = cols_group+ ['date_fizcalizacion'], how = 'left')
    first_6 = [f'{i}_anterior' for i in range(6, 0, -1)]
    last_6  = [f'{i}_anterior' for i in range(12, 6, -1)]
    firs_6_col_name = f'is_first_6_below_{"".join(cols_group)}'
    last_6_col_name = f'is_last_6_below_{"".join(cols_group)}'
    df[firs_6_col_name] = (((df.first_6-df[first_6].mean(axis=1))/df.first_6*100)>th).astype(int)
    df[last_6_col_name] = (((df.last_6-df[last_6].mean(axis=1))/df.last_6*100)>th).astype(int)
    df.drop(columns=['first_6','last_6'], inplace=True)
    
    return df

def llenar_val_vacios_ciclo(df, cant_ciclos_validos):
    cols_consumo = [f'{i}_anterior' for i in range(cant_ciclos_validos, 0, -1)]
    df.loc[:, cols_consumo] = df.loc[:, cols_consumo].fillna(method='ffill', axis=1)
    df.loc[:, cols_consumo] = df.loc[:, cols_consumo].fillna(method='bfill', axis=1)
    return df

def compute_change_trend_percentaje_vars(df,config_caidas):
    for c in config_caidas:
        last_base_value,last_eval_value,threshold = c
        trend_perc_model = ChangeTrendPercentajeIdentifierWideTransform(last_base_value,last_eval_value,threshold)
        df[f'trend_perc_{c[0]}_{c[1]}'] = trend_perc_model.fit_transform(df)
    return df

def compute_constant_consumption_vars(df,config_constantes):
    cols_cons = [str(i)+'_anterior' for i in range(1,13)][::-1]
    for c in config_constantes:
        const_model = ConstantConsumptionClassifierWide(c)
        df[f'constant_{c}'] = const_model.fit_transform(df[cols_cons])
    return df

def compute_tsfel_consumption_vars(df,cant_periodos):
    pipe_feature_eng_train = Pipeline(
        [
            ("tsfel vars", TsfelVars(features_names_path=None, num_periodos= cant_periodos)),
            ("add vars3", ExtraVars(num_periodos=3)),
            ("add vars6", ExtraVars(num_periodos=6)),
            ("add vars12", ExtraVars(num_periodos=12)),

        ]
    )
    df = pipe_feature_eng_train.fit_transform(df, None)
    return df

def compute_stat_consumo_group(df,config_below_g,fecha_fraud):
    for g,f in config_below_g:
        df_stat = pd.read_parquet(f.format(fecha_fraud))
        df = consumo_below_mean_group(df,50, df_stat, g)
    return df


def prepare_stat_consumo_below_group(fecha_fraud_list,config_below_g,df_consumo,cant_periodos):
    for cols_group,file in config_below_g:
        for f in tqdm(fecha_fraud_list, total=len(fecha_fraud_list)):
            df_stat = compute_stat_consumo_by_group(df_consumo, f, cols_group, cant_periodos)
            df_stat = df_stat.pivot(index=cols_group + ['date_fizcalizacion'],
                                                                    columns=['periodos'], values='mean').reset_index()
            df_stat['first_6'] = df_stat['first_6'].fillna(df_stat['last_6'])
            df_stat['last_6'] = df_stat['last_6'].fillna(df_stat['first_6'])
            df_stat.date_fizcalizacion = pd.to_datetime(df_stat.date_fizcalizacion)
            df_stat.to_parquet(file.format(f))
            

def prepare_data(df_ordenes,df_consumo,df_notas,df_hist_puntos,df_static_puntos):
    
    df_hist_puntos.rename(columns={'fecha':'fecha_referencia'}, inplace=True)
    df_static_puntos.rename(columns={'fecha':'fecha_referencia'}, inplace=True)
    df_consumo.sort_values('fecha_referencia', inplace=True)
    df_hist_puntos.sort_values('fecha_referencia', inplace=True)
    df_static_puntos.sort_values('fecha_referencia', inplace=True)
    df_static_puntos = df_static_puntos.drop_duplicates(subset=['localizacion'],keep='last')
    
    df_consumo.consumo_medido = df_consumo.consumo_medido.apply(lambda x: None if x<0 else x)
    df_consumo['is_mora'] = df_consumo.dia_pago>df_consumo.fecha_vence
    df_consumo = df_consumo.merge(df_static_puntos[['localizacion','id_distrito','id_cant_fases']], on = 'localizacion', how='left')
    df_consumo = df_consumo.merge(df_hist_puntos[['localizacion','fecha_referencia','id_tarifa']],on = ['localizacion','fecha_referencia'], how='left')
    df_consumo['control_manzana'] = df_consumo.localizacion.str[:6]
    
    # Tramites : nos quedamos con un tramite por localizacion y fecha de referencia
    df_ordenes['fecha_referencia'] = df_ordenes.fecha_creacion.dt.date.astype(str).str.split('-').str[:-1].apply(lambda x: "-".join(x)+"-01")
    df_ordenes['fecha_referencia'] = pd.to_datetime(df_ordenes['fecha_referencia'])
    df_ordenes['cant_tramites'] = df_ordenes.groupby(['fecha_referencia','localizacion']).id.transform('count')
    df_ordenes_not_duplicates = df_ordenes[df_ordenes.cant_tramites==1].copy()
    df_ordenes_duplicates = df_ordenes[df_ordenes.cant_tramites>1].copy()
    df_ordenes_duplicates['resultado_num'] = df_ordenes_duplicates.resultado.map({'sin fraude' : 0,'fraude':1, 'irregularidad':1 })
    df_ordenes_duplicates = df_ordenes_duplicates.sort_values(['localizacion','fecha_referencia','resultado_num']).drop_duplicates(subset=['localizacion','fecha_referencia'],keep='last')
    df_ordenes = pd.concat([df_ordenes_not_duplicates,df_ordenes_duplicates])#df_ordenes_not_duplicates.append(df_ordenes_duplicates)
    df_ordenes['target'] = df_ordenes.resultado.map({'sin fraude' : 0,'fraude':1, 'irregularidad':1})
    return df_ordenes,df_consumo,df_notas,df_hist_puntos,df_static_puntos


def create_dataset_by_date(fecha_fraud,df_consumo,df_ordenes,df_hist_puntos,df_notas,df_static_puntos,cant_periodos,
                           config_below_g,config_caidas,config_constantes,var_puntos,select_localizacion=None):
    
    df_etiquetado_fraud = df_consumo[df_consumo.fecha_referencia < fecha_fraud].copy()
    date_inicial = str(pd.to_datetime(fecha_fraud) - pd.DateOffset(months=cant_periodos))
    df_etiquetado_fraud = df_etiquetado_fraud[df_etiquetado_fraud['fecha_referencia'] >= date_inicial]
    if select_localizacion is None:
        select_localizacion = df_ordenes[(df_ordenes.fecha_referencia == fecha_fraud)].localizacion.unique().tolist()
    
    df_etiquetado_fraud = df_etiquetado_fraud[df_etiquetado_fraud.localizacion.isin(select_localizacion)]

    # a los consumos le agregamos los fraudes anteriores
    df_etiquetado_fraud = df_etiquetado_fraud.merge(df_ordenes[['localizacion','fecha_referencia','target']],\
                              on=['localizacion','fecha_referencia'],how='left')
    df_etiquetado_fraud.target = df_etiquetado_fraud.target.fillna(0)

    # a los consumos le agregamos variables de puntos
    df_etiquetado_fraud = df_etiquetado_fraud.merge(df_hist_puntos[['localizacion','fecha_referencia','estado_medidor','estado_contrato']],\
                          on=['localizacion','fecha_referencia'],how='left')

    # a los consumos le agregamos variables de notas
    df_etiquetado_fraud = df_etiquetado_fraud.merge(df_notas[['localizacion','fecha_referencia','id_note_group']],\
                          on=['localizacion','fecha_referencia'],how='left')
    df_etiquetado_fraud.id_note_group = df_etiquetado_fraud.id_note_group.fillna('3')

    df_etiquetado_fraud.replace('sin_dato',None, inplace=True)

    # cant nise, cant_ev_fact, cant mora, cant fraud
    df_cant = df_etiquetado_fraud.groupby(['localizacion']).\
    agg({'nise':pd.Series.nunique,
         'id_evento_facturacion':pd.Series.nunique,
         'is_mora':sum,
         'target':sum}).\
    rename(columns={'nise':'cant_nise',
                    'id_evento_facturacion':'cant_ev_fact',
                    'is_mora':'cant_mora',
                    'target':'cant_fraud'}).\
    reset_index()

    # cant tipo evento facturacion
    selec_efact_type = ['e01','5','12','E20']
    df_ev_lperiodica = df_etiquetado_fraud.groupby(['localizacion']).\
    id_evento_facturacion.value_counts(normalize=False).\
    unstack(fill_value=0)
    cols_name = df_ev_lperiodica.columns.tolist()
    not_in_df = list(set(selec_efact_type) - set(cols_name))
    df_ev_lperiodica = df_ev_lperiodica.assign(**dict(zip(not_in_df, [0]*len(not_in_df))))[selec_efact_type].\
                       reset_index()
    df_ev_lperiodica = df_ev_lperiodica.\
                    rename(columns={'e01':'cant_efactype_e01',
                                    '5':'cant_efac_type_5',
                                    '12':'cant_efac_type_12',
                                    'E20':'cant_efac_type_E20'})

    # cant 'estado_contrato','tarifa','clasif_tarifa'
    df_cant_2 = df_etiquetado_fraud.groupby(['localizacion']).\
    agg({'estado_contrato':pd.Series.nunique,
         'id_tarifa':pd.Series.nunique,
         'estado_medidor' : pd.Series.nunique,
         }).\
    rename(columns={'estado_contrato':'cant_estado_contrato',
                    'id_tarifa':'cant_tarifa',
                    'estado_medidor' : 'cant_estado_medidor'
                    }).\
    reset_index()

    # cambios en la serie
    df_cambios = df_etiquetado_fraud.groupby('localizacion').\
    agg({'id_tarifa': lambda serie: (serie.dropna() != serie.dropna().shift()).sum() - 1 ,
         'estado_medidor': lambda serie: (serie.dropna() != serie.dropna().shift()).sum() - 1 },).\
    rename(columns={'id_tarifa':'cambios_tarifa',
                    'estado_medidor':'cambios_estado_medidor',
                   }).\
    reset_index()

    # estado_contrato,estado_medidor
    df_cant_3 = df_etiquetado_fraud.groupby('localizacion').\
    agg({'estado_contrato':lambda serie: (serie=='inactivo').sum(),
        'estado_medidor':lambda serie: (serie=='n').sum()}

       ).\
    rename(columns={'estado_contrato':'cant_estado_contrato_i',
                   'estado_medidor':'cant_estado_medidor_n',}).\
    reset_index()

    # cant tipo evento notas 12
    selec_evnotas_type = ['3', '6', '9', '1', '2', '4', '5']
    df_ev_notas = df_etiquetado_fraud.groupby(['localizacion']).\
    id_note_group.value_counts(normalize=False).\
    unstack(fill_value=0)
    cols_name = df_ev_notas.columns.tolist()
    not_in_df = list(set(selec_evnotas_type) - set(cols_name))
    df_ev_notas = df_ev_notas.assign(**dict(zip(not_in_df, [0]*len(not_in_df))))[selec_evnotas_type].\
                       reset_index()
    df_ev_notas = df_ev_notas.\
                    rename(columns={'3':f'l_{cant_periodos}_notas_g3',
                                    '6':f'l_{cant_periodos}_notas_g6',
                                    '9':f'l_{cant_periodos}_notas_g9',
                                    '1':f'l_{cant_periodos}_notas_g1',
                                    '2':f'l_{cant_periodos}_notas_g2',
                                    '4':f'l_{cant_periodos}_notas_g4',
                                    '5':f'l_{cant_periodos}_notas_g5'
                                   })

    # cant tipo evento notas 3
    cant_periodos_n = 3
    date_inicial = str(pd.to_datetime(fecha_fraud) - pd.DateOffset(months=cant_periodos_n))
    selec_evnotas_type = ['3', '6', '9', '1', '2', '4', '5']
    df_ev_notas_3 = df_etiquetado_fraud[df_etiquetado_fraud['fecha_referencia']>= date_inicial].groupby(['localizacion']).\
    id_note_group.value_counts(normalize=False).\
    unstack(fill_value=0)
    cols_name = df_ev_notas_3.columns.tolist()
    not_in_df = list(set(selec_evnotas_type) - set(cols_name))
    df_ev_notas_3 = df_ev_notas_3.assign(**dict(zip(not_in_df, [0]*len(not_in_df))))[selec_evnotas_type].\
                       reset_index()
    df_ev_notas_3 = df_ev_notas_3.\
                    rename(columns={'3':f'l_{cant_periodos_n}_notas_g3',
                                    '6':f'l_{cant_periodos_n}_notas_g6',
                                    '9':f'l_{cant_periodos_n}_notas_g9',
                                    '1':f'l_{cant_periodos_n}_notas_g1',
                                    '2':f'l_{cant_periodos_n}_notas_g2',
                                    '4':f'l_{cant_periodos_n}_notas_g4',
                                    '5':f'l_{cant_periodos_n}_notas_g5'
                                   })

    # consumos
    df_var_to_add = df_etiquetado_fraud.drop_duplicates(subset=['localizacion'],keep='last')[['localizacion','id_distrito','id_cant_fases','id_tarifa']]
    df_var_to_add.fillna('sin_dato', inplace=True)
    cols_ant = [str(x) + '_anterior' for x in range(cant_periodos, 0, -1)]
    df_etiquetado_fraud = df_etiquetado_fraud.pivot_table(index=['localizacion'], columns=['fecha_referencia'], values='consumo_medido')
    df_etiquetado_fraud.columns = cols_ant
    df_etiquetado_fraud['date_fizcalizacion'] = fecha_fraud
    df_etiquetado_fraud.date_fizcalizacion = pd.to_datetime(df_etiquetado_fraud.date_fizcalizacion)
    df_etiquetado_fraud.reset_index(inplace=True)
    df_etiquetado_fraud['control_manzana'] = df_etiquetado_fraud.localizacion.str[:6]
    df_etiquetado_fraud = df_etiquetado_fraud.merge(df_var_to_add, on = 'localizacion')

    # add other vars
    df_etiquetado_fraud['cant_null_row'] = df_etiquetado_fraud[cols_ant].isnull().sum(axis=1)
    df_etiquetado_fraud = df_etiquetado_fraud.merge(df_cant, on=['localizacion'])
    df_etiquetado_fraud = df_etiquetado_fraud.merge(df_ev_lperiodica, on=['localizacion'])
    df_etiquetado_fraud = df_etiquetado_fraud.merge(df_cant_2, on=['localizacion'])
    df_etiquetado_fraud = df_etiquetado_fraud.merge(df_cambios, on=['localizacion'])
    df_etiquetado_fraud = df_etiquetado_fraud.merge(df_cant_3, on=['localizacion'])
    df_etiquetado_fraud = df_etiquetado_fraud.merge(df_ev_notas, on=['localizacion'])
    df_etiquetado_fraud = df_etiquetado_fraud.merge(df_ev_notas_3, on=['localizacion'], how='left')
    df_etiquetado_fraud = compute_stat_consumo_group(df_etiquetado_fraud,config_below_g,fecha_fraud)
    df_etiquetado_fraud = llenar_val_vacios_ciclo(df_etiquetado_fraud, cant_periodos) 
    df_etiquetado_fraud = compute_change_trend_percentaje_vars(df_etiquetado_fraud,config_caidas)
    df_etiquetado_fraud = compute_constant_consumption_vars(df_etiquetado_fraud,config_constantes)

    var_puntos_selected = list(set(var_puntos)-set(df_etiquetado_fraud.columns.tolist()))
    df_etiquetado_fraud = df_etiquetado_fraud.merge(df_static_puntos[['localizacion']+var_puntos_selected], on=['localizacion'], how='left')
    df_etiquetado_fraud = df_etiquetado_fraud.merge(df_ordenes[['id','localizacion','fecha_referencia','resultado','target']],
                                              left_on=['localizacion', 'date_fizcalizacion'],
                                              right_on=['localizacion', 'fecha_referencia'])
    return df_etiquetado_fraud

def var_notas_l3_correction(df):
    var_notas_l3 =  ['l_3_notas_g3', 'l_3_notas_g6', 'l_3_notas_g9','l_3_notas_g1', 'l_3_notas_g2', 'l_3_notas_g4', 'l_3_notas_g5']
    df[var_notas_l3] = df[var_notas_l3].fillna(0)
    df['total_notas_3'] = df[var_notas_l3].sum(axis=1)
    df.l_3_notas_g3 = (3 - df.total_notas_3 + df.l_3_notas_g3)
    return df
    
def create_train_data(fecha_fraud_list,df_consumo,df_ordenes,df_hist_puntos,df_notas,df_static_puntos,
                                                  cant_periodos,config_below_g,config_caidas,
                                                 config_constantes,var_puntos, file_name_traindata, file_name_traindata_final):
    
    for fecha_fraud in tqdm(fecha_fraud_list, total=len(fecha_fraud_list)):
        df = create_dataset_by_date(fecha_fraud,df_consumo,df_ordenes,df_hist_puntos,df_notas,df_static_puntos,
                                                  cant_periodos,config_below_g,config_caidas,
                                                 config_constantes,var_puntos)

        df = var_notas_l3_correction(df)
        df.to_parquet(file_name_traindata.format(fecha_fraud))

    list_df = []
    for fecha_fraud in tqdm(fecha_fraud_list, total=len(fecha_fraud_list)):
        df = pd.read_parquet(file_name_traindata.format(fecha_fraud))
        list_df.append(df)

    df_etiquetado_fraud = pd.concat(list_df) 
    df_etiquetado_fraud.reset_index(drop=True, inplace=True)
    df_etiquetado_fraud['index'] = range(len(df_etiquetado_fraud))
    df_etiquetado_fraud.to_parquet(file_name_traindata_final.format('final'))
    # tsfel
    df_etiquetado_fraud = compute_tsfel_consumption_vars(df_etiquetado_fraud,cant_periodos)
    df_etiquetado_fraud.to_parquet(file_name_traindata_final.format('final_tsfel'))

def create_predict_data(fecha_fraud,select_localizacion,df_consumo,df_ordenes,df_hist_puntos,df_notas,df_static_puntos,
                                                  cant_periodos,config_below_g,config_caidas,
                                                 config_constantes,var_puntos, file_name_inference):
    
    df_etiquetado_fraud = create_dataset_by_date(fecha_fraud,df_consumo,df_ordenes,df_hist_puntos,df_notas,df_static_puntos,
                                              cant_periodos,config_below_g,config_caidas,
                                             config_constantes,var_puntos,select_localizacion)

    df_etiquetado_fraud = var_notas_l3_correction(df_etiquetado_fraud)
    
    df_etiquetado_fraud.reset_index(drop=True, inplace=True)
    df_etiquetado_fraud['index'] = range(len(df_etiquetado_fraud))
    df_etiquetado_fraud.to_parquet(file_name_inference.format('final'))
    # tsfel
    df_etiquetado_fraud = compute_tsfel_consumption_vars(df_etiquetado_fraud,cant_periodos)
    df_etiquetado_fraud.to_parquet(file_name_inference.format('final_tsfel'))    
    
def get_start_end_years_and_years_range(date_list, periodos):
    # Convertir las fechas a objetos datetime
    date_objects = [datetime.strptime(date_str, '%Y-%m-%d') for date_str in date_list]
    # Encontrar la fecha mínima y máxima
    min_date = min(date_objects)
    max_date = max(date_objects)
    # Calcular el año de inicio (periodos meses antes del año de la fecha mínima)
    start_year = min_date.year - 1 if min_date.month <= periodos else min_date.year
    # Calcular el año de fin (año de la fecha máxima)
    end_year = max_date.year
    # Generar la lista de años desde el año de inicio hasta el año de fin
    years_range = list(range(start_year, end_year + 1))
    return years_range

    
def load_data(data_name,lista_anios, cols=None):
    list_df = [] 
    for y in tqdm(lista_anios, total = len(lista_anios)):
        df = pd.read_parquet(data_name.format(y),columns=cols)
        list_df.append(df)
    return pd.concat(list_df)