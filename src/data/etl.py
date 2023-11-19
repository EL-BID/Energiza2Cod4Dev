import pandas as pd
import unidecode
import re

def save_file(df,filename):
    df.to_parquet(filename, index=False)
    
def normalize_text(text):
    # Eliminar acentos
    normalized_text = unidecode.unidecode(text)
    # Eliminar caracteres especiales y convertir a minúscula
    normalized_text = re.sub(r'[^a-zA-Z0-9\sÑñ]', '', normalized_text).lower().strip()
    # Reemplazar paréntesis con espacio
    normalized_text = normalized_text.replace('(', ' ').replace(')', ' ')
    return normalized_text

def corregir_hist_puntos(df):
    # Correccion de nombres para que sea iguales a los actuales
    df.rename(columns={'ciu':'id_ciu',
                                   'id_ciu':'ciu',
                                   'cant_fases':'id_cant_fases',
                                   'id_cant_fases':'cant_fases',
                                   'tipo_medidor':'id_tipo_medidor',
                                   'id_tipo_medidor':'tipo_medidor',
                                   'tarifa':'id_tarifa',
                                   'id_tarifa':'tarifa',
                                   'tipo_sistema':'id_tipo_sistema',
                                   'id_tipo_sistema':'tipo_sistema',
                                   'ciclo_lectura':'id_ciclo_lectura',
                                   'id_ciclo_lectura':'ciclo_lectura',
                                   'circuito':'id_circuito',
                                   'id_circuito':'circuito',
                                   'estado_contrato':'id_estado_contrato',
                                   'id_estado_contrato':'estado_contrato',

                                   }, inplace=True)

    # Correccion de clasif_tarifa
    valid_tarifas = ['1', '4', '2', '28', '9', '6', '5']
    df['tarifa_aux'] = df.tarifa
    df['clasif_tarifa_aux'] = df.clasif_tarifa
    df.loc[~df.tarifa_aux.isin(valid_tarifas),'tarifa'] = df.loc[~df.tarifa_aux.isin(valid_tarifas),'clasif_tarifa']
    df.loc[~df.tarifa_aux.isin(valid_tarifas),'clasif_tarifa'] = df.loc[~df.tarifa_aux.isin(valid_tarifas),'tarifa_aux']
    df.tarifa = df.tarifa.apply(lambda x:  'sin_dato' if x not in valid_tarifas else x)
    df.drop(columns=['tarifa_aux','clasif_tarifa_aux'], inplace=True)
    df.ind_fraude = df.ind_fraude.replace({'#':'n'})
    df.tipo_cliente = df.tipo_cliente.replace({'juridico':'j', 'nacional':'n'})
    return df
    
def run_etl_historico_ordenes(DATA_PATH_RAW, file_name,var_tramites):
    df = pd.read_csv(DATA_PATH_RAW + file_name)
    df.columns = df.columns.str.lower()
    # dates
    dates_cols = ['fecha_creacion','fecha_ejecucion','fecha_cancelacion']
    for x in dates_cols:
        df[x] = pd.to_datetime(df[x])
        
    # str's
    str_cols = ['id','localizacion','id_tipo_orden','tipo_orden','id_cod_resultado','resultado','estado_orden']
    for x in str_cols:
        df[x] = df[x].fillna('sin_dato').astype(str).str.strip().str.split('.').str[0].str.lower()
    
    # debe tener 10 digitos
    df.localizacion = df.localizacion.str.zfill(10)
    # correccion estado orden
    df['estado_orden'] = df['estado_orden'].str.replace('terminado', 'terminada')
    # filtros para ordenes de fraudes
    df_t_f  = df[(df.id_tipo_orden == '47')&
            (df.estado_orden=='terminada')&
           df.resultado.isin(['sin fraude','fraude','irregularidad'])][var_tramites].copy()
    df  = df[var_tramites]
    return df,df_t_f


# Definir función de transformación
def transform_string(s):
    if pd.isnull(s) or s.strip().lower() == "#null#" or s.strip() == "" or s.strip() == '#':
        return 'sin_dato'
    return str(s).strip().lower()

def run_etl_datos_hist_puntos(DATA_PATH_RAW,file_name,dict_types):
    
    df = pd.read_csv(DATA_PATH_RAW + file_name,dtype=dict_types)
    df.columns = df.columns.str.lower()
    str_cols = [x.lower() for x in dict_types.keys()]
    df[str_cols] = df[str_cols].applymap(transform_string)
    # debe tener 10 digitos
    df.localizacion = df.localizacion.str.zfill(10)
    # df.fecha_instalacion = df.fecha_instalacion.replace('1065-10-02 00:00:00.000',None)
    df.fecha_instalacion = pd.to_datetime(df.fecha_instalacion,errors='coerce')
    df.canton = df.id_canton.str.replace('mi','sin_dato')
    df.ind_fraude = df['ind_fraude'].replace("sin_dato",'n')
    df.fecha = pd.to_datetime(df.fecha)
    # 'id_cant_fases', 'cant_fases'
    df.id_cant_fases = df['id_cant_fases'].replace("0",'sin_dato')
    df.cant_fases = df['cant_fases'].replace("0-0",'sin_dato')
    df = df.drop_duplicates(subset=['localizacion','fecha'])
    return df

def run_etl_historico_consumo(DATA_PATH_RAW, file_name, var_fact):
    df = pd.read_csv(DATA_PATH_RAW + file_name)
    df.columns = df.columns.str.lower()
    
    # str's
    str_cols = ['localizacion','nise','cod_facturacion','id_evento_facturacion']
    for x in str_cols:
        df[x] = df[x].fillna('sin_dato').astype(str).str.strip().str.split('.').str[0].str.lower()
    
    # debe tener 10 digitos
    df.localizacion = df.localizacion.str.zfill(10)
    
    # dates
    dates_cols = ['fecha_referencia','fecha_vence','fecha_medicion','dia_pago']
    for x in dates_cols:
        df[x] = pd.to_datetime(df[x])

    # debe tener 10 digitos
    df.localizacion = df.localizacion.str.zfill(10)
    
    #Todo : tratar consumos negativos
    df.consumo_medido = df.consumo_medido.apply(lambda x: None if x<0 else x)

    # filtros
    df = df[var_fact]
    return df

def load_notas_group(DATA_PATH_RAW,file_name):
    df_notas_grupo = pd.read_csv(DATA_PATH_RAW + file_name,sep=';')
    df_notas_grupo.columns = df_notas_grupo.columns.str.lower()
    df_notas_grupo['nota_lector_clean'] = df_notas_grupo.nota_lector.apply(normalize_text)
    df_notas_grupo =  df_notas_grupo.drop_duplicates(subset=['nota_lector','nota_lector_clean','id_note_group'])
    return df_notas_grupo
    
def run_etl_historico_notas_lector(DATA_PATH_RAW,file_name,df_g):
    df = pd.read_csv(DATA_PATH_RAW + file_name)
    df.columns = df.columns.str.lower()
    # localizacion	fecha_referencia	nota_lector
    # str's
    str_cols = ['localizacion']
    for x in str_cols:
        df[x] = df[x].fillna('sin_dato').astype(str).str.strip().str.split('.').str[0].str.lower()
    
    # debe tener 10 digitos
    df.localizacion = df.localizacion.str.zfill(10)
    
    # dates
    dates_cols = ['fecha_referencia']
    for x in dates_cols:
        df[x] = pd.to_datetime(df[x])
    
    # clean texto
    df['nota_lector_clean'] = df.nota_lector.apply(normalize_text)
    
    # agrego grupo
    df = df.merge(df_g[['id','nota_lector_clean','id_note_group']],on = ['nota_lector_clean'], how='left')
    
    # tratamiento de columna grupo
    df.id_note_group = df.id_note_group.astype(str)
    df.id_note_group.fillna('sin_dato', inplace=True)
    
    return df