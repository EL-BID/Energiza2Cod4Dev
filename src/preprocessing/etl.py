import sys
from datetime import datetime
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from src.preprocessing.preprocessing import llenar_val_vacios_ciclo, TsfelVars, ExtraVars
from src.modeling.feature_selection import feature_selection_by_constant, feature_selection_by_boruta
module_path = os.path.abspath(os.path.join('../../'))
if module_path not in sys.path:
    sys.path.append(module_path)

nombre_carpeta_datos = '../../data'
nombre_path_historico = nombre_carpeta_datos+'/historico/df_raw_completo.parquet'
nombre_path_historico_instalacion_cuenta = nombre_carpeta_datos+'/historico/df_instalacion_cuenta.parquet'
nombre_path_historico_inspecciones = nombre_carpeta_datos+'/historico/df_inspecciones_completo.parquet'
nombre_path_historico_datos_completos_etiquetados = nombre_carpeta_datos+'/historico/df_completo_etiquetado_train.parquet'
nombre_path_historico_series_etiquetadas = nombre_carpeta_datos+'/historico/df_series_etiquetadas_train.parquet'
nombre_path_historico_series_features = nombre_carpeta_datos+'/historico/df_series_features_train.parquet'
nombre_path_historico_features_selected = nombre_carpeta_datos+'/historico/features_selected.parquet'
nombre_path_historico_series_mes = nombre_carpeta_datos+'/historico/df_series_mes.parquet'
nombre_path_historico_series_features_mes = nombre_carpeta_datos+'/historico/df_series_features_mes.parquet'
nombre_path_historico_series_features_mes_parcial = nombre_carpeta_datos+'/historico/df_series_features_mes_parcial.parquet'
nombre_path_historico_X = nombre_carpeta_datos+'/historico/X.parquet'
nombre_path_historico_Y = nombre_carpeta_datos+'/historico/Y.parquet'
nombre_carpeta_consumo = nombre_carpeta_datos+'/consumo'
nombre_carpeta_procesados = nombre_carpeta_datos+'/procesados'
nombre_carpeta_inspecciones = nombre_carpeta_datos+'/inspecciones'

def transform_raw_input_consumos(df):
    '''
    CONVIERTE EL ARCHIVO EXCEL DE ENTRADA EN FORMATO DE EPMAPS AL FORMATO DE DATASET DE CONSUMO PARA UNIR CON LAS INFRACCIONES
    '''
    print('---Transformación de datos nuevos de consumo---')
    df.columns = df.columns.str.lower()
    df.rename(columns={"cuenta contrato": "numcta","feclvcálc":"mesfac","instalación":"instalacion","mz":"manzana","sec":"secuencia"}, inplace=True)
    df['instalacion'] = pd.to_numeric(df['instalacion'], errors='coerce')
    df = df[(~df['mesfac'].isnull()) & (~df['instalacion'].isnull())]
    df['instalacion'] = df['instalacion'].astype(int)
    df['instalacion'] = df['instalacion'].astype('string[pyarrow]')
    df['mesfac'] = df['mesfac'].astype('string[pyarrow]')
    df = df[ (df['mesfac'].str.startswith('20'))]

    df['year']=df['mesfac'].str[:4]
    df['mes']=df['mesfac'].str[5:7]
    df['mes'] = pd.to_numeric(df['mes'], errors='coerce')
    df = df[~df['mes'].isnull()]
    df['mes'] = df['mes'].astype(int)
    df = df[(df['mes']>=1) & (df['mes']<=12)]
    df['mes'] = df['mes'].astype('string[pyarrow]').str.zfill(2)
    df['date']= df['mes'].astype('str')+'-'+df['year'].astype('str')
    df.date = pd.to_datetime(df.date)

    campos_int = ['consumo medido','cl.instalación']
    for col in campos_int:
        df[col] = pd.to_numeric(df[col].astype('str').str.replace(',','.').str.extract('(\d+)', expand=False), errors='coerce',downcast='unsigned').fillna(0).astype('int32')#Agregando r quita el warning: r'(\d+)

    #Datos categóricos
    cat_tipo_tarifa = ['DOM','COM','OFI','PUB','IND','MUN','SAL']
    df['tipo de tarifa'] = pd.Series(np.where(df['tipo de tarifa'].isin(cat_tipo_tarifa),df['tipo de tarifa'],None)).astype('category')
    cat_tipo_fact = ['Real','Estimada']
    df['tipo facturación'] = pd.Series(np.where(df['tipo facturación'].isin(cat_tipo_fact),df['tipo facturación'],None)).astype('category')
    cat_cl_inst = [1,2,3,4]
    df['cl.instalación'] = pd.Series(np.where(df['cl.instalación'].isin(cat_cl_inst),df['cl.instalación'],None)).astype('category')
    df['ciclo'] = df['ciclo'].str.extract(r'([A-Z]0\d{2})', expand=False).astype('category')
    df['ramo'] = df['ramo'].str.extract(r'([A-Z]{,3}\_\d{,2}\_\d{,2}|[A-Z]{,3})', expand=False).astype('string[pyarrow]')
    df['código hidráulico'] = df['código hidráulico'].str.extract(r'(\d{1,3}\_\d{1,3}\_\d{1,3}\_\d{1,3})', expand=False).astype('category')

    #QUITAR CARACTERES RAROS QUE QUEDARON EN LUGAR DE LA BARRA EN MESFAC
    df.loc[ ~df.mesfac.str.contains('/'),'mesfac'] = df[ ~df.mesfac.str.contains('/') ].year.astype(str) + '/' + df[ ~df.mesfac.str.contains('/') ].mes.astype(str)

    #Convertir object a string[pyarrow]
    campos_object = df[df.columns[df.dtypes == 'object']].dtypes.keys().to_list()
    for col in campos_object:
        df[col] = pd.Series(df[col], dtype="string[pyarrow]")

    #Cambiar nombres a campos
    df.rename(columns={'tipo de tarifa':'categoria','tipo facturación':'metodo_consumo','consumo medido':'consumo_medido'}, inplace=True)
    df.rename(columns={'cl.instalación':'cl_inst','código hidráulico':'cod_hidraulico'}, inplace=True)
    df = df[['mesfac', 'numcta', 'instalacion', 'ciclo', 'sector', 'ruta', 'manzana', 'secuencia', 'categoria',
        'consumo_medido', 'metodo_consumo', 'cl_inst', 'ramo', 'cod_hidraulico', 'year', 'mes', 'date']]
    print(f'Dimensiones raw input final: {df.shape}')
    print('---Finalizado: Transformación de datos nuevos de consumo---')
    return df

def concatenar_nuevos_datos_consumo():
    '''
    AÑADE NUEVOS DATOS DE CONSUMO AL HISTORICO COMPLETO:
        DEVUELVE DF_HISTORICO CON LOS ULTIMOS ARCHIVOS INCORPORADOS AL DATASET Y DF_INSTALACION_CUENTA CON MAPEO DE NUMCTA E INSTALACION
        Busca archivos nuevos en la carpeta de datos de consumo, y mueve esos archivos a la carpeta procesados
        Devuelve el df completo con los datos nuevos agregados y un df para mapear numcta con instalación actualizado al último consumo
    '''
    print('---Importación de datos nuevos de consumos---')
    if os.path.exists(nombre_path_historico):
        df_historico = pd.read_parquet(nombre_path_historico)
    else:
        df_historico = pd.DataFrame()
    print(f'Tamaño original df datos histórico: {df_historico.shape}')
    contenido = os.listdir(nombre_carpeta_consumo)
    df_consumo = pd.DataFrame()
    for elemento in contenido:
        ruta_completa = os.path.join(nombre_carpeta_consumo, elemento)
        if os.path.isfile(ruta_completa):  # miramos si es fichero
            agregar = False
            if elemento[-4:].lower()=='.csv':
                df_in = pd.read_csv(ruta_completa, sep=';', encoding='latin-1', warn_bad_lines=True, error_bad_lines=False, lineterminator='\n')
                agregar = True
            elif elemento[-5:].lower()=='.xlsx':
                df_in = pd.read_excel(ruta_completa, engine='openpyxl')
                agregar = True
            if agregar:
                df_in = transform_raw_input_consumos(df_in)
                #df_consumo = df_consumo.append(pd.concat([df_in], axis=0, ignore_index=True), ignore_index=True)
                df_consumo = pd.concat([df_consumo, df_in], ignore_index=True)
                print(f'Añadido: {ruta_completa}')
                #MOVEMOS ARCHIVO A CARPETA PROCESADOS
                os.rename(os.path.join(nombre_carpeta_consumo, elemento), os.path.join(nombre_carpeta_procesados, elemento))
    print(f'Tamaño datos agregados: {df_consumo.shape}')
    
    # 18-11-2023: Agregamos la columna ramo al dataset final, 
    # por lo que hay que verificar que esté presente en el histórico
    if len(df_historico)>0 and 'ramo' not in df_historico.columns.tolist():
        df_historico['ramo'] = ''
        df_historico['ramo'] = df_historico['ramo'].astype('string[pyarrow]')
        df_historico = df_historico[ df_consumo.columns.tolist() ]
    # SI EL CONCAT DA PROBLEMA PUEDE SER PORQUE LOS TIPOS DE LAS COLUMNAS NO SON LOS MISMOS, O LOS NOMBRES NO LO SON
    if len(df_consumo > 0):
        df_historico = pd.concat([df_historico,df_consumo.astype(df_historico.dtypes)])
    # RECONVERTIR A LOS TIPOS DE DATOS ORIGINALES STRING Y CATEGORY
    cols_string = ['mesfac', 'numcta', 'instalacion', 'year', 'mes', 'ramo']
    cols_categ = ['ciclo', 'categoria', 'metodo_consumo', 'cl_inst', 'cod_hidraulico']
    for col in cols_string:
        df_historico[col] = df_historico[col].astype('string[pyarrow]')
    for col in cols_categ:
        df_historico[col] = df_historico[col].astype('category')
    # CAMBIAR LOS VALORES DE ESTADOS A STR FLOAT
    df_historico.cl_inst = df_historico.cl_inst.map({'0':'0.0','1':'1.0','2':'2.0','3':'3.0','4':'4.0','0.0':'0.0','1.0':'1.0','2.0':'2.0','3.0':'3.0','4.0':'4.0'})
    
    #REMOVER DATOS DE MÁS DE 30 MESES (TRES AÑOS)
    date_min = str(pd.to_datetime('today')- pd.DateOffset(months = 36))
    df_historico = df_historico[ df_historico['date']>= date_min]
    
    df_historico.to_parquet(nombre_path_historico)
    print(f'Tamaño final df datos histórico: {df_historico.shape}')
    cols_reg_maestro = ['numcta', 'instalacion','ciclo', 'sector', 'ruta', 'manzana', 'secuencia']
    df_instalacion_cuenta = df_historico[['date']+cols_reg_maestro].sort_values('date')[cols_reg_maestro]\
        .drop_duplicates(subset=['numcta'],keep='last')
    print(f'Cantidad total de Cuentas: {df_instalacion_cuenta.numcta.nunique()}')
    print(f'Cantidad total de Instalaciones: {df_instalacion_cuenta.instalacion.nunique()}')
    df_instalacion_cuenta.to_parquet(nombre_path_historico_instalacion_cuenta)
    print('---Finalizado: Importación de datos nuevos de consumos---')
    return df_historico, df_instalacion_cuenta

def transform_raw_input_inspecciones(df_fraudes_agregar):
    '''
    CONVIERTE EL ARCHIVO EXCEL DE ENTRADA EN FORMATO DE EPMAPS AL FORMATO DE DATASET DE INFRACCIONES PARA UNIR CON LOS CONSUMOS
    '''
    print('---Transformación de datos nuevos de inspecciones---')
    
    df_fraudes_agregar.columns = df_fraudes_agregar.columns.str.lower()
    df_fraudes_agregar = df_fraudes_agregar[['fecha','instalacion','inspecciones realizadas','notificaciones']]
    df_fraudes_agregar = df_fraudes_agregar[ df_fraudes_agregar['inspecciones realizadas'] == 1 ]
    df_fraudes_agregar['instalacion'] = pd.to_numeric(df_fraudes_agregar['instalacion'], errors='coerce')
    df_fraudes_agregar['instalacion'] = df_fraudes_agregar['instalacion'].astype(int)
    df_fraudes_agregar['instalacion'] = df_fraudes_agregar['instalacion'].astype('string[pyarrow]')
    df_fraudes_agregar['fecha'] = df_fraudes_agregar['fecha'].astype(str)
    df_fraudes_agregar['fecha'] = df_fraudes_agregar['fecha'].str[:4] + '-' + df_fraudes_agregar['fecha'].str[5:7] + '-' + df_fraudes_agregar['fecha'].str[8:10]
    df_fraudes_agregar['year'] = df_fraudes_agregar['fecha'].str[:4]
    df_fraudes_agregar['mes'] = df_fraudes_agregar['fecha'].str[5:7]
    df_fraudes_agregar['dia'] = df_fraudes_agregar['fecha'].str[8:10]
    df_fraudes_agregar['date'] = df_fraudes_agregar['year']+'/'+df_fraudes_agregar['mes']
    df_fraudes_agregar['mesmulta'] = df_fraudes_agregar['year']+'/'+df_fraudes_agregar['mes']
    df_fraudes_agregar.date = pd.to_datetime(df_fraudes_agregar.date)
    df_fraudes_agregar.rename(columns = {'fecha':'fecha_multa','notificaciones':'is_fraud'}, inplace = True)
    df_fraudes_agregar['is_fraud'] = df_fraudes_agregar['is_fraud'].astype(int)
    
    # Vinculamos el número de instalación con el número de cuenta para agregar al registro de inspecciones
    df_map_ins_cuenta = pd.read_parquet(nombre_path_historico_instalacion_cuenta)
    df_fraudes_agregar = df_fraudes_agregar.merge(df_map_ins_cuenta[['numcta', 'instalacion']], on=['instalacion'], how='left').drop_duplicates(subset='instalacion')
    df_fraudes_agregar = df_fraudes_agregar[['numcta','instalacion','fecha_multa','year','mes','dia','date','mesmulta','is_fraud']]
    
    print(f'Dimensiones raw input inspecciones final: {df_fraudes_agregar.shape}')
    print('---Finalizada transformación de datos nuevos de inspecciones---')
    
    return df_fraudes_agregar

def concatenar_nuevos_datos_inspecciones():
    '''
    AÑADE NUEVOS DATOS DE INSPECCIONES AL HISTORICO COMPLETO: DEVUELVE DF_INSPECCIONES CON LOS ULTIMOS ARCHIVOS INCORPORADOS AL DATASET
        Busca archivos excel o csv en la carpeta de inspecciones y los agrega al histórico de inspecciones
    '''
    print('---Importación de datos nuevos de inspecciones---')
    if os.path.exists(nombre_path_historico_inspecciones):
        df_historico_inspecciones = pd.read_parquet(nombre_path_historico_inspecciones)
    else:
        df_historico_inspecciones = pd.DataFrame()
    print(f'Tamaño original df datos histórico inspecciones: {df_historico_inspecciones.shape}')
    contenido = os.listdir(nombre_carpeta_inspecciones)
    df_inspecciones = pd.DataFrame()
    for elemento in contenido:
        ruta_completa = os.path.join(nombre_carpeta_inspecciones, elemento)
        if os.path.isfile(ruta_completa):  # miramos si es fichero
            agregar = False
            if elemento[-4:].lower()=='.csv':
                df_in = pd.read_csv(ruta_completa, sep=';', 
                                   encoding='latin-1', warn_bad_lines=True, error_bad_lines=False,
                                   lineterminator='\n')
                agregar = True
            elif elemento[-5:].lower()=='.xlsx':
                df_in = pd.read_excel(ruta_completa, engine='openpyxl')
                agregar = True
            if agregar:
                df_in = transform_raw_input_inspecciones(df_in)
                #df_consumo = df_consumo.append(pd.concat([df_in], axis=0, ignore_index=True), ignore_index=True)
                df_inspecciones = pd.concat([df_inspecciones, df_in], ignore_index=True)
                print(f'Añadido: {ruta_completa}')
                #MOVEMOS ARCHIVO A CARPETA PROCESADOS
                os.rename(os.path.join(nombre_carpeta_inspecciones, elemento), os.path.join(nombre_carpeta_procesados, elemento))
    print(f'Tamaño datos nuevos datos de inspecciones: {df_inspecciones.shape}')
    df_historico_inspecciones = pd.concat([df_historico_inspecciones, df_inspecciones])
    df_historico_inspecciones.to_parquet(nombre_path_historico_inspecciones)
    print(f'Tamaño final df datos histórico inspecciones: {df_historico_inspecciones.shape}')
    print('---Finalizado: Importación de datos nuevos de inspecciones---')
    return df_historico_inspecciones

def series_de_consumo_etiquetadas(df_fraudes_completo_etiquetado = None):
    print('---Generación de series de consumo etiquetadas---')
    ## NO CARGAMOS REGISTROS ANTERIORES PORQUE EL FUNCIONAMIENTO ES INCOMPATIBLE
    #if os.path.exists(nombre_path_historico_series_etiquetadas):
    #    df_wide_normal_and_fraud_historico = pd.read_parquet(nombre_path_historico_series_etiquetadas)
    #    print(f'Tamaño original df datos fraudes completo etiquetado: {df_fraudes_completo_etiquetado.shape}')
    #    df_fraudes_completo_etiquetado['date_str'] = df_fraudes_completo_etiquetado.date.astype(str)
    #    df_wide_normal_and_fraud_historico['date_str'] = df_wide_normal_and_fraud_historico.date_fiscalizacion.astype(str)
    #    solap = df_fraudes_completo_etiquetado.merge(df_wide_normal_and_fraud_historico[['instalacion','date_str']],on=['instalacion','date_str'],how='outer',indicator='merge_result')
    #    df_fraudes_completo_etiquetado = solap.query('merge_result == "left_only"').drop(columns=['date_str','merge_result'])
    #else:
    #    df_wide_normal_and_fraud_historico = pd.DataFrame()
    
    #Añadimos 12 meses a la mínima fecha de consumo para que puedan computarse series de 12 consumos completas
    if df_fraudes_completo_etiquetado is None:
        df_fraudes_completo_etiquetado = pd.read_parquet(nombre_path_historico_datos_completos_etiquetados)
    fecha_minima_serie = df_fraudes_completo_etiquetado.date.min()+pd.DateOffset(months=12)
    #Tabla fraudes
    fecha_fraud_list = df_fraudes_completo_etiquetado[(df_fraudes_completo_etiquetado.is_fraud==1)&(df_fraudes_completo_etiquetado.date>= fecha_minima_serie)]['date'].astype(str).unique().tolist()
    list_df = []
    for fecha_fraud in tqdm(fecha_fraud_list, total=len(fecha_fraud_list)):
        df_etiquetado_fraud = df_fraudes_completo_etiquetado[df_fraudes_completo_etiquetado.date<=fecha_fraud].copy()
        ctas_fraud = df_etiquetado_fraud[(df_etiquetado_fraud.date==fecha_fraud)&(df_etiquetado_fraud.is_fraud==1)].instalacion.unique().tolist()
        df_etiquetado_fraud = df_etiquetado_fraud[df_etiquetado_fraud.instalacion.isin(ctas_fraud)]
        date_inicial = str(pd.to_datetime(fecha_fraud)- pd.DateOffset(months = 12))
        df_etiquetado_fraud = df_etiquetado_fraud[df_etiquetado_fraud['date']>=date_inicial]
        # Obtenemos datos categóricos para la serie anterior al fraude, sin incluirlo
        df_previo = df_etiquetado_fraud[ df_etiquetado_fraud['date']<fecha_fraud ]
        df_cant_null = df_previo.groupby(['instalacion']).metodo_consumo.sum().reset_index(name='cant_consumo_est')
        df_cant_estado = df_previo.groupby(['instalacion']).cl_inst.value_counts().unstack().reset_index()
        df_cant_estado.rename(columns={'0.0':'cant_estado_0','1.0':'cant_estado_1','2.0':'cant_estado_2','3.0':'cant_estado_3','4.0':'cant_estado_4'},inplace=True)
        cols_cant_estado = ['instalacion','cant_estado_0','cant_estado_1','cant_estado_2','cant_estado_3','cant_estado_4']
        # PARA EVITAR SI NO EXISTE ALGUN ESTADO EN EL CONJUNTO DE INSTALACIONES
        for col in cols_cant_estado:
            if col not in df_cant_estado.columns:
                df_cant_estado[col] = np.nan
        df_cant_estado = df_cant_estado[cols_cant_estado] # REORDENAR COLS
        df_cant_estado.fillna(0, inplace=True)
        
        df_categoria = df_previo.groupby(['instalacion']).categoria.apply(list).reset_index()
        df_categoria["cant_categorias"] = df_categoria.apply(lambda x: len(list(set(x["categoria"]))) if isinstance(x["categoria"],list) else 0,axis=1)
        df_categoria["ult_categoria"] = df_categoria.apply(lambda x: x["categoria"][-1],axis=1)

        def cat_mas_freq(row):
            serie = pd.Series(row["categoria"])
            if not serie.value_counts().empty:
                return serie.value_counts().idxmax()
            else:
                return ''
        df_categoria["categ_mas_frecuente"] = df_categoria.apply(cat_mas_freq,axis=1)
        def cambios_categ(row):
            serie = pd.Series(row["categoria"])
            if not serie.value_counts().empty:
                return serie.ne(serie.shift().bfill()).astype(int).sum()
            else:
                return 0
        df_categoria["cambios_categoria"] = df_categoria.apply(cambios_categ,axis=1)

        cols_ant = [str(x)+'_anterior' for x in range(12,-1,-1)]
        df_etiquetado_fraud = df_etiquetado_fraud.pivot_table(index=['instalacion'], columns=['date'] , values='consumo_medido')
        df_etiquetado_fraud.columns = cols_ant
        df_etiquetado_fraud['date_fiscalizacion'] = fecha_fraud
        df_etiquetado_fraud.reset_index(inplace=True)
        df_etiquetado_fraud = df_etiquetado_fraud.merge(df_cant_null, on='instalacion', how='left')
        df_etiquetado_fraud = df_etiquetado_fraud.merge(df_cant_estado, on='instalacion', how='left')
        df_etiquetado_fraud['mes']=mes=int(fecha_fraud[5:7])
        df_etiquetado_fraud['bimestre']=(mes-1)//2+1
        df_etiquetado_fraud['trimestre']=(mes-1)//3+1
        df_etiquetado_fraud['cuatrimestre']=(mes-1)//4+1
        df_etiquetado_fraud['semestre']=(mes-1)//6+1
        df_etiquetado_fraud = df_etiquetado_fraud.merge(df_categoria.drop(columns=['categoria']), on='instalacion', how='left')
        list_df.append(df_etiquetado_fraud)
    df_fraud_wide = pd.concat(list_df)

    #Datos de no fraude
    fecha_normal_list = df_fraudes_completo_etiquetado[(df_fraudes_completo_etiquetado.is_fraud==0)&(df_fraudes_completo_etiquetado.date>= fecha_minima_serie)]['date'].astype(str).unique().tolist()
    list_df_normal = []
    for fecha_normal in tqdm(fecha_normal_list, total=len(fecha_normal_list)):
        df_etiquetado_normal = df_fraudes_completo_etiquetado[df_fraudes_completo_etiquetado.date<=fecha_normal].copy()
        ctas_normal = df_etiquetado_normal[(df_etiquetado_normal.date==fecha_normal)&(df_etiquetado_normal.is_fraud==0)].instalacion.unique().tolist()
        df_etiquetado_normal = df_etiquetado_normal[df_etiquetado_normal.instalacion.isin(ctas_normal)]
        date_inicial = str(pd.to_datetime(fecha_normal)- pd.DateOffset(months = 12))
        df_etiquetado_normal = df_etiquetado_normal[df_etiquetado_normal['date']>=date_inicial]

        # Obtenemos datos categóricos para la serie anterior al control, sin incluirlo
        df_previo = df_etiquetado_normal[ df_etiquetado_normal['date']<fecha_normal ]

        df_cant_null = df_previo.groupby(['instalacion']).metodo_consumo.sum().reset_index(name='cant_consumo_est')
        df_cant_estado = df_previo.groupby(['instalacion']).cl_inst.value_counts().unstack().reset_index()
        df_cant_estado.rename(columns={'0.0':'cant_estado_0','1.0':'cant_estado_1','2.0':'cant_estado_2','3.0':'cant_estado_3','4.0':'cant_estado_4'},inplace=True)
        cols_cant_estado = ['instalacion','cant_estado_0','cant_estado_1','cant_estado_2','cant_estado_3','cant_estado_4']
        # PARA EVITAR SI NO EXISTE ALGUN ESTADO EN EL CONJUNTO DE INSTALACIONES
        for col in cols_cant_estado:
            if col not in df_cant_estado.columns:
                df_cant_estado[col] = np.nan
        df_cant_estado = df_cant_estado[cols_cant_estado] # REORDENAR COLS
        
        df_cant_estado.columns = ['instalacion', 'cant_estado_0', 'cant_estado_1', 'cant_estado_2', 'cant_estado_3', 'cant_estado_4']
        df_cant_estado.fillna(0, inplace=True)
        df_categoria = df_previo.groupby(['instalacion']).categoria.apply(list).reset_index()
        df_categoria["cant_categorias"] = df_categoria.apply(lambda x: len(list(set(x["categoria"]))) if isinstance(x["categoria"],list) else 0,axis=1)
        df_categoria["ult_categoria"] = df_categoria.apply(lambda x: x["categoria"][-1],axis=1)

        def cat_mas_freq(row):
            serie = pd.Series(row["categoria"])
            if not serie.value_counts().empty:
                return serie.value_counts().idxmax()
            else:
                return ''
        df_categoria["categ_mas_frecuente"] = df_categoria.apply(cat_mas_freq,axis=1)
        def cambios_categ(row):
            serie = pd.Series(row["categoria"])
            if not serie.value_counts().empty:
                return serie.ne(serie.shift().bfill()).astype(int).sum()
            else:
                return 0
        df_categoria["cambios_categoria"] = df_categoria.apply(cambios_categ,axis=1)

        cols_ant = [str(x)+'_anterior' for x in range(12,-1,-1)]
        df_etiquetado_normal = df_etiquetado_normal.pivot_table(index=['instalacion'], columns=['date'] , values='consumo_medido')
        df_etiquetado_normal.columns = cols_ant
        df_etiquetado_normal['date_fiscalizacion'] = fecha_normal
        df_etiquetado_normal.reset_index(inplace=True)
        df_etiquetado_normal = df_etiquetado_normal.merge(df_cant_null, on='instalacion', how='left')
        df_etiquetado_normal = df_etiquetado_normal.merge(df_cant_estado, on='instalacion', how='left')

        df_etiquetado_normal['mes']=mes=int(fecha_normal[5:7])
        df_etiquetado_normal['bimestre']=(mes-1)//2+1
        df_etiquetado_normal['trimestre']=(mes-1)//3+1
        df_etiquetado_normal['cuatrimestre']=(mes-1)//4+1
        df_etiquetado_normal['semestre']=(mes-1)//6+1

        df_etiquetado_normal = df_etiquetado_normal.merge(df_categoria.drop(columns=['categoria']), on='instalacion', how='left')

        list_df_normal.append(df_etiquetado_normal)

    df_normal_wide = pd.concat(list_df_normal)
    #Sacar fraudes
    df_normal_wide=df_normal_wide[~df_normal_wide.instalacion.isin(df_fraud_wide.instalacion.unique())]

    #Conservar solamente registros normales de las 
    #fechas de fiscalización de los fraudes
    fechas_fraude_unicas = df_fraud_wide.date_fiscalizacion.unique().tolist()
    df_normal_wide = df_normal_wide[df_normal_wide.date_fiscalizacion.isin(fechas_fraude_unicas)]
    df_normal_wide['is_fraud']=0
    df_fraud_wide['is_fraud']=1
    #Otros filtros
    df_fraud_wide = df_fraud_wide[df_fraud_wide.cant_consumo_est<=6]
    df_normal_wide = df_normal_wide[df_normal_wide.cant_consumo_est==0]
    df_wide_normal_and_fraud = pd.concat([df_normal_wide,df_fraud_wide]).reset_index(drop=True)
    #La siguiente línea duplica los registros que tienen más de una categoría.. Hay que controlar eso
    #df_wide_normal_and_fraud = df_wide_normal_and_fraud.merge(df_fraudes_completo_etiquetado[['instalacion','categoria']].drop_duplicates(), on='instalacion')
    df_wide_normal_and_fraud['id'] = list(range(len(df_wide_normal_and_fraud)))
    
    #if len(df_wide_normal_and_fraud_historico)>0:
    #    df_wide_normal_and_fraud = pd.concat([df_wide_normal_and_fraud_historico, df_wide_normal_and_fraud])
    #    df_wide_normal_and_fraud['id'] = list(range(len(df_wide_normal_and_fraud)))
    df_wide_normal_and_fraud.to_parquet(nombre_path_historico_series_etiquetadas)
    print(f'Tamaño final series etiquetadas: {df_wide_normal_and_fraud.shape}')
    print('---Finalizado: Generación de series de consumo etiquetadas---')
    return df_wide_normal_and_fraud

def cleaning_feature_engineering(df_wide_completo = None):
    print('---Generación de features de series de consumo etiquetadas---')
    if df_wide_completo is None:
        df_wide_completo = pd.read_parquet(nombre_path_historico_series_etiquetadas)
    #PREGUNTAMOS SI EXISTE HISTORICO, PARA NO RECALCULAR FEATURES YA COMPUTADAS
    if os.path.exists(nombre_path_historico_series_features):
        #QUITAMOS SERIES EXISTENTES DE DF_WIDE_COMPLETO
        df_series_features_existentes = pd.read_parquet(nombre_path_historico_series_features)
        print(f'-- Usando datos históricos de feature eng: {df_series_features_existentes.shape} {nombre_path_historico_series_features}')
        df_series_features_existentes = df_series_features_existentes[['instalacion','date_fiscalizacion']]
        df_wide_completo = df_series_features_existentes.merge(df_wide_completo, on=['instalacion','date_fiscalizacion'],how='outer',indicator=True)
        df_wide_completo = df_wide_completo.query('_merge=="right_only"').drop(columns=['_merge'])
        #print(f'dimensiones luego de preprocesar historico: {df_wide_completo.shape}')
    
    df = df_wide_completo.rename(columns={'is_fraud':'target','id':'index'})
    print(f'Tamaño datos wide a agregar (nuevos datos no catalogados antes): {df.shape}')
    variables_consumo = [x for x in df.columns if '_anterior' in x and x!='0_anterior']#sin 0_anterior (cons actual)#
    df.loc[:,['index']+variables_consumo] = llenar_val_vacios_ciclo(df.loc[:,['index']+variables_consumo], 12)
    df.loc[:,['index']+variables_consumo] = df.loc[:,['index']+variables_consumo].fillna(0)
    pipe_feature_engeniering_consumo = Pipeline(
        [
            ("tsfel vars", TsfelVars(features_names_path=None,num_periodos= 12)),
            ("add vars3",  ExtraVars(num_periodos=3)),
            ("add vars6",  ExtraVars(num_periodos=6)),
            ("add vars12", ExtraVars(num_periodos=12)),
        ]
            )
    # CARGAMOS HISTÓRICO SI EXISTE
    if os.path.exists(nombre_path_historico_series_features):
        df_series_features_existentes = pd.read_parquet(nombre_path_historico_series_features)
    else:
        df_series_features_existentes = pd.DataFrame()
    # SI HAY NUEVOS REGS SIN FEATURES CALCULAR, SINO ASIGNAR VACÍO
    if df.shape[0] > 0:
        df_features = pipe_feature_engeniering_consumo.fit_transform(
            df[['index']+variables_consumo])
        cols_fs = ['index']+df_features.columns.tolist()[13:]
        df_completo = df.merge(df_features[cols_fs], how='inner', on=['index'], indicator=False)
    else:
        df_completo = pd.DataFrame()
    guardar = True # CONTROL PARA SABER SI HAY QUE ACTUALIZAR ARCHIVO
    if len(df_series_features_existentes) > 0:
        if len(df_completo) > 0:
            df_completo = pd.concat([df_series_features_existentes, df_completo])
        else:
            guardar = False
            df_completo = df_series_features_existentes
    elif len(df_completo) == 0:
        guardar = False
    if guardar:
        df_completo.to_parquet(nombre_path_historico_series_features)
    print(f'Tamaño final series con features: {df_completo.shape}')
    print('---Finalizado: Generación de features de series de consumo etiquetadas---')
    return df_completo

def feature_selection_dataset_entrenamiento(df = None):
    if df is None:
        df = pd.read_parquet(nombre_path_historico_series_features)
    print('---Selección de Features y Armado de Dataset de Entrenamiento---')
    #df = pd.read_parquet(nombre_path_historico_series_features)
    X = df.drop('target', axis=1)
    y = df['target']
    print(f'Tamaño original X: {X.shape}, Tamaño original y: {y.shape}')
    variables_consumo = [x for x in X.columns if '_anterior' in x]# and x!='0_anterior'
    variables_categoricas = ['cant_consumo_est', 'cant_estado_0', 'cant_estado_1', 'cant_estado_2', 'cant_estado_3', 'cant_estado_4', 'mes', 'bimestre', 'trimestre', 'cuatrimestre', 'semestre', 'cant_categorias', 'ult_categoria', 'categ_mas_frecuente', 'cambios_categoria']
    cols_excluir = ['index', 'instalacion', 'date_fiscalizacion'] + variables_consumo + variables_categoricas
    cols_for_feature_sel = [x for x in X.columns if x not in cols_excluir]
    select_by_constant = feature_selection_by_constant(X, y, cols_for_feature_sel, th=0.99)
    select_by_boruta = feature_selection_by_boruta(X[select_by_constant].fillna(0), y, N=5)
    print(f" # variables seleccionadas por Boruta : {len(select_by_boruta)}")
    feat_selec = pd.DataFrame({'constant':[select_by_constant],'boruta':[select_by_boruta]})
    feat_selec.to_parquet(nombre_path_historico_features_selected)
    X.to_parquet(nombre_path_historico_X)
    y_guardar = pd.DataFrame(y)
    y_guardar.to_parquet(nombre_path_historico_Y)
    print(f'Tamaño final X: {X.shape}, Tamaño y: {y.shape}')
    print('---Finalizado: Selección de Features y Armado de Dataset de Entrenamiento---')
    return X, y, feat_selec

def etiquetar_consumos_inspecciones_train(df = None, df_fraudes = None):
    '''
    CRUZAR DATOS DE CONSUMO CON INSPECCIONES REALIZADAS, PARA ETIQUETAR LOS CONSUMOS EN LAS CUENTAS INSPECCIONADAS
    GENERA DATASET DE CONSUMOS, CON SUS ETIQUETAS DE FRAUDE O NO FRAUDE (is_fraud)
    '''
    print('---Generación de datos de consumo etiquetados---')
    if df is None:
        df = pd.read_parquet(nombre_path_historico)
    if df_fraudes is None:
        df_fraudes = pd.read_parquet(nombre_path_historico_inspecciones)
    print(f'Tamaño original df datos histórico: {df.shape}')
    print(f'Tamaño original df datos histórico fraudes: {df_fraudes.shape}')
    list_inst_dataset = df_fraudes.instalacion.unique().tolist()
    df_fraudes_completo = df[ df['instalacion'].isin(list_inst_dataset) ].drop_duplicates(subset=['instalacion','mesfac'])

    cols = ['instalacion','mesmulta','is_fraud']
    df_fraudes_completo_etiquetado = df_fraudes_completo.merge(df_fraudes[cols], how = 'left', 
               left_on = ['instalacion','mesfac'], right_on = ['instalacion','mesmulta'], indicator = True)
    #df_fraudes_completo_etiquetado['is_fraud'] = np.where(df_fraudes_completo_etiquetado['codigo_novedad'].isnull(),0,1)
    # codigo_novedad>0 -> nunique encontró valores no nulos (NaN/None)
    facts_duplicados_fraudulentos = df_fraudes_completo_etiquetado.groupby(['instalacion', 'mesfac'])\
        .agg({'is_fraud':'sum', 'mesmulta':'count'}).reset_index()\
        .query('mesmulta>1 & is_fraud>0')[['instalacion','mesfac']].values.tolist()
    # PARA DUPLICADOS: SI ALGUN REGISTRO TIENE MULTA MARCO TODOS LOS DUPLICADOS COMO MULTA
    for f in facts_duplicados_fraudulentos:
        df_fraudes_completo_etiquetado.loc[ (df_fraudes_completo_etiquetado['instalacion']==f[0]) & (df_fraudes_completo_etiquetado['mesfac']==f[1]),'is_fraud'] = 1
    df_fraudes_completo_etiquetado = df_fraudes_completo_etiquetado.drop_duplicates(subset=['instalacion','mesfac'])
    df_fraudes_completo_etiquetado['metodo_consumo'] = np.where(df_fraudes_completo_etiquetado.metodo_consumo=='Real',0,1)
    df_fraudes_completo_etiquetado.to_parquet(nombre_path_historico_datos_completos_etiquetados)
    print(f'Tamaño final df datos histórico completos etiquetados: {df_fraudes_completo_etiquetado.shape}')
    print('---Finalizado: Generación de datos de consumo etiquetados---')
    return df_fraudes_completo_etiquetado

def series_de_consumo_mes_especifico(df_consumo=None,mes=False):
    if df_consumo is None:
        df_consumo = pd.read_parquet(nombre_path_historico)
    mes = df_consumo.date.max().strftime("%Y/%m") if not mes else mes
    print(f'---Generación de series de consumo para predicción de mes específico: {mes}---')
    mensajeError = 'Ingrese un mes correcto: El formato es AÑO/MES, separados por /, por ejemplo 2022/12'
    assert len(mes.split("/"))==2, mensajeError
    assert mes.split("/")[0].isnumeric() & mes.split("/")[1].isnumeric(), mensajeError
    # FILTRAR CONSUMOS DE LAS FECHAS DENTRO DEL INTERVALO E INSTALACIONES DEL PERÍODO
    print('Filtrado de datos del período')
    mesDate = datetime.strptime(mes, '%Y/%m')
    mesInicialDate = mesDate - pd.DateOffset(months = 11)
    df_filtrado = df_consumo[ (df_consumo['date'] >= mesInicialDate) & (df_consumo['date'] <= mesDate) ]
    ctas_mes = df_filtrado[ df_filtrado.date==mesDate ].instalacion.unique().tolist()
    df_filtrado = df_filtrado[df_filtrado.instalacion.isin(ctas_mes)]
    print('Consumos reales/estimados: Contabilizando cantidad de estimaciones')
    df_filtrado.metodo_consumo = np.where(df_filtrado.metodo_consumo=='Real',0,1)
    df_cant_null = df_filtrado.groupby(['instalacion']).metodo_consumo.sum().reset_index(name='cant_consumo_est')
    df_filtrado.cl_inst = df_filtrado.cl_inst.map({'0.0':0,'1.0':1,'2.0':2,'3.0':3,'4.0':4}).fillna(0).astype(int)
    print(f'Estados: Contabilizando por tipo')
    df_cant_estado = df_filtrado.groupby(['instalacion']).cl_inst.value_counts().unstack().reset_index()
    df_cant_estado.rename(columns={0:'cant_estado_0',1:'cant_estado_1',2:'cant_estado_2',3:'cant_estado_3',4:'cant_estado_4'},inplace=True)
    cols_cant_estado = ['instalacion','cant_estado_0','cant_estado_1','cant_estado_2','cant_estado_3','cant_estado_4']
    # PARA EVITAR SI NO EXISTE ALGUN ESTADO EN EL CONJUNTO DE INSTALACIONES
    for col in cols_cant_estado:
        if col not in df_cant_estado.columns:
            df_cant_estado[col] = np.nan
    df_cant_estado = df_cant_estado[cols_cant_estado]#.astype(int) # REORDENAR COLS
    df_cant_estado.fillna(0, inplace=True)
    print('Categorías: Contabilizando categorías del período y última categoría por instalación')
    df_categoria = df_filtrado.groupby(['instalacion']).categoria.apply(list).reset_index()
    df_categoria["cant_categorias"] = df_categoria.apply(lambda x: len(list(set(x["categoria"]))) if isinstance(x["categoria"],list) else 0,axis=1)
    df_categoria["ult_categoria"] = df_categoria.apply(lambda x: x["categoria"][-1],axis=1)
    print('Categorías: Identificando categorias mas frecuentes del período por instalación')
    def cat_mas_freq(row):
        serie = pd.Series(row["categoria"])
        if not serie.value_counts().empty:
            return serie.value_counts().idxmax()
        else:
            return ''
    df_categoria["categ_mas_frecuente"] = df_categoria.apply(cat_mas_freq,axis=1)
    print('Categorías: Contabilizando cambios')
    def cambios_categ(row):
        serie = pd.Series(row["categoria"])
        if not serie.value_counts().empty:
            return serie.ne(serie.shift().bfill()).astype(int).sum()
        else:
            return 0
    df_categoria["cambios_categoria"] = df_categoria.apply(cambios_categ,axis=1)
    print('Iniciando armado de series')
    cols_ant = [str(x)+'_anterior' for x in range(12,0,-1)]
    
    df_series_final = df_filtrado.pivot_table(index=['instalacion'], columns=['date'] , values='consumo_medido')
    print('Iniciando unión de series y características')
    df_series_final.columns = cols_ant
    #df_series_final['date_fiscalizacion'] = fecha_fraud
    df_series_final.reset_index(inplace=True)
    df_series_final = df_series_final.merge(df_cant_null, on='instalacion')
    df_series_final = df_series_final.merge(df_cant_estado, on='instalacion')
    df_series_final['mes']=mesSerie=int(mes[5:7])
    df_series_final['bimestre']=(mesSerie-1)//2+1
    df_series_final['trimestre']=(mesSerie-1)//3+1
    df_series_final['cuatrimestre']=(mesSerie-1)//4+1
    df_series_final['semestre']=(mesSerie-1)//6+1
    df_categoria.rename(columns={'categoria':'lista_categorias'})
    df_series_final = df_series_final.merge(df_categoria, on='instalacion')
    
    #AGREGAMOS DATO DE RAMO DEL ÚLTIMO CONSUMO
    df_ramo = df_filtrado[['instalacion','date','ramo']].sort_values('date').drop_duplicates(subset=['instalacion'],keep='last').drop(columns=['date'])
    df_series_final = df_series_final.merge(df_ramo, on='instalacion', how='left')
    
    #AGREGAMOS MES
    df_series_final['mesfac'] = mes
    
    df_series_final.to_parquet(nombre_path_historico_series_mes)
    print('----Finalizado armado de series del mes----')
    return df_series_final

def cleaning_feature_engineering_mes_especifico(df_series_mes=None,num_bloques=30):
    print('---Generación de features de series de consumo de mes específico---')
    if df_series_mes is None:
        df_series_mes = pd.read_parquet(nombre_path_historico_series_mes)
    variables_consumo = [x for x in df_series_mes.columns if '_anterior' in x]
    df_series_mes.loc[:,variables_consumo] = llenar_val_vacios_ciclo(df_series_mes.loc[:,variables_consumo], 12)
    df_series_mes.loc[:,variables_consumo] = df_series_mes.loc[:,variables_consumo].fillna(0)
    df_series_mes['index'] = list(range(len(df_series_mes)))
    pipe_feature_engeniering_consumo = Pipeline(
        [
            ("tsfel vars", TsfelVars(features_names_path=None,num_periodos= 12)),
            ("add vars3",  ExtraVars(num_periodos=3)),
            ("add vars6",  ExtraVars(num_periodos=6)),
            ("add vars12", ExtraVars(num_periodos=12)),
        ]
            )
    
    batch_size = len(df_series_mes) // num_bloques
    
    # Ver si existe archivo incremental de ejecuciones previas fallidas
    if os.path.exists(nombre_path_historico_series_features_mes_parcial):
        # Determinar último batch procesado
        table = pd.read_parquet(nombre_path_historico_series_features_mes_parcial)
        first_batch = len(table) // batch_size
        print(f'Continuamos desde batch {first_batch}...')
    else:
        # Iniciamos desde el principio el procesamiento si no existe archivo parcial
        print('Iniciando proceso de extracción de features desde el principio...')
        table = pd.DataFrame()
        first_batch = 0
    for i in range(first_batch, num_bloques):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size if i < num_bloques - 1 else len(df_series_mes)

        batch_data = df_series_mes.iloc[start_idx:end_idx]
        processed_data = pipe_feature_engeniering_consumo.fit_transform(batch_data[['index']+variables_consumo])
        cols_fs = ['index']+processed_data.columns.tolist()[len(variables_consumo)+1:]
        df_completo = batch_data.merge(processed_data[cols_fs], how='inner', on=['index'], indicator=False)

        # Write the processed data to the Parquet file
        if i == 0:
            table = df_completo.copy()
        else:
            table = pd.read_parquet(nombre_path_historico_series_features_mes_parcial)
            #os.rename(nombre_path_historico_series_features_mes_parcial, nombre_path_historico_series_features_mes_parcial+'2')
            table = pd.concat([table,df_completo])
            table['index'] = list(range(len(table)))
        table.to_parquet(nombre_path_historico_series_features_mes_parcial)
        df_completo_final = table
        print(f'Batch {i} procesado y agregado a {nombre_path_historico_series_features_mes_parcial}. Dimensiones: {table.shape}')
    
    # Renombramos el archivo parcial a final al completar el proceso
    os.rename(nombre_path_historico_series_features_mes_parcial, nombre_path_historico_series_features_mes)
    print(f'Dimensiones finales del dataset de features para el mes: {df_completo_final.shape}')
    print('---Finalizado: Generación de features de series de consumo etiquetadas mes específico---')
    return df_completo_final

#### CODIGOS VIEJOS ####

def etiquetar_consumos_inspecciones_train_formato_viejo(df, df_fraudes):
    '''
    CRUZAR DATOS DE CONSUMO CON INSPECCIONES REALIZADAS, PARA ETIQUETAR LOS CONSUMOS EN LAS CUENTAS INSPECCIONADAS
    GENERA DATASET DE CONSUMOS, CON SUS ETIQUETAS DE FRAUDE O NO FRAUDE (is_fraud)
    '''
    print('---Generación de datos de consumo etiquetados---')
    print(f'Tamaño original df datos histórico: {df.shape}')
    print(f'Tamaño original df datos histórico fraudes: {df_fraudes.shape}')
    list_inst_dataset = df_fraudes.instalacion.unique().tolist()
    df_fraudes_completo = df[ df['instalacion'].isin(list_inst_dataset) ].drop_duplicates(subset=['instalacion','mesfac'])
    cols = ['instalacion','mesmulta','codigo_novedad']
    df_fraudes_completo_etiquetado = df_fraudes_completo.merge(df_fraudes[cols], how = 'left', left_on = ['instalacion','mesfac'], 
            right_on = ['instalacion','mesmulta'], indicator = True)
    df_fraudes_completo_etiquetado['is_fraud'] = np.where(df_fraudes_completo_etiquetado['codigo_novedad'].isnull(),0,1)
    # codigo_novedad>0 -> nunique encontró valores no nulos (NaN/None)
    facts_duplicados_fraudulentos = df_fraudes_completo_etiquetado.groupby(['instalacion','mesfac']).agg({'codigo_novedad':'nunique','mesmulta':'count'})\
        .reset_index().query('mesmulta>1 & codigo_novedad>0')[['instalacion','mesfac']].values.tolist()
    # PARA DUPLICADOS: SI ALGUN REGISTRO TIENE MULTA MARCO TODOS LOS DUPLICADOS COMO MULTA
    for f in facts_duplicados_fraudulentos:
        df_fraudes_completo_etiquetado.loc[ (df_fraudes_completo_etiquetado['instalacion']==f[0]) & (df_fraudes_completo_etiquetado['mesfac']==f[1]),'is_fraud'] = 1
    df_fraudes_completo_etiquetado = df_fraudes_completo_etiquetado.drop_duplicates(subset=['instalacion','mesfac'])
    df_fraudes_completo_etiquetado['metodo_consumo'] = np.where(df_fraudes_completo_etiquetado.metodo_consumo=='Real',0,1)
    df_fraudes_completo_etiquetado.to_parquet(nombre_path_historico_datos_completos_etiquetados)
    print(f'Tamaño final df datos histórico completos etiquetados: {df_fraudes_completo_etiquetado.shape}')
    print('---Finalizado: Generación de datos de consumo etiquetados---')
    return df_fraudes_completo_etiquetado

def transform_raw_input_inspecciones_formato_viejo(df_fraudes):
    '''
    CONVIERTE EL ARCHIVO EXCEL DE ENTRADA EN FORMATO DE EPMAPS AL FORMATO DE DATASET DE INFRACCIONES PARA UNIR CON LOS CONSUMOS
    '''
    print('---Transformación de datos nuevos de inspecciones---')
    df_fraudes = df_fraudes[['Cuenta contrato','Instalación','Fe.inic.extrema','Fecha entrada','Fecha fin real','Fe.inicio real','Código novedad','Causa Generación']]
    df_fraudes.rename(columns={'Cuenta contrato':'numcta','Instalación':'instalacion','Fe.inic.extrema':'fecha_multa','Fecha entrada':'fecha_entrada'}, inplace=True)
    df_fraudes.rename(columns={'Fecha fin real':'fecha_fin_real','Fe.inicio real':'fecha_inicio_real','Código novedad':'codigo_novedad','Causa Generación':'causa_generacion'}, inplace=True)
    df_fraudes = df_fraudes[ (~df_fraudes.instalacion.isnull()) & (~df_fraudes.fecha_multa.isnull()) ]#~df_fraudes.instalacion.isnull()
    df_fraudes.instalacion = df_fraudes.instalacion.astype(int)
    df_fraudes.instalacion = df_fraudes.instalacion.astype(str)
    df_fraudes.fecha_multa = df_fraudes.fecha_multa.astype(str)

    df_fraudes['year'] = df_fraudes['fecha_multa'].str[:4]
    df_fraudes['mes'] = df_fraudes['fecha_multa'].str[5:7]
    df_fraudes['dia'] = df_fraudes['fecha_multa'].str[8:10]
    df_fraudes['date'] =  df_fraudes['year']+'/'+df_fraudes['mes']
    df_fraudes['mesmulta'] = df_fraudes['year']+'/'+df_fraudes['mes']
    df_fraudes.date = pd.to_datetime(df_fraudes.date)
    print(f'Dimensiones raw input inspecciones final: {df_fraudes.shape}')
    print('---Transformación de datos nuevos de inspecciones---')
    return df_fraudes

def transform_raw_input_inspecciones_formato_viejo_dos_archivos(df_inspecciones, df_multas, df_instalacion_cuenta):
    '''
    CONVIERTE LOS ARCHIVOS EXCEL DE ENTRADA EN FORMATO DE EPMAPS AL FORMATO DE DATASET DE INFRACCIONES
    PARA UNIR CON LOS CONSUMOS
    df_inspecciones: Datos de inspecciones realizadas
    df_multas: Datos de multas sobre los registros de inspecciones, para etiquetar
    df_instalacion_cuenta: Mapeo Instalación - Cuenta para cruzar los datos cuando se requiere
    '''
    
    # PROCESAMIENTO DE INSPECCIONES
    print('---Carga de datos de Inspecciones---')
    df_inspecciones = df_inspecciones[['Cta.contrato','Fe.contab.']]
    df_inspecciones.rename(columns={'Cta.contrato':'numcta','Fe.contab.':'fecha_multa'}, inplace=True)
    df_inspecciones = df_inspecciones[ (~df_inspecciones.numcta.isnull()) & (~df_inspecciones.fecha_multa.isnull()) ]
    df_inspecciones.numcta = df_inspecciones.numcta.astype(int)
    df_inspecciones.numcta = df_inspecciones.numcta.astype(str)
    df_inspecciones.fecha_multa=df_inspecciones.fecha_multa.astype(str)
    fecha = df_inspecciones['fecha_multa'].str.split('.', expand=True)
    df_inspecciones['year'] = fecha[2]
    df_inspecciones['mes'] = fecha[1].str.zfill(2)
    df_inspecciones['dia'] = fecha[0]
    df_inspecciones['date'] = df_inspecciones['year']+'/'+df_inspecciones['mes']
    df_inspecciones['mesmulta'] = df_inspecciones['year']+'/'+df_inspecciones['mes']
    df_inspecciones.date = pd.to_datetime(df_inspecciones.date)
    df_inspecciones['is_fraud'] = 0

    # PROCESAMOS MULTAS
    print('---Carga de datos de Multas---')
    df_multas.rename(columns=lambda x: x.strip().lower(), inplace=True)
    df_multas.rename(columns={'instalación':'instalacion', 'cuenta':'numcta'}, inplace=True)
    fecha = df_multas['fecha'].astype(str).str.split('-', expand=True)
    df_multas['year'] = fecha[0]
    df_multas['mes'] = fecha[1].str.zfill(2)
    df_multas['dia'] = fecha[2]
    df_multas['date'] = df_multas['year'] + '/' + df_multas['mes']
    df_multas['mesmulta'] = df_multas['year']+'/'+df_multas['mes']
    df_multas = df_multas[~df_multas['instalacion'].isnull()]
    df_multas.instalacion = df_multas.instalacion.astype(int)
    df_multas.instalacion = df_multas.instalacion.astype(str)

    # OBTENER NUMERO DE INSTALACION A PARTIR DEL ARCHIVO DE MAPEO CON CUENTA CONTRATO
    df_inspecciones_inst = df_inspecciones.merge(df_instalacion_cuenta, on=['numcta'], how='inner')
    df_inspecciones_etiquetado = df_inspecciones_inst.merge(df_multas[['instalacion','mesmulta']].drop_duplicates(), 
        on=['instalacion','mesmulta'], how='left', indicator=True)
    df_inspecciones_etiquetado.loc[ df_inspecciones_etiquetado._merge=='both','is_fraud'] = 1
    cols = ['numcta', 'instalacion', 'fecha_multa', 'year', 'mes', 'dia', 'date', 'mesmulta', 'is_fraud']
    df_inspecciones_etiquetado = df_inspecciones_etiquetado[cols]
    return df_inspecciones_etiquetado

def concatenar_nuevos_datos_inspecciones_formato_viejo():
    '''
    AÑADE NUEVOS DATOS DE INSPECCIONES AL HISTORICO COMPLETO: DEVUELVE DF_INSPECCIONES CON LOS ULTIMOS ARCHIVOS INCORPORADOS AL DATASET
        df_historico: DF completo al que se le agregarán los nuevos datos de consumo de los últimos meses
        Busca archivos nuevos en la carpeta de datos de consumo, y mueve esos archivos a la carpeta procesados
        Devuelve el df completo con los datos nuevos agregados
    '''
    print('---Importación de datos nuevos de inspecciones---')
    if os.path.exists(nombre_path_historico_inspecciones):
        df_historico_inspecciones = pd.read_parquet(nombre_path_historico_inspecciones)
    else:
        df_historico_inspecciones = pd.DataFrame()
    print(f'Tamaño original df datos histórico inspecciones: {df_historico_inspecciones.shape}')
    contenido = os.listdir(nombre_carpeta_inspecciones)
    df_inspecciones = pd.DataFrame()
    for elemento in contenido:
        ruta_completa = os.path.join(nombre_carpeta_inspecciones, elemento)
        if os.path.isfile(ruta_completa):  # miramos si es fichero
            agregar = False
            if elemento[-4:].lower()=='.csv':
                df_in = pd.read_csv(ruta_completa, sep=';', 
                                   encoding='latin-1', warn_bad_lines=True, error_bad_lines=False,
                                   lineterminator='\n')
                agregar = True
            elif elemento[-5:].lower()=='.xlsx':
                df_in = pd.read_excel(ruta_completa, engine='openpyxl')
                agregar = True
            if agregar:
                df_in = transform_raw_input_inspecciones_formato_viejo(df_in)
                #df_consumo = df_consumo.append(pd.concat([df_in], axis=0, ignore_index=True), ignore_index=True)
                df_inspecciones = pd.concat([df_inspecciones, df_in], ignore_index=True)
                print(f'Añadido: {ruta_completa}')
                #MOVEMOS ARCHIVO A CARPETA PROCESADOS
                os.rename(os.path.join(nombre_carpeta_inspecciones, elemento), os.path.join(nombre_carpeta_procesados, elemento))
    print(f'Tamaño datos nuevos datos de inspecciones: {df_inspecciones.shape}')
    df_historico_inspecciones = pd.concat([df_historico_inspecciones, df_inspecciones])
    df_historico_inspecciones.to_parquet(nombre_path_historico_inspecciones)
    print(f'Tamaño final df datos histórico inspecciones: {df_historico_inspecciones.shape}')
    print('---Finalizado: Importación de datos nuevos de inspecciones---')
    return df_historico_inspecciones

def concatenar_nuevos_datos_inspecciones_formato_viejo_dos_archivos():
    '''
    AÑADE NUEVOS DATOS DE INSPECCIONES AL HISTORICO COMPLETO: DEVUELVE DF_INSPECCIONES CON LOS ULTIMOS ARCHIVOS 
    INCORPORADOS AL DATASET
    '''
    print('---Importación de datos nuevos de inspecciones---')
    if os.path.exists(nombre_path_historico_inspecciones):
        df_historico_inspecciones = pd.read_parquet(nombre_path_historico_inspecciones)
    else:
        df_historico_inspecciones = pd.DataFrame()
    print(f'Tamaño original df datos histórico inspecciones: {df_historico_inspecciones.shape}')
    contenido = os.listdir(nombre_carpeta_inspecciones)
    df_inspecciones = pd.DataFrame()
    df_multas = pd.DataFrame()
    list_procesados = []
    for elemento in contenido:
        ruta_completa = os.path.join(nombre_carpeta_inspecciones, elemento)
        if os.path.isfile(ruta_completa):  # miramos si es fichero
            agregar = False
            if elemento[-4:].lower()=='.csv':
                df_in = pd.read_csv(ruta_completa, sep=';', 
                                   encoding='latin-1', warn_bad_lines=True, error_bad_lines=False,
                                   lineterminator='\n')
                agregar = True
            elif elemento[-5:].lower()=='.xlsx':
                df_in = pd.read_excel(ruta_completa, engine='openpyxl')
                agregar = True
            if agregar:
                if elemento[:12].lower() == 'inspecciones':
                    print(f'Procesando Inspecciones: {elemento}')
                    df_inspecciones = pd.concat([df_inspecciones, df_in])
                elif elemento[:6].lower() == 'multas':
                    print(f'Procesando Multas: {elemento}')
                    df_multas = pd.concat([df_multas, df_in])
                    #df_consumo = df_consumo.append(pd.concat([df_in], axis=0, ignore_index=True), ignore_index=True)
                else:
                    agregar = False
            if agregar:
                print(f'Añadido: {ruta_completa}')
                #AÑADIMOS A LISTA PARA MOVER A CARPETA PROCESADOS
                list_procesados.append(elemento)
    df_map_inst_cuenta = pd.read_parquet(nombre_path_historico_instalacion_cuenta)
    if len(df_inspecciones)>0 and len(df_multas)>0:
        df_inspecciones = transform_raw_input_inspecciones_formato_viejo_dos_archivos(df_inspecciones, df_multas, df_map_inst_cuenta)
        print(f'Tamaño datos nuevos de inspecciones: {df_inspecciones.shape}')

        # TRANSFORMAR HISTORICO SI TIENE LA COLUMNA CODIGO_NOVEDAD
        print('Transformar archivo de inspecciones formato viejo a nuevo (quitar codigo_novedad y agregar is_fraud)')
        if 'codigo_novedad' in df_historico_inspecciones.columns:
            df_historico_inspecciones['is_fraud'] = np.where(df_historico_inspecciones['codigo_novedad'].isnull(),0,1)
            cols = ['numcta', 'instalacion', 'fecha_multa', 'year', 'mes', 'dia', 'date', 'mesmulta', 'is_fraud']
            df_historico_inspecciones = df_historico_inspecciones[cols]
        df_historico_inspecciones['numcta'] = df_historico_inspecciones['numcta'].fillna(0).astype(int).astype('string[pyarrow]')
        df_historico_inspecciones = pd.concat([df_historico_inspecciones, df_inspecciones])
        for elemento in list_procesados:
            os.rename(os.path.join(nombre_carpeta_inspecciones, elemento), os.path.join(nombre_carpeta_procesados, elemento))
            print(f'Movido {elemento} a carpeta /procesados')
        print(f'Tamaño final df datos histórico inspecciones: {df_historico_inspecciones.shape}')
        print('---Finalizado: Importación de datos nuevos de inspecciones---')
    else:
        print('--- No se agregaron datos nuevos de inspecciones/multas')
    df_historico_inspecciones = df_historico_inspecciones.drop_duplicates()
    df_historico_inspecciones.to_parquet(nombre_path_historico_inspecciones)
    return df_historico_inspecciones