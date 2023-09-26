import pandas as pd

class PreprocessingFile:

    def __init__(self, df):
        self.df = df

    def create_index(self, cols):
        if len(cols) == 1:
            self.df['index'] = self.df[cols[0]].factorize()[0] + 1
        else:  # crear indice con mas de 1 columna
            triples = self.df[cols].apply(tuple, axis=1)
            self.df['index'] = pd.factorize(triples)[0]

#     def filter_date(self, date1=None, date2=None):
#         if (date1 is None)&(date2 is None):
#             pass
#         else:
#             print('[INFO]...Filtrando Fechas')
#             self.df = self.df.date.between(date1, date2)
#             self.df.loc[(~self.df.date.between(date1, date2)), 'causa'] = 'dates_without_analysis'
#         return self.df

    def clean_estado(self, estados=[5, 6, 7, 8]):
        list_estados = self.df[self.df.estado.isin(estados)]['index'].unique().tolist()
        print('[INFO]...cantidad de numcta a eliminar con estado ' + str(estados) + ':' + str(len(list_estados)))
        self.df.loc[(self.df['index'].isin(list_estados)), 'causa'] = 'estcta_eliminados'
        
        return self.df

    def get_cant_category(self):
        # obtener cantidad de categorias de acuerdo al indice generado (get number of categories according to the index generated)
        df_categoria = self.df.groupby(['index']).categoria.nunique().reset_index(name='cant_categoria')
        df_categoria.cant_categoria.value_counts()
        print('[INFO]...Cantidad Instalaciones y Clientes Con Mas de 1 categoria:',df_categoria[df_categoria.cant_categoria > 1].shape)
        return df_categoria

    def delete_gt_one_category(self):
        print('[INFO]...Marcando usuario con mas de 1 categoria')
        df_categoria = self.get_cant_category()
        self.df.loc[(self.df['index'].isin(df_categoria[df_categoria.cant_categoria > 1]['index'].unique())), 'causa'] = 'delete_gt_one_category'

    def get_cant_method_consump_percentage(self):
        df_metodos = self.df.groupby(['index']).metodo_consumo.value_counts(normalize=True).reset_index(name='cant_metodos')
        return df_metodos

    def delete_gt_x_percentage(self, percent=0.5):
        print('[INFO]...Marcando usuarios con mas del x porcentage en metcon')
        df_metodos = self.get_cant_method_consump_percentage()
        mask_metodos = (df_metodos.cant_metodos > percent) & (df_metodos.metodo_consumo == 1)
        print('[INFO]...Cantidad Instalaciones y Clientes Con Mas de 1 metodo de consumo:', df_metodos[mask_metodos].shape)        
        self.df.loc[(self.df['index'].isin(df_metodos[mask_metodos]['index'].tolist())), 'causa'] = 'deleted_gt_one_metcon'

    def get_discontinous_dates(self):
        self.df.sort_values(['index', 'date'], inplace=True)
        df_diff_days =self.df.groupby(['index']).apply(lambda x: sum(x.date.diff(periods=1).astype('timedelta64[D]') > 31))
        df_diff_days = df_diff_days.rename('diff_m').reset_index()
        return df_diff_days

    def delete_discontinous_dates(self):
        print('[INFO]...Marcando Fechas Discontinuas')
        df_diff_days = self.get_discontinous_dates()
        self.df.loc[(self.df['index'].isin(df_diff_days[df_diff_days.diff_m > 0]['index'].tolist())), 'causa'] = 'deleted_discountinuos_dates'

    def return_clean_df(self):
        self.df.causa.fillna('',inplace=True)
        df_eliminar = self.df[self.df.causa != '']
        print("[INFO]...Se eliminaron -->" + str(df_eliminar.causa.value_counts()))
        self.df = self.df[self.df.causa == '']
        cols_to_drop = ['causa']
        self.df.drop(columns=cols_to_drop, inplace=True)
        return self.df, df_eliminar

    def create_clean_file_run(self, cols):
        "cols: columnas que van a formar el indice"
        print('[INFO]...Creando Indice')
        self.create_index(cols)
        self.df['causa'] = ""
        print('[INFO]...Eliminando Estados')
        self.clean_estado(estados=[5, 6, 7, 8])
        self.df, df_eliminar = self.return_clean_df()
        return self.df, df_eliminar
        
#     def get_num_cons_por_user(self, cant_cons=7):
#         df_id_instalacion_total = self.df.groupby(['index']).size().rename('cant_mediciones').reset_index()
#         df_id_instalacion_total = df_id_instalacion_total[df_id_instalacion_total.cant_mediciones>=cant_cons]
#         return df_id_instalacion_total
        
#     def delete_lt_minimum_datapoints(self, cant_cons):
#         df_id_instalacion_total = self.get_num_cons_por_user(cant_cons)
#         self.df.loc[(self.df['index'].isin(df_id_instalacion_total['index'].unique())), 'causa'] = 'delete_cons_insuficientes'
        

    def data_wrangling(self):
        self.df['causa'] = ""
#         self.create_clean_file_run(cols)
#         self.filter_date(date1, date2)
        self.delete_gt_one_category()
        self.delete_gt_x_percentage(percent=0.5)
        self.delete_discontinous_dates()
        self.df, df_eliminar = self.return_clean_df()
        return self.df, df_eliminar


