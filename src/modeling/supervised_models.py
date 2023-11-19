"""
Este módulo contiene las implementaciones de los modelos supervisados utilizados en el proyecto de aprendizaje automático.

Los modelos implementados son:

- LGBMModel: Implementa un modelo de Gradient Boosting con LightGBM.
- CATModel: Implementa un modelo de Gradient Boosting con CatBoost.
- NNModel: Implementa un modelo de Red Neuronal.

Cada modelo tiene métodos para entrenar y realizar predicciones.

Nota: Este módulo requiere que se instalen las librerías LightGBM, CatBoost y TensorFlow para poder utilizar los modelos correspondientes.

"""

import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from imblearn.pipeline import Pipeline, make_pipeline
from src.preprocessing.preprocessing  import ToDummy, TeEncoder, CardinalityReducer, MinMaxScalerRow
from lightgbm import LGBMClassifier, early_stopping #, Dataset,log_evaluation
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import catboost as cb
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression


def binary_area_peligrosa(x):
    return x.map({'zona peligrosa':1,'zona no peligrosa':0}).to_frame()

def get_preprocesor(preprocesor):
    
    if preprocesor==-1:
        preprocessor = ColumnTransformer(transformers= [],remainder='passthrough')
        
    if preprocesor==2:
        # Actividad 
        pipe_actividad = Pipeline([
                    ('cardinality_reducer', CardinalityReducer(threshold=0.001)),
                    ('a_dummy',ToDummy(['distrito']))
                ])

        # Segmento Tarifa
        pipe_tarifa = Pipeline([
                    ('cardinality_reducer', CardinalityReducer(threshold=0.001)),
                    ('tarifa_te',TeEncoder(['tipo_tarifa'],w=20))
                ])
        
        pipe_distrito = Pipeline([
                    ('cardinality_reducer', CardinalityReducer(threshold=0.001)),
                    ('distrito_te',TeEncoder(['id_distrito'],w=20))
                ])
        
        vars_enc = ['id_cant_fases','id_tarifa']
        t_features = [
            ('var_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), vars_enc),
            # ('a_dummy',ToDummy(['clasif_tarifa'])),
            # ('material_isntalacion_te', TeEncoder(['material_instalacion'],w=10), ['material_instalacion']),
            # ('actividad_cr_dummy', pipe_actividad, ['distrito']),
            ('ditrito_cr_te', pipe_distrito, ['id_distrito']),
            ('area_peligrosa_binary', FunctionTransformer(), 'area_peligrosa'),
            ]

        preprocessor = ColumnTransformer(transformers= t_features,remainder='passthrough')
        
    if preprocesor==3:
        pipe_tarifa = Pipeline([
                    ('cardinality_reducer', CardinalityReducer(threshold=0.003)),
                    ('te',TeEncoder(['id_tarifa'],w=1))
                ])
        pipe_marca_medidor = Pipeline([
                    ('cardinality_reducer', CardinalityReducer(threshold=0.03)),
                    ('te',TeEncoder(['marca_medidor'],w=1))
                ])
        pipe_distrito = Pipeline([
                    ('cardinality_reducer', CardinalityReducer(threshold=0.01)),
                    ('te',TeEncoder(['id_distrito'],w=20))
                ])
        pipe_tipo_medidor = Pipeline([
                    ('cardinality_reducer', CardinalityReducer(threshold=0.01)),
                    ('te',TeEncoder(['id_tipo_medidor'],w=20))
                ])
        pipe_id_ciu = Pipeline([
                    ('cardinality_reducer', CardinalityReducer(threshold=0.002)),
                    ('te',TeEncoder(['id_ciu'],w=20))
                ])
        vars_dummy = ['tip_instalacion','tipo_cliente']
        t_features = [
            ('dummy_var', ToDummy(vars_dummy), vars_dummy),
            ('pipe_tarifa', pipe_tarifa, ['id_tarifa']),
            ('pipe_marca_medidor', pipe_marca_medidor, ['marca_medidor']),
            ('pipe_distrito', pipe_distrito, ['id_distrito']),
            ('pipe_tipo_medidor', pipe_tipo_medidor, ['id_tipo_medidor']),
            ('pipe_id_ciu', pipe_id_ciu, ['id_ciu']),
            ('area_peligrosa_binary', FunctionTransformer(binary_area_peligrosa), 'area_peligrosa'),
            ]

        preprocessor = ColumnTransformer(transformers= t_features,remainder='passthrough')
    return preprocessor

class LGBMModel():
    
        def __init__(self,cols_for_model,hyperparams,search_hip=False,sampling_th = 0.5,preprocesor_num = 3,sampling_method = 'under'):
            """
            Initializes the LGBMModel.

            Args:
                cols_for_model (list): The columns to be used for modeling.
                hyperparams: The hyperparameters for the LGBMClassifier.
                search_hip (bool): Flag indicating whether to perform hyperparameter search.
                sampling_th (float): The sampling threshold.
                preprocesor_num (int): The preprocessor number.
                sampling_method (str): The sampling method ('over' or 'under').
            """
            self.cols_for_model = cols_for_model
            self.sampling_th = sampling_th
            self.preprocesor_num = preprocesor_num
            self.sampling_method = sampling_method
            self.search_hip = search_hip
            self.hyperparams = hyperparams
        
        def build_pipeline_preproceso_model(self):
            preprocessor = get_preprocesor(self.preprocesor_num)
            lgbm_model_search = LGBMClassifier(random_state=314, metric='None',  n_estimators=1000)
            if  self.sampling_method  == 'over':
                over = RandomOverSampler(sampling_strategy=self.sampling_th,random_state=40)
                return make_pipeline(preprocessor,over,lgbm_model_search)
            elif self.sampling_method  == 'under':
                under = RandomUnderSampler(sampling_strategy=self.sampling_th,random_state=40)
                return make_pipeline(preprocessor,under,lgbm_model_search)
            else:
                return make_pipeline(preprocessor,lgbm_model_search)
        
        def train(self,df_train,y_train,df_val=None,y_val=None):
            if  df_val is None:
                 df_train, df_val, y_train, y_val = train_test_split(df_train,y_train, test_size=0.1, random_state=42)
            
            pipe_preproceso_model = self.build_pipeline_preproceso_model()
            
            preprocessor_features = pipe_preproceso_model.steps[0][1]
            # print(preprocessor_features)
            preprocessor_features.fit(df_train[self.cols_for_model], y_train)
            df_val_tra = preprocessor_features.transform(df_val[self.cols_for_model])
            # print(df_val_tra.shape)
            # print(df_val_tra.columns)
            if self.search_hip:
                self.best_score_, self.hyperparams = self.find_hyp_lgbm_model(df_train[self.cols_for_model],y_train,df_val_tra,y_val,pipe_preproceso_model)
                
            params = self.hyperparams
            fit_params = {
                    'eval_metric' : ['auc'],
                    "eval_set": [(df_val_tra, y_val)],
                    "eval_names": ["valid"],
                    "callbacks": [early_stopping(stopping_rounds=30, first_metric_only=True, verbose=False)
                                  #,log_evaluation(0)
                                 ],
                    "categorical_feature": "auto",
                    "feature_name": "auto",
                }
            new_fit_params = {'lgbmclassifier__' + key: fit_params[key] for key in fit_params}
            
            if params is not None:
                pipe_preproceso_model.set_params(**params)
                
            #pipe_preproceso_model.set_params(**params)
            pipe_preproceso_model.fit(df_train[self.cols_for_model], y_train, **new_fit_params)
        
            return pipe_preproceso_model
            
        def find_hyp_lgbm_model(self, X_train,y_train,X_val,y_val,imba_pipeline):
                fit_params = {
                     'eval_metric' : ['auc'],
                    "eval_set": [(X_val, y_val)],
                    "eval_names": ["valid"],
                    "callbacks": [early_stopping(stopping_rounds=30, first_metric_only=True, verbose=False) #,log_evaluation(0)
                                 ],
                    "categorical_feature": "auto",
                    "feature_name": "auto",
                }

                param_test ={
                    'num_leaves': sp_randint(6, 50), 
                     'max_bin': sp_randint(60, 255),
                     'max_depth': sp_randint(5, 20),
                     'min_child_samples': sp_randint(100, 500), 
                     'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
                     'subsample': sp_uniform(loc=0.2, scale=0.8), 
                     'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
                     'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
                     'reg_lambda': [0, 1e-5, 1e-3, 1e-2, 1e-1, 1, 5, 10, 20, 50, 100],
                     'scale_pos_weight':[1,5,20,100],
                #             'is_unbalance':[True,False],
                     'learning_rate':sp_uniform(loc=0.01, scale=0.1),
                     'subsample_freq': sp_randint(5, 20)
                    }

                new_params = {'lgbmclassifier__' + key: param_test[key] for key in param_test}
                new_fit_params = {'lgbmclassifier__' + key: fit_params[key] for key in fit_params}

                random_imba = RandomizedSearchCV(estimator = imba_pipeline, 
                                       param_distributions = new_params, 
                                       cv = 3, 
            #                            scoring = 'average_precision',
                                       scoring= 'roc_auc',
                                       n_jobs = 35,
                                       n_iter = 60,
                                       refit=True,
                                       random_state = 314)
                random_imba.fit(X_train, y_train, **new_fit_params);
                # print('\nBest score reached: {} with params: {} '.format(random_imba.best_score_, random_imba.best_params_))
                return random_imba.best_score_, random_imba.best_params_
            
class CATModel():
        def __init__(self,cols_for_model,hyperparams,search_hip=False,sampling_th = 0.5,preprocesor_num = 3,sampling_method = 'under'):
            self.cols_for_model = cols_for_model
            self.sampling_th = sampling_th
            self.preprocesor_num = preprocesor_num
            self.sampling_method = sampling_method
            self.search_hip = search_hip
            self.hyperparams = hyperparams
            
        def build_pipeline_preproceso_model(self,cat_features):
            cb_model_search = cb.CatBoostClassifier(iterations = 1000,
                                      eval_metric = 'AUC',
                                      loss_function = 'Logloss',
                                      random_seed = 42,
                                      cat_features= cat_features,
                                      logging_level='Silent'
                                     )
            if  self.sampling_method == 'over':
                over = RandomOverSampler(sampling_strategy=self.sampling_th,random_state=40)
                return make_pipeline(over,cb_model_search)
            elif self.sampling_method == 'under':
                under = RandomUnderSampler(sampling_strategy=self.sampling_th,random_state=40)
                return make_pipeline(under,cb_model_search)
            else:
                return make_pipeline(cb_model_search)
        
        def train(self,df_train,y_train,df_val=None,y_val=None):
            if  df_val is None:
                 df_train, df_val, y_train, y_val = train_test_split(df_train,y_train, test_size=0.1, random_state=42)
                    
            cat_features = df_train[self.cols_for_model].select_dtypes(include=['object']).columns.tolist()
            pipe_preproceso_model = self.build_pipeline_preproceso_model(cat_features)
            
            if self.search_hip:
                self.best_score_, self.hyperparams = self.find_hyp_catboost_model(df_train[self.cols_for_model],y_train,df_val[self.cols_for_model],y_val,pipe_preproceso_model)
                
            params = self.hyperparams
            fit_params = {
                        "eval_set": [(df_val[self.cols_for_model],y_val)],
                        'early_stopping_rounds':30
                            }
            new_fit_params = {'catboostclassifier__' + key: fit_params[key] for key in fit_params}
            
            if params is not None:
                pipe_preproceso_model.set_params(**params)
            
            pipe_preproceso_model.fit(df_train[self.cols_for_model], y_train, **new_fit_params);
            return pipe_preproceso_model

        def find_hyp_catboost_model(self,X_train,y_train,X_val,y_val,imba_catboost):
            fit_params = {
                    "eval_set": [(X_val, y_val)],
                    'early_stopping_rounds':30
                        }

            param_test = {
                'depth': sp_randint(4,16),
            #     'min_child_samples': sp_randint(100,500), 
                'learning_rate': sp_uniform(loc = 0.01, scale = 0.1),
                'l2_leaf_reg': sp_uniform(loc=1, scale = 100),
                'border_count':[50,128,254],
            #     'subsample': sp_uniform(loc=0.2, scale=0.8), 
                'bagging_temperature' : [0,1,10,100]
            }

            new_params = {'catboostclassifier__' + key: param_test[key] for key in param_test}
            new_fit_params = {'catboostclassifier__' + key: fit_params[key] for key in fit_params}

            random_imba = RandomizedSearchCV(estimator = imba_catboost, 
                                       param_distributions = new_params, 
                                       cv = 3, 
                                       scoring= 'roc_auc',
                                       n_jobs = 5,
                                       n_iter = 60,
                                       refit=True,
                                       random_state = 314)

            random_imba.fit(X_train, y_train, **new_fit_params);
            # print('\nBest score reached: {} with params: {} '.format(random_imba.best_score_, random_imba.best_params_))
            return random_imba.best_score_, random_imba.best_params_
            
class NNModel():
    
        def __init__(self,features_names,spents_names,search_hip=False,sampling_th = 0.5,preprocesor_num = 3,sampling_method = 'under'):
            """
            Clase para un modelo de red neuronal feedforward.

            Args:
            - features_names: lista de nombres de las características.
            - spents_names: lista de nombres de los consumos.
            - search_hip: booleano que indica si se debe buscar hiperparámetros (opcional).
            - sampling_th: umbral de muestreo para el método de muestreo (opcional).
            - preprocesor_num: número del preprocesador a utilizar (opcional).
            - sampling_method: método de muestreo a utilizar ('over' o 'under') (opcional).
            """
            self.features_names = features_names
            self.spents_names = spents_names
            self.sampling_th = sampling_th
            self.preprocesor_num = preprocesor_num
            self.sampling_method = sampling_method
            self.search_hip = search_hip
#             self.hyperparams = hyperparams
            
            self.EPOCHS = 1000
            self.BATCH_SIZE = 2048
        
        def build_pipeline_preproceso(self):
            preprocessor = get_preprocesor(self.preprocesor_num)
            pipe_features = Pipeline([
                          ('features_var',  preprocessor),
                          ('scaler', MinMaxScaler())
                          ])

            pipe_spent = Pipeline([
                    ('scaler', MinMaxScalerRow())
                ])
            
            if self.sampling_method == 'over':
                ramdom_s = RandomOverSampler(sampling_strategy=self.sampling_th,random_state=40)
            if self.sampling_method == 'under':
                ramdom_s = RandomUnderSampler(sampling_strategy=self.sampling_th, random_state=40)

            return pipe_features,pipe_spent,ramdom_s
            
        def train(self,df_train,y_train,df_val=None,y_val=None):
            if df_val is None:
                 df_train, df_val, y_train, y_val = train_test_split(df_train,y_train, test_size=0.1, random_state=42)
            
            
            
            pipe_features,pipe_spent,ramdom_s = self.build_pipeline_preproceso()
            
            X_train_features = pipe_features.fit_transform(df_train[self.features_names], y_train)
            X_train_spents = pipe_spent.fit_transform(df_train[self.spents_names], y_train)
            
            X_val_features = pipe_features.transform(df_val[self.features_names])
            X_val_spents = pipe_spent.transform(df_val[self.spents_names])
            
            X_train_features = np.concatenate([X_train_features,X_train_spents],axis=1)
            X_val_features = np.concatenate([X_val_features,X_val_spents],axis=1)
            
            X_resample, y_resample = ramdom_s.fit_resample(X_train_features, y_train)
            
            rnn_final_model = self.train_rnn_model(X_resample,y_resample,X_val_features,y_val,output_bias=None)
            return rnn_final_model,pipe_features,pipe_spent

        def train_rnn_model(self,X_train,y_train,X_val,y_val,output_bias=None):
            y_train = np.asarray(y_train)
            y_val = np.asarray(y_val)
            METRICS = [
              tf.keras.metrics.AUC(name='auc'),
              tf.keras.metrics.AUC(name='prc', curve='PR'), 
            ]
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_prc', 
                verbose=1,
                patience=50,
                mode='max',
                restore_best_weights=True)

            if output_bias is not None:
                output_bias = tf.keras.initializers.Constant(output_bias)


            model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, 
        #                           kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                         activation='relu', input_shape=(X_train.shape[-1],)),
        #     tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, 
        #                           kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                         activation='relu'),
        #     tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(32, 
        #                           kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                         activation='relu'),
        #     tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(16, 
        #                           kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                         activation='relu'),
        #     tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)
            ])

            model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=METRICS)

            baseline_history = model.fit(
            X_train,
            y_train,
            batch_size=self.BATCH_SIZE,
            epochs=self.EPOCHS,
            callbacks=[early_stopping],
            validation_data=(X_val, y_val),
            verbose=0
            )
            return model
        
class LSTMNNModel():
    
        def __init__(self,features_names,spents_names,search_hip=False,sampling_th = 0.5,preprocesor_num = 3,sampling_method = 'under'):
            """
            Clase para un modelo de red neuronal LSTM.

            Args:
            - features_names: lista de nombres de las características.
            - spents_names: lista de nombres de los gastos.
            - search_hip: booleano que indica si se debe buscar hiperparámetros (opcional).
            - sampling_th: umbral de muestreo (opcional).
            - preprocesor_num: número de preprocesador a utilizar (opcional).
            - sampling_method: método de muestreo a utilizar (opcional).
            """
            self.features_names = features_names
            self.spents_names = spents_names
            self.sampling_th = sampling_th
            self.preprocesor_num = preprocesor_num
            self.sampling_method = sampling_method
            self.search_hip = search_hip
#             self.hyperparams = hyperparams
            self.periodo = 12
            self.EPOCHS = 1000
            self.BATCH_SIZE = 2048

        
        def build_pipeline_preproceso(self):
            preprocessor = get_preprocesor(self.preprocesor_num)
            pipe_features = Pipeline([
                          ('features_var',  preprocessor),
                          ('scaler', MinMaxScaler())
                          ])

            pipe_spent = Pipeline([
                    ('scaler', MinMaxScalerRow())
                ])
            
            if self.sampling_method == 'over':
                ramdom_s = RandomOverSampler(sampling_strategy=self.sampling_th,random_state=40)
            if self.sampling_method == 'under':
                ramdom_s = RandomUnderSampler(sampling_strategy=self.sampling_th, random_state=40)

            return pipe_features,pipe_spent,ramdom_s
        
        def train(self,df_train,y_train,df_val=None,y_val=None):
            if df_val is None:
                 df_train, df_val, y_train, y_val = train_test_split(df_train,y_train, test_size=0.1, random_state=42)
                    
            pipe_features,pipe_spent,ramdom_s = self.build_pipeline_preproceso()
            
            X_train_features = pipe_features.fit_transform(df_train[self.features_names], y_train)
            X_train_spents = pipe_spent.fit_transform(df_train[self.spents_names], y_train)

            X_val_features = pipe_features.transform(df_val[self.features_names])
            X_val_spents = pipe_spent.transform(df_val[self.spents_names])
            X_val_spents = X_val_spents.reshape((X_val_spents.shape[0],self.periodo,1))

            X_train_features = np.concatenate([X_train_spents,X_train_features],axis=1)
            X_resample, y_resample = ramdom_s.fit_resample(X_train_features, y_train)

            X_resample_features = X_resample[:,self.periodo:]
            X_resample_spents = X_resample[:,:self.periodo]
            X_resample_spents = X_resample_spents.reshape((X_resample_spents.shape[0],self.periodo,1))

            lstm_rnn_final_model = self.train_lstm_rnn_model(X_resample_spents,X_resample_features,y_resample,X_val_spents,X_val_features,y_val)
            return lstm_rnn_final_model,pipe_features,pipe_spent

        def train_lstm_rnn_model(self,X_train_spents,X_train_features,y_train,X_val_spents,X_val_features,y_val,output_bias=None):
            
            y_train = np.asarray(y_train)
            y_val = np.asarray(y_val)
            METRICS = [
              tf.keras.metrics.AUC(name='auc'),
              tf.keras.metrics.AUC(name='prc', curve='PR'), 
            ]
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_prc', 
                verbose=1,
                patience=50,
                mode='max',
                restore_best_weights=True)

            if output_bias is not None:
                output_bias = tf.keras.initializers.Constant(output_bias)


            spents_inputs = tf.keras.Input(X_train_spents.shape[1:])
            x = tf.keras.layers.LSTM(128, activation='relu')(spents_inputs)
            features_inputs = tf.keras.Input(X_train_features.shape[1])

            concat = tf.keras.layers.Concatenate()([features_inputs, x])
            x = tf.keras.layers.Dense(64, activation='relu')(concat)
            x =  tf.keras.layers.Dropout(0.5)(x)
            x = tf.keras.layers.Dense(32, activation='relu')(x)
            x =  tf.keras.layers.Dropout(0.5)(x)
            x = tf.keras.layers.Dense(16, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            outputs = tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)(x)

            model = tf.keras.models.Model([spents_inputs, features_inputs], outputs)

            model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=METRICS)

            history = model.fit([X_train_spents, X_train_features], y_train,
                  validation_data=([X_val_spents, X_val_features], y_val),
                  batch_size=self.BATCH_SIZE,
                  epochs=self.EPOCHS,
        #           steps_per_epoch= steps_per_epoch,
                  callbacks=[early_stopping],
                  verbose=0
        #           class_weight=class_weight
                           )
            return model
        
        
def compute_stacking_model(final_lgbm_model,final_cb_model,df_val_f,y_val_f,cols_for_model):
    # Definir el meta-modelo (StackingClassifier)
    meta_model = LogisticRegression(class_weight='balanced')
    # Crear el modelo de stacking
    stacking_model = StackingClassifier(
        estimators=[
            ('lgbm1', final_lgbm_model),
            ('catboost', final_cb_model)
        ],
        final_estimator=meta_model,
        cv='prefit'
    )

    # Hacer predicciones df_val_f, y_val_f
    combination_final = stacking_model.fit(df_val_f[cols_for_model],y_val_f)
    return combination_final