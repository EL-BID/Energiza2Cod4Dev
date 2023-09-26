import numpy as np
import pandas as pd
import warnings
import tensorflow as tf
from tqdm import tqdm

warnings.filterwarnings('ignore')


class CustomGenerator:
    def __init__(self, df_train, steps_back, steps_forward, sel_vars_to_sup, batch_size=32, shuffle_data=True):
        print('llamo a custom Generator')
        self.df_train = df_train
        self.x_series = {}
        self.y_series = {}
        self.steps_back = steps_back
        self.steps_forward = steps_forward
        self.time_series_col = sel_vars_to_sup  # Columnas que se usan para hacer el to_supervised por ejemplo consumo + dummies
        self.batch_size = batch_size
        self.shuffle_data = shuffle_data
        self.x_time_series_dict = {}
        self.y_time_series_dict = {}
        self.variables = {}
        self.pred_columns = ['consumo']
        self.idx_ucs = []
        self.idxs = []

    def get_unique_ucs_and_idx(self):
        print('get_unique_ucs_and_idx')
        index_df = self.df_train.groupby(['index']).size().reset_index(name='cantidad')  # cantidad por instalacion
        self.idx_ucs = list(index_df['index'].unique())
        self.idxs = [[i, self.idx_ucs[j]] for j in range(len(self.idx_ucs)) for i in range(
            index_df[index_df['index'] == self.idx_ucs[j]].cantidad.values[
                0] - self.steps_back)]  # indices y UCs-->idxs

        # crea diccionario con los id_instalacion y sus volumenes
        for col in tqdm(self.time_series_col, desc='Crear Dict'):
            self.variables[col] = self.df_train.groupby(['index'])[col].apply(np.array)

    def built_dict(self):
        """construye dict. almacenando las uc con los valores de consumo y las variables sel_vars_to_sup """

        for uc in tqdm(self.idx_ucs, desc='dict data gen'):
            self.x_time_series_dict[uc] = np.concatenate \
                ([self.variables[col].loc[uc].reshape(-1, 1) for col in self.time_series_col], axis=1)
            self.y_time_series_dict[uc] = np.concatenate \
                ([self.variables[col].loc[uc].reshape(-1, 1) for col in self.pred_columns], axis=1)

    def get_series(self, uc, start_idx):
        """Devuelve las series de acuerdo a las ventanas por ejemplo de 1 a 6. steps_back es el tamano de la ventana
        y a partir del id de la uc """
        self.x_series = self.x_time_series_dict[uc][start_idx:start_idx + self.steps_back]
        self.y_series = self.y_time_series_dict[uc][
                        start_idx + self.steps_back:start_idx + self.steps_back + self.steps_forward, 0].copy().reshape(
            -1, 1)
        return self.x_series, self.y_series

    def get_dataset(self):
        """ Get random tf.data dataset with given batch size and number of batches. """
        self.get_unique_ucs_and_idx()
        self.built_dict()

        def _generator():
            """
            Yields the next training batch.
            Suppose `samples` is an array [[image1_filename,label1], [image2_filename,label2],...].
            """
            # calcula la cantidad de elementos que habran
            num_samples = int(len(self.idxs) / self.batch_size) * self.batch_size
            print(num_samples)
            if self.shuffle_data:
                idxs_sh = np.random.choice([i for i in range(len(self.idxs))], len(self.idxs), replace=False)
            else:
                idxs_sh = [i for i in range(len(self.idxs))]

            for offset in range(0, num_samples, self.batch_size):
                samples = idxs_sh[offset:offset + self.batch_size]
                X_train = []
                Y_train = []
                Z_index = []
                for idx in samples:
                    start_idx, uc = self.idxs[idx]
                    #                     print('START IDX',start_idx)
                    x, y = self.get_series(uc, start_idx)
                    X_train.append(x)
                    Y_train.append(y)
                    Z_index.append(np.array([[uc]]))

                X_train = np.array(X_train)
                Y_train = np.array(Y_train)
                Z_index = np.array(Z_index)

                yield X_train, Y_train, Z_index

        dataset = tf.data.Dataset.from_generator(
            _generator,
            (tf.float64, tf.float64, tf.int64),  # output_types
            ((self.batch_size, self.steps_back, len(self.time_series_col)), (self.batch_size, self.steps_forward, 1),
             (self.batch_size, 1, 1))  # output_shapes
        )

        return dataset

    def conv_tensor_to_df(self, df):
        x_tensors = []
        y_tensors = []
        indice_uc = []
        for item in tqdm(df, desc="Convirtiendo a Tensor"):
            x_tensors.append(item[0].numpy())
            y_tensors.append(item[1].numpy())
            indice_uc.append(item[2].numpy())

        x_tensors = np.vstack(x_tensors)
        y_tensors = np.vstack(y_tensors)
        indice_uc = np.vstack(indice_uc)

        x_df = pd.DataFrame(x_tensors.reshape((x_tensors.shape[0], self.steps_back  * len(self.time_series_col))))
        y_df = pd.DataFrame(y_tensors.reshape((y_tensors.shape[0], 1)), columns=['y'])
        idx_df = pd.DataFrame(indice_uc.reshape((y_tensors.shape[0], 1)), columns=['index'])
        df_total = pd.concat([idx_df, x_df, y_df], axis=1)
        df_total = self.set_names_cols(df_total)
        return df_total

    def set_names_cols(self,df):
        columns_name = [f'{i}_anterior_{v}' for i in range(self.steps_back, 0, -1) for v in self.time_series_col]
        df.columns = ['index'] + columns_name + ['y']
        return df
