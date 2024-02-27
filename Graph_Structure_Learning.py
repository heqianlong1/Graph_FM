import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn import metrics
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn import preprocessing



def data_pre(raw):
    cols = raw.columns.values
    dense_feats = [f for f in cols if f[0] == 'I']
    sparse_feats = [f for f in cols if f[0] == 'C']

    def process_dense_feats(data, feats):
        d = data.copy()
        d = d[feats].fillna(0.0)
        for f in feats:
            d[f] = d[f].apply(lambda x: np.log(x + 1) if x > -1 else -1)
        return d

    data_dense = process_dense_feats(raw, dense_feats)

    def process_spares_feats(data, feats):
        d = data.copy()
        d = d[feats].fillna('-1')
        for f in feats:
            d[f] = LabelEncoder().fit_transform(d[f])
        return d

    def OneHot(data, feats):
        d = data.copy()
        sum1 = 0
        for feat in feats:
            if len(d[feat].unique()) > 100:
                d = d.drop(feat, axis=1)
            else:
                sum1 += 1
                d = pd.concat([d, pd.get_dummies(d[feat], prefix=feat)], axis=1)
                d = d.drop(feat, axis=1)
        return d

    data_sparse = process_spares_feats(raw, sparse_feats)
    data_sparse = OneHot(data_sparse, sparse_feats)
    data = pd.concat([data_dense, data_sparse], axis=1)

    transformer = MinMaxScaler()
    data = transformer.fit_transform(data)
    return data


class FMLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim=16, **kwargs):  
        self.input_dim = input_dim
        self.output_dim = output_dim
        super(FMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.input_dim, self.output_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(FMLayer, self).build(input_shape)

    def call(self, x):
        a = K.pow(K.dot(x, self.kernel), 2)
        b = K.dot(K.pow(x, 2), K.pow(self.kernel, 2))
        return K.sum(a - b, 1, keepdims=True) * 0.5

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim



def FM(feature_dim):
    inputs = tf.keras.Input((feature_dim,))
    # LR
    liner = tf.keras.layers.Dense(units=1,
                                  bias_regularizer=tf.keras.regularizers.l2(0.0),
                                  kernel_regularizer=tf.keras.regularizers.l1(0.0),
                                  )(inputs)

    cross = FMLayer(feature_dim)(inputs)

    add = tf.keras.layers.Add()([liner, cross])
    predictions = tf.keras.layers.Activation('sigmoid')(add)

    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['binary_accuracy', 'AUC', 'Recall', 'Precision'])

    return model



def trainFM(data):
    fm = FM(data.shape[1], )
    fm.fit(X_train, y_train, epochs=300, verbose=2, batch_size=256, validation_data=(X_test, y_test))
    return fm

def write(matrix,alpha):
    f = open('Graph.txt', 'w')
    for i in range(0, matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            if a[i][j]>alpha:
                f.write(str(i) + ' ' + str(j) + '\n')
    f.close()
    print('write over')

if __name__ == '__main__':
    #In the "your_data" dataset, columns starting with "C" represent categorical data columns, while those starting with "I" represent numerical data columns.
    raw = pd.read_csv('your_data.csv')
    data = data_pre(raw)
    y = raw['label']
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.25,random_state=42, stratify=y)
    fm = trainFM(data)
    vfk = fm.trainable_variables[2].numpy()
    matrix = np.dot(vfk, vfk.T)
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(matrix)
    write(matrix,alpha)