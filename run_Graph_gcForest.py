import numpy as np
import pandas as pd
from sklearn import preprocessing
from source.Cascade_Forest.Layer import layer
from source.Base_module import Base_estimators as Be
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import os

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

le = preprocessing.LabelEncoder()

"Get the data"
data_raw = pd.read_csv('your_data.csv')
data = data_pre(data_raw)
train_data = data.iloc[:, 0:-1].to_numpy()
label = le.fit_transform(data_raw.iloc[:,-1].to_numpy())

transformer =StandardScaler(with_mean=False).fit(train_data)
train_data = transformer.transform(train_data)
x = train_data
y = label

"Spilt the data"
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42, stratify=y)


"Generate 4 random forest classifier and 4 completely random forest classifier"
clf1 = Be.RFC(n_estimators=25, max_depth=None, random_state=None, n_jobs=1)
clf2 = Be.RFC(n_estimators=25, max_depth=None, max_features="auto", random_state=None,
              n_jobs=-1)
clf3 = Be.RFC(n_estimators=25, max_depth=None, random_state=None, n_jobs=1)
clf4 = Be.RFC(n_estimators=25, max_depth=None, max_features="auto", random_state=None,
              n_jobs=-1)
clf5 = Be.RFC(n_estimators=25, max_depth=None, random_state=None, n_jobs=1)
clf6 = Be.RFC(n_estimators=25, max_depth=None, max_features="auto", random_state=None,
              n_jobs=-1)
clf7 = Be.RFC(n_estimators=25, max_depth=None, random_state=None, n_jobs=1)
clf8 = Be.RFC(n_estimators=25, max_depth=None, max_features="auto", random_state=None,
              n_jobs=-1)

"Generate 10 layer and filled with classifer's set"
c = {"crf1": clf1, "crf2": clf3, "crf3": clf5, "crf4": clf7, "crf5": clf2, "crf6": clf4, "crf7": clf6, "crf8": clf8}
layer1 = layer()
layer1.add(**c)
layer2 = layer()
layer2.add(**c)
layer3 = layer()
layer3.add(**c)
layer4 = layer()
layer4.add(**c)
layer5 = layer()
layer5.add(**c)
layer6 = layer()
layer6.add(**c)
layer7 = layer()
layer7.add(**c)
layer8 = layer()
layer8.add(**c)
layer9 = layer()
layer9.add(**c)
layer10 = layer()
layer10.add(**c)

from Cascade_Forest.Cascade_Forest import cascade_forest



"Initialize cascade forest structure, you want save the model generated in the validation step into the 'yeast' directory"

cs1 = cascade_forest(random_state=None, n_jobs=-1, directory='source/model_pkl', metrics='accuracy')
cs2 = cascade_forest(random_state=None, n_jobs=-1, directory='source/model_pkl', metrics='accuracy')
cs3 = cascade_forest(random_state=None, n_jobs=-1, directory='source/model_pkl', metrics='accuracy')

"Add each layer to cascade forest structure"
cs1.add(layer1)
cs1.add(layer2)
cs1.add(layer3)
cs1.add(layer4)
cs1.add(layer5)
cs1.add(layer6)
cs1.add(layer7)
cs1.add(layer8)
cs1.add(layer9)
cs1.add(layer10)

cs2.add(layer1)
cs2.add(layer2)
cs2.add(layer3)
cs2.add(layer4)
cs2.add(layer5)
cs2.add(layer6)
cs2.add(layer7)
cs2.add(layer8)
cs2.add(layer9)
cs2.add(layer10)

cs3.add(layer1)
cs3.add(layer2)
cs3.add(layer3)
cs3.add(layer4)
cs3.add(layer5)
cs3.add(layer6)
cs3.add(layer7)
cs3.add(layer8)
cs3.add(layer9)
cs3.add(layer10)




import networkx as nx

"Read the graph input"
G1 = nx.read_edgelist('Graph.txt', create_using=nx.Graph(), nodetype=None, data=[('weight', int)])
print(len(G1.nodes))

from Cascade_Forest.Multi_grained_scanning import scanner

"Generate the scanner"


################## graph-based gcForest ########################

"Using the graph-based approach to scan the data"


if __name__=='__main__':
    sc1 = scanner(stratify=True, clf_set=(clf1, clf2), n_splits=3, random_state=None,
                  walk_length=(15,), num_walks=2, p=1, q=100, scale=(200, 56, 8))
#80,56,8


    transformed_train1, transformed_test1 = sc1.graph_embedding(G1, X_train, y_train, X_test)
    print('embedding')
    "Training"
    cs1.fit(transformed_train1, y_train)
    "Predicting"
    cs1.predict(transformed_test1)
    "The accuracy score"
    score = cs1.score(y_test)
    print("Graph_based gcForest's accuracy :{:.2f} %".format(score * 100))
    pred_ans = cs1.pre_result()
    pred = np.where(pred_ans > 0.5, pred_ans, 0)
    pred = np.where(pred < 0.5, pred, 1)

    from sklearn.metrics import log_loss

    logloss = log_loss(y_test, pred_ans)
    print("Logloss Score : %.4g" % logloss)
    auc_score = metrics.roc_auc_score(y_test, pred)
    print("AUC Score : %.4g" % auc_score)
    from sklearn.metrics import f1_score

    score2 = f1_score(y_true=y_test, y_pred=pred, average='binary', pos_label=1)
    print("f1_score : %.4g" % score2)
    score =metrics.recall_score(y_test,pred)
    print("recall Score : %.4g" % score)
    score =metrics.precision_score(y_test,pred)
    print("precision Score : %.4g" % score)

