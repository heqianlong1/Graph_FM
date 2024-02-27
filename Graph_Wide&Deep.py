import numpy as np
from collections import OrderedDict
import torch.utils.data as Data
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data as GData
from torch_geometric.loader import DataLoader as GDataLoader
import warnings
from scipy.sparse import coo_matrix
from torch_geometric.nn import GCNConv, GATConv,Linear ,TopKPooling ,SAGPooling
warnings.filterwarnings('ignore')


class Linear(nn.Module):
    """
    Linear part
    """

    def __init__(self, input_dim):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=1)

    def forward(self, x):
        return self.linear(x)


class Dnn(nn.Module):
    """
    Dnn part
    """

    def __init__(self, hidden_units, dropout=0.):

        super(Dnn, self).__init__()

        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])
        self.relus = nn.ModuleList(
            [nn.ReLU() for i in range(len(hidden_units) - 1)])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        for i in range(len(self.linears)):
            fc = self.linears[i](x)
            fc = self.relus[i](fc)
            fc = self.dropout(fc)
            x = fc
        return x


class Graph_WideDeep(nn.Module):
    def __init__(self,feat_sizes,feat_sizes_raw, ratio ,new_features_length ,num_features,sparse_feature_columns, dense_feature_columns, dnn_hidden_units, embedding_size=8,l2_reg_dnn=1, batch_size=256,dnn_dropout=0,device='cpu'):
        super(Graph_WideDeep, self).__init__()
        self.ratio =ratio
        self.new_feature_length = new_features_length
        self.num_featrues =num_features
        self.l2_reg_dnn =l2_reg_dnn
        #fm
        self.n = feat_sizes_raw
        self.k =embedding_size
        self.linear_fm = nn.Linear(self.n, 1, bias=True)
        self.v = nn.Parameter(torch.Tensor(self.k, self.n))  
        nn.init.xavier_uniform_(self.v)
        #GAT
        self.conv1 = GATConv(1, 1, add_self_loops=True, dropout=0.0).to(device)
        self.conv2 = GATConv(1, 1, add_self_loops=True, dropout=0.0).to(device)
        self.conv3 = GCNConv(1, 1).to(device)
        # self.pool1 = TopKPooling(1, ratio=self.ratio).to(device)
        self.pool1 = SAGPooling(1, ratio=self.ratio).to(device)
        self.dense_feature_cols = dense_feature_columns
        self.sparse_feature_cols = sparse_feature_columns
        self.embedding_size =embedding_size
        self.batch_size = batch_size
        self.device =device
        self.feat_sizes =feat_sizes
        self.feature_index = self.build_input_features(self.feat_sizes)
        # embedding
        self.embedding_dict1 = self.create_embedding_matrix(self.sparse_feature_cols, feat_sizes, self.embedding_size,
                                                            sparse=False, device=self.device)
        self.embedding_dict2 = self.create_embedding_matrix(self.sparse_feature_cols, feat_sizes, 1,
                                                            sparse=False, device=self.device)
        self.dnn_input_size = self.embedding_size * len(self.sparse_feature_cols) + len(self.dense_feature_cols)
        hidden_units = [self.dnn_input_size] + dnn_hidden_units
        self.dnn_network = Dnn(hidden_units,dnn_dropout)
        self.linear = Linear(len(self.dense_feature_cols))
        self.final_linear = nn.Linear(hidden_units[-1], 1)

    def forward(self, X):
        X = X[:, :-(self.new_feature_length)]
        sparse_embedding_list1 = [self.embedding_dict2[feat](
            X[:, self.feature_index[feat][0]:self.feature_index[feat][1]].long())
            for feat in self.sparse_feature_cols]
        nums = int(X.shape[0])
        sparse_emb = torch.cat(sparse_embedding_list1, dim=-1).reshape(nums, -1)
        dense_emb = X[:, len(self.sparse_feature_cols):]
        inputs = [sparse_emb, dense_emb]
        conv_featrues = torch.cat(inputs, dim=1)
        x = conv_featrues
        x1 = self.linear_fm(x)
        square_of_sum = torch.mm(x, self.v.T) * torch.mm(x, self.v.T)
        sum_of_square = torch.mm(x * x, self.v.T * self.v.T)
        x2 = 0.5 * torch.sum((square_of_sum - sum_of_square), dim=-1, keepdim=True)
        fm_result = x1 + x2
        v1 =self.v
        v1 = torch.nn.functional.normalize(v1, dim=0)
        vtv = abs(torch.mm(v1.T , v1))
        temp = torch.zeros(self.n, self.n)
        vtv = torch.where(vtv > 0.2, vtv, temp)

        arr_edge1 = coo_matrix(vtv.detach().numpy())
        edge_index = [arr_edge1.row, arr_edge1.col]
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        dataset = []
        for j in range(conv_featrues.shape[0]):
            datax = conv_featrues[j]
            datax = datax.reshape(-1, 1)
            data = GData(x=datax, edge_index=edge_index)
            dataset.append(data)
        train_loader = GDataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for data in train_loader:
            dataNew = self.conv1(data.x, data.edge_index)
            x, edge_index, _, batch, _, _ = self.pool1(dataNew, data.edge_index, None, data.batch)
            new_features = x.reshape(nums, -1)
        new_features = torch.nn.functional.normalize(new_features, dim=0)
        inputs = [X, new_features]

        X = torch.cat(inputs, dim=1)

        sparse_input, dense_input = X[:, :len(self.sparse_feature_cols)], X[:, len(self.sparse_feature_cols):]
        sparse_embedding_list = [self.embedding_dict1[feat](
            X[:, self.feature_index[feat][0]:self.feature_index[feat][1]].long())
            for feat in self.sparse_feature_cols]

        sparse_embeds = torch.cat(sparse_embedding_list, axis=-1)
        sparse_embeds = sparse_embeds.reshape(-1,len(self.sparse_feature_cols)*self.embedding_size)
        dnn_input = torch.cat([sparse_embeds, dense_input], axis=-1)

        # Wide
        wide_out = self.linear(dense_input)

        # Deep
        deep_out = self.dnn_network(dnn_input)
        deep_out = self.final_linear(deep_out)

        # out
        outputs = F.sigmoid(0.5 * (wide_out + deep_out +fm_result))

        return outputs


    def fit(self, train_input, y_label, val_input, y_val, epochs=15, verbose=5):
        x = [train_input[feature] for feature in self.feature_index]
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1) 

        train_tensor_data = Data.TensorDataset(torch.from_numpy(np.concatenate(x, axis=-1)), torch.from_numpy(y_label))
        train_loader = DataLoader(dataset=train_tensor_data,shuffle=True, batch_size=self.batch_size)


        print(self.device, end="\n")
        model = self.train()
        loss_func = F.binary_cross_entropy
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay = 0.0)

        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // self.batch_size + 1

        print("Train on {0} samples,  {1} steps per epoch".format(
            len(train_tensor_data), steps_per_epoch))

        for epoch in range(epochs):
            loss_epoch = 0
            total_loss_epoch = 0.0
            train_result = {}
            pred_ans = []
            true_ans = []

            with torch.autograd.set_detect_anomaly(True):
                for index, (x_train,y_train) in enumerate(train_loader):
                    x = x_train.to(self.device).float()
                    y = y_train.to(self.device).float()
                    y_pred = model(x).squeeze()
                    optimizer.zero_grad()
                    loss = loss_func(y_pred, y.squeeze(),reduction='mean')
                    loss = loss + self.l2_reg_dnn * self.get_L2_Norm()
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    total_loss_epoch = total_loss_epoch + loss.item()
                    y_pred = y_pred.cpu().data.numpy()  # .squeeze()
                    pred_ans.append(y_pred)
                    true_ans.append(y.squeeze().cpu().data.numpy())

            if (epoch % verbose == 0):
                print('epoch %d train loss is %.4f train AUC is %.4f' %
                      (epoch,total_loss_epoch / steps_per_epoch,roc_auc_score(np.concatenate(true_ans), np.concatenate(pred_ans))))
                self.val_auc_logloss(val_input, y_val, batch_size=50000)
                print(" ")

    def predict(self, test_input, batch_size = 256, use_double=False):
        """

        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param test_input:test dataset.
        :param batch_size: Integer. If unspecified, it will default to 256.
        :return: Numpy array(s) of predictions.
        """
        model = self.eval()
        x = [test_input[feature] for feature in self.feature_index]

        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)  

        tensor_data = Data.TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1)))
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=self.batch_size)


        pred_ans = []
        with torch.no_grad():
            for index, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()

                y_pred = model(x).cpu().data.numpy()  # .squeeze()
                pred_ans.append(y_pred)

        if use_double:
            return np.concatenate(pred_ans).astype("float64")
        else:
            return np.concatenate(pred_ans)

    def get_L2_Norm(self ):

        loss = torch.zeros((1,), device=self.device)

        for t in self.embedding_dict1.parameters():
            loss = loss+ torch.norm(t)
        for t in self.embedding_dict2.parameters():
            loss = loss+ torch.norm(t)
        for t in self.conv1.parameters():
            loss = loss+ torch.norm(t)
        for t in self.linear_fm.parameters():
            loss = loss+ torch.norm(t)
        for t in self.pool1.parameters():
            loss = loss+ torch.norm(t)
        loss = loss + torch.norm(self.v)
        return  loss

    def create_embedding_matrix(self ,sparse_feature_columns, feat_sizes,embedding_size,init_std=0.0001, sparse=False, device='cpu'):
        embedding_dict = nn.ModuleDict(
            {feat: nn.Embedding(feat_sizes[feat], embedding_size, sparse=False)
             for feat in sparse_feature_columns}
        )
        for tensor in embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)
        return embedding_dict.to(device)

    def build_input_features(self, feat_sizes):
        features = OrderedDict()
        start = 0
        for feat in feat_sizes:
            feat_name = feat
            if feat_name in features:
                continue
            features[feat_name] = (start, start + 1)
            start += 1
        return  features

    def val_auc_logloss(self, val_input, y_val, batch_size=50000, use_double=False):
        pred_ans = self.predict(val_input, batch_size)
        pred = np.where(pred_ans > 0.5, pred_ans, 0)
        pred = np.where(pred <= 0.5, pred, 1)
        print(round(metrics.log_loss(y_val, pred_ans),4),
            round(metrics.roc_auc_score(y_val, pred_ans),4),
            round(metrics.accuracy_score(y_val,pred),4),
            round(metrics.f1_score(y_val, pred), 4),
            round(metrics.recall_score(y_val, pred), 4),
            round(metrics.precision_score(y_val, pred), 4)
            )