import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import torch
from torch import nn
from torch.nn import MSELoss
from torch.utils.data import Dataset, DataLoader
from model.utils import cluster_train_test_index, data_drop_na, normalize_train_test

class InputOutputSet(Dataset):
    def __init__(self, input_dt, output_dt):
        super().__init__()
        self.input_dt = input_dt
        self.output_dt = output_dt

    def __getitem__(self, i):
        return self.input_dt[i], self.output_dt[i]

    def __len__(self):
        return len(self.input_dt)

class FnnRegresson(nn.Module):
    def __init__(self, **input_shapes):
        super(FnnRegresson, self).__init__()
        self.input_dim = input_shapes['input_dim']
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out = self.layers(x)
        return out

class TrainTest:
    def __init__(self, model_name: str, normalize=False, **input_shapes):
        self.model_name = model_name
        self.normalize = normalize
        self.input_shapes = input_shapes
        self.loss_function = MSELoss()

    def train_model(self, train_loader):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        if self.model_name == "FNN":
            model = FnnRegresson(self.input_shapes)

        total_data_num = 0
        total_loss = 0
        for train_x, train_y in train_loader:
            train_x = train_x.float()
            train_y = train_y.float()
            optimizer.zero_grad()
            pred = model(train_x).float()
            loss = self.loss_function(pred, train_y)
            total_loss += (loss*train_x.size(0)).item()
            total_data_num += train_x.size(0)
            loss.backward()
            optimizer.step()
        return model

    def predict_model(self, model, test_loader):
        current_test_pred = []
        for test_x, test_y in test_loader:
            test_pred = model(test_x.float())
            current_test_pred.append(test_pred.detach().numpy())
        current_test_pred = np.vstack(current_test_pred)
        return current_test_pred

    def train_predict(self, input_dt: pd.DataFrame, label_dt: pd.Series, train_test_data_id: dict):
        all_pred_info = {}
        for cluster_id in train_test_data_id.keys():
            print(f"Cluster{cluster_id} Train and Test")
            set_index = train_test_data_id[cluster_id]
            train_index, test_index = cluster_train_test_index(set_index)
            train_dt, train_label = input_dt.loc[train_index], label_dt[train_index]
            test_dt, test_label = input_dt.loc[test_index], label_dt[test_index]
            train_dt, test_dt = data_drop_na(train_dt, test_dt)
            if self.normalize:
                train_dt, test_dt = normalize_train_test(train_dt, test_dt)

            cluster_model = self.train_model(train_dt, train_label)
            test_pred = self.predict_model(cluster_model, test_dt)
            all_pred_info[f"cluster{cluster_id}"] = test_pred
            r2_val = round(r2_score(test_label, test_pred), 4)
            print(f"R-squared: {r2_val}")
        return all_pred_info