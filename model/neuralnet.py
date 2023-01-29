import numpy as np
import torch
from torch import nn
from torch.nn import MSELoss
from torch.utils.data import DataLoader

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
        out = self.layers(x).flatten()
        return out

class NnModel:
    def __init__(self, model):
        self.model = model
        self.loss_function = MSELoss()

    def fit(self, train_loader: DataLoader, epoch: int):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        for e in range(epoch):
            total_data_num = 0
            total_loss = 0
            for train_x, train_y in train_loader:
                train_x = train_x.float()
                train_y = train_y.float()
                optimizer.zero_grad()
                pred = self.model(train_x).float()
                loss = self.loss_function(pred, train_y)
                total_loss += (loss*train_x.size(0)).item()
                total_data_num += train_x.size(0)
                loss.backward()
                optimizer.step()
            mean_loss = total_loss/total_data_num
            print(f"Epoch{e+1} - Mean Loss: {mean_loss}")

    def predict(self, pred_loader: DataLoader):
        all_pred_vals = []
        for pred_x, _ in pred_loader:
            pred_x = pred_x.float()
            pred_val = self.model(pred_x).float()
            all_pred_vals.append(pred_val.detach().numpy())
        all_pred_vals = np.hstack(all_pred_vals)
        return all_pred_vals

class TrainTest:
    def __init__(self, model_name: str, input_dim: dict):
        self.model_name = model_name
        self.input_dim = input_dim

    def define_model(self, input_dim):
        if self.model_name == 'FNN':
            model = FnnRegresson(input_dim=input_dim)
        return NnModel(model)

    def train(self, train_dataset: dict, epoch: int):
        self.all_models = {}
        for cluster_id in train_dataset.keys():
            print(f"Cluster{cluster_id} Train")
            data_loader = train_dataset[cluster_id]
            model = self.define_model(self.input_dim[cluster_id])
            model.fit(data_loader, epoch)
            self.all_models[cluster_id] = model

    def predict(self, pred_dataset: dict):
        all_pred_vals = {}
        for cluster_id in pred_dataset.keys():
            data_loader = pred_dataset[cluster_id]
            pred_val = self.all_models[cluster_id].predict(data_loader)
            all_pred_vals[f"cluster{cluster_id}"] = pred_val
        return all_pred_vals
