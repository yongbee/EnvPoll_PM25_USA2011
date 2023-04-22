import numpy as np
import torch
from torch import nn
from torch.nn import MSELoss
from torch.utils.data import DataLoader

class AutoencoderCnn(nn.Module):
    def __init__(self, encode_dim, **input_shapes):
        super(AutoencoderCnn, self).__init__()
        self.input_dim = input_shapes['input_dim']
        self.input_len = input_shapes['input_len']
        self.encode_dim = encode_dim
        self._define_encoder()
        self._define_decoder()

    def _define_encoder(self):
        self.enocde_layer1 = nn.Conv1d(in_channels=self.input_dim, out_channels=16, kernel_size=1)
        self.encode_layer2 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=1)
        self.encode_layer3 = nn.Conv1d(in_channels=8, out_channels=4, kernel_size=3, padding="same")
        self.encode_layer4 = nn.Linear(4*self.input_len, self.encode_dim*self.input_len)

    def _define_decoder(self):
        self.deocde_layer1 = nn.Conv1d(in_channels=self.encode_dim, out_channels=4, kernel_size=3, padding="same")
        self.decode_layer2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, padding="same")
        self.decode_layer3 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=1)
        self.decode_layer4 = nn.Conv1d(in_channels=16, out_channels=self.input_dim, kernel_size=1)

    def encode(self, x):
        x = torch.relu(self.enocde_layer1(x))
        x = torch.relu(self.encode_layer2(x))
        x = torch.relu(self.encode_layer3(x))
        x = x.view(-1, 4*self.input_len)
        x = self.encode_layer4(x)
        return x
    
    def decode(self, x):
        x = x.view(-1, self.encode_dim, self.input_len)
        x = torch.relu(self.deocde_layer1(x))
        x = torch.relu(self.decode_layer2(x))
        x = torch.relu(self.decode_layer3(x))
        x = self.decode_layer4(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

class AutoencodeModel:
    def __init__(self, model):
        self.model = model
        self.loss_function = MSELoss()

    def fit(self, train_loader: DataLoader, epoch: int):
        optimizer = torch.optim.NAdam(self.model.parameters(), lr=0.001)
        
        for e in range(epoch):
            total_data_num = 0
            total_loss = 0
            for train_x in train_loader:
                train_x = train_x.float()
                optimizer.zero_grad()
                pred = self.model(train_x).float()
                loss = self.loss_function(pred, train_x)
                total_loss += (loss*train_x.size(0)).item()
                total_data_num += train_x.size(0)
                loss.backward()
                optimizer.step()
            mean_loss = total_loss/total_data_num
            print(f"Epoch{e+1} - Mean Loss: {mean_loss}")

    def encode(self, encode_loader: DataLoader):
        all_encode_vals = []
        for encode_x in encode_loader:
            encode_x = encode_x.float()
            encode_val = self.model.encode(encode_x).float()
            all_encode_vals.append(encode_val.detach().numpy())
        all_encode_vals = np.vstack(all_encode_vals)
        return all_encode_vals
    
    def decode(self, decode_loader: DataLoader):
        all_decode_vals = []
        for decode_x in decode_loader:
            decode_x = decode_x.float()
            decode_val = self.model(decode_x).float()
            all_decode_vals.append(decode_val.detach().numpy())
        all_decode_vals = np.vstack(all_decode_vals)
        return all_decode_vals

class AutoencoderTrainTest:
    def __init__(self, model_name: str, encode_dim: int, input_shape: dict):
        self.model_name = model_name
        self.encode_dim = encode_dim
        self.input_shape = input_shape

    def define_model(self, input_shape):
        if self.model_name == 'CNN':
            model = AutoencoderCnn(self.encode_dim, input_dim=input_shape[0], input_len=input_shape[1])
        return AutoencodeModel(model)

    def train(self, train_dataset: dict, epoch: int):
        self.all_models = {}
        for cluster_id in train_dataset.keys():
            print(f"Cluster{cluster_id} Train")
            data_loader = train_dataset[cluster_id]
            model = self.define_model(self.input_shape[cluster_id])
            model.fit(data_loader, epoch)
            self.all_models[cluster_id] = model

    def encode(self, encode_dataset: dict):
        all_encode_vals = {}
        for cluster_id in encode_dataset.keys():
            data_loader = encode_dataset[cluster_id]
            encode_val = self.all_models[cluster_id].encode(data_loader)
            all_encode_vals[f"cluster{cluster_id}"] = encode_val
        return all_encode_vals
    
    def decode(self, decode_dataset: dict):
        all_decode_vals = {}
        for cluster_id in decode_dataset.keys():
            data_loader = decode_dataset[cluster_id]
            decode_val = self.all_models[cluster_id].decode(data_loader)
            all_decode_vals[f"cluster{cluster_id}"] = decode_val
        return all_decode_vals
