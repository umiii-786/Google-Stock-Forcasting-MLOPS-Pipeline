import torch
from torch import nn
from torch.utils.data import Dataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class MyDataset(Dataset):
    def __init__(self, features, target):
        self.features = features
        self.target = target

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.target[index]
        
class LSTM_Model(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        lstm_layers,
        dense_layers,
        dense_units,
        activation,
        dropout
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )

        layers = []
        in_features = hidden_size

        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "leakyrelu": nn.LeakyReLU()
        }

        for i in range(dense_layers):

            layers.append(nn.Linear(in_features, dense_units[i]))
            layers.append(activations[activation])
            layers.append(nn.Dropout(dropout))

            in_features = dense_units[i]

        layers.append(nn.Linear(in_features, 1))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):

        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x


# class MYLSTM(nn.Module):
#     def __init__(self):
#         super().__init__()

#         # First LSTM layer
#         self.lstm1 = nn.LSTM(
#             input_size=1,
#             hidden_size=128,
#             batch_first=True,
#         )

#         self.fdPart = nn.Sequential(
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),

#             nn.Linear(32, 1)
#         )

#     def forward(self, features):
#         out, (h_n, _) = self.lstm1(features)

#         out = h_n[-1]          # Last hidden state of last LSTM
#         out = self.fdPart(out)

#         return out
    

def convert_into_tensor_and_reshape(X,y):
    X=torch.tensor(X,dtype=torch.float32)
    y=torch.tensor(y,dtype=torch.float32)
    X=X.unsqueeze(-1)

    X=X.to(device)
    y=y.to(device)

    return X,y