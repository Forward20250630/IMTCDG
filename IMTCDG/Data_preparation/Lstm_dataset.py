#
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#
import torch
from torch.utils.data import Dataset,DataLoader
#
from Data_preparation.CL_Augmentation import jitter, scaling, CoTmixup

class subDataset(Dataset):
    def __init__(self, X, Y):
        # print(X.shape)
        # num, _, length = X.shape
        # X = X.reshape(num, seq_len, length)
        self.X = torch.from_numpy(X).to(torch.float32)
        # print("self.X", self.X.shape)
        self.Y = torch.from_numpy(Y).to(torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return  self.X[index], self.Y[index]

def dataset_setting(datapath, seq_len, batch_size, whether_test=False):
    Data = pd.read_csv(datapath)

    X = np.array(Data.iloc[:, 0:25])
    Y = np.array(Data.iloc[:, 25])


    M_SSX = MinMaxScaler()
    M_SSY = MinMaxScaler()

    SX = M_SSX.fit_transform(X)
    SY = M_SSY.fit_transform(Y.reshape(-1, 1))


    Data_SX, Data_SY = [], []
    for index in range(len(SX) - seq_len):
        Data_SX.append(SX[index:index + seq_len, :])
        Data_SY.append(SY[index:index + seq_len, :])

    Data_X = np.array(Data_SX)
    Data_Y = np.array(Data_SY)

    # if whether_test == True:
    train_size = int(np.round(0.8 * len(X)))
    train_X, train_Y  = Data_X[:train_size, :, :], Data_Y[:train_size, :, :]
    test_X, test_Y = Data_X[train_size:, :, :], Data_Y[train_size:, :, :]

    train_Dataset = subDataset(train_X, train_Y)
    test_Dataset = subDataset(test_X, test_Y)


    train_Dataloader = DataLoader(train_Dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_Dataloader = DataLoader(test_Dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_Dataloader, test_Dataloader