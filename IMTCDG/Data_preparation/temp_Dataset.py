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

def Dataset_setting(datapath, auxi_datapath, unseen_datapath, seq_len, batch_size, whether_test=False):
    Data = pd.read_csv(datapath)
    Auxi_Data = pd.read_csv(auxi_datapath)
    Unseen_Data = pd.read_csv(unseen_datapath)
    X = np.array(Data.iloc[:, 0:25])
    Y = np.array(Data.iloc[:, 25])
    AX = np.array(Auxi_Data.iloc[:, 0:25])
    AY = np.array(Auxi_Data.iloc[:, 25])
    UX = np.array(Unseen_Data.iloc[:, 0:25])
    UY = np.array(Unseen_Data.iloc[:, 25])

    M_SSX = MinMaxScaler()
    M_SSY = MinMaxScaler()
    M_SAX = MinMaxScaler()
    M_SAY = MinMaxScaler()
    # unseen target domain
    M_SUX = MinMaxScaler()
    M_SUY = MinMaxScaler()
    SX = M_SSX.fit_transform(X)
    SY = M_SSY.fit_transform(Y.reshape(-1, 1))
    AX = M_SAX.fit_transform(AX)
    AY = M_SAY.fit_transform(AY.reshape(-1, 1))
    UX = M_SUX.fit_transform(UX)
    UY = M_SUY.fit_transform(UY.reshape(-1, 1))

    Data_SX, Data_SY = [], []
    for index in range(len(SX) - seq_len):
        Data_SX.append(SX[index:index + seq_len, :])
        Data_SY.append(SY[index:index + seq_len, :])

    Data_X = np.array(Data_SX)
    Data_Y = np.array(Data_SY)
    # print(Data_X.shape)
    # print(Data_Y.shape)

    Data_AX, Data_AY = [], []
    for index in range(len(AX) - seq_len):
        Data_AX.append(AX[index:index + seq_len, :])
        Data_AY.append(AY[index:index + seq_len, :])

    Data_AX = np.array(Data_AX)
    Data_AY = np.array(Data_AY)

    # unseen target domain
    Data_UX, Data_UY = [], []
    for index in range(len(UX) - seq_len):
        Data_UX.append(UX[index:index + seq_len, :])
        Data_UY.append(UY[index:index + seq_len, :])

    Data_UX = np.array(Data_UX)
    Data_UY = np.array(Data_UY)

    if whether_test == True:
        train_size = int(np.round(0.8 * len(X)))
        train_X, train_Y  = Data_X[:train_size, :, :], Data_Y[:train_size, :, :]
        test_X, test_Y = Data_X[train_size:, :, :], Data_Y[train_size:, :, :]

        train_Dataset = subDataset(train_X, train_Y)
        test_Dataset = subDataset(test_X, test_Y)
    else:
        train_Dataset = subDataset(Data_X, Data_Y)
        test_Dataset = None
        # test_Dataloader = None

    auxi_Dataset = subDataset(Data_AX, Data_AY)
    unseen_Dataset = subDataset(Data_UX, Data_UY)

    train_Dataloader = DataLoader(train_Dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    auxi_Dataloader = DataLoader(auxi_Dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_Dataloader = DataLoader(test_Dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    unseen_Dataloader = DataLoader(unseen_Dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_Dataloader, auxi_Dataloader, test_Dataloader, unseen_Dataloader