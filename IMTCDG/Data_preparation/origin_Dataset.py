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

def Dataset_setting(datapath, unseen_datapath, batch_size, whether_test=False):
    Data = pd.read_csv(datapath)
    Unseen_Data = pd.read_csv(unseen_datapath)
    X = np.array(Data.iloc[:, 0:25])
    Y = np.array(Data.iloc[:, 25])
    UX = np.array(Unseen_Data.iloc[:, 0:25])
    UY = np.array(Unseen_Data.iloc[:, 25])

    M_SSX = MinMaxScaler()
    M_SSY = MinMaxScaler()
    # unseen target domain
    M_SUX = MinMaxScaler()
    M_SUY = MinMaxScaler()
    Data_SX = M_SSX.fit_transform(X)
    Data_SY = M_SSY.fit_transform(Y.reshape(-1, 1))
    Data_UX = M_SUX.fit_transform(UX)
    Data_UY = M_SUY.fit_transform(UY.reshape(-1, 1))

    # unseen target domain

    if whether_test == True:
        train_size = int(np.round(0.8 * len(X)))
        train_X, train_Y  = Data_SX[:train_size, :, :], Data_SY[:train_size, :, :]
        test_X, test_Y = Data_SX[train_size:, :, :], Data_SY[train_size:, :, :]

        train_Dataset = subDataset(train_X, train_Y)
        test_Dataset = subDataset(test_X, test_Y)
    else:
        train_Dataset = subDataset(Data_UX, Data_UY)
        test_Dataset = None
        # test_Dataloader = None

    unseen_Dataset = subDataset(Data_UX, Data_UY)

    train_Dataloader = DataLoader(train_Dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_Dataloader = DataLoader(test_Dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    unseen_Dataloader = DataLoader(unseen_Dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_Dataloader, test_Dataloader, unseen_Dataloader
