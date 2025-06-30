#
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#
import torch
from torch.utils.data import Dataset, DataLoader
#
# from AuxiliarydomainConstruction.TDC import TDC
from Data_preparation.MyDataloader import InfiniteDataLoader
from AuxiliarydomainConstruction.My_Auxiliarydomain_Construction import Data_Segment, get_maxSeg_


class SubDataset(Dataset):
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

def dataset_setting_intra(datapath, unseen_datapath, seq_len, batch_size, whether_test=False):

    Data = pd.read_csv(datapath)
    X = np.array(Data.iloc[:, 0:25])
    Y = np.array(Data.iloc[:, 25])

    """归一化"""
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

    if whether_test == True:
        train_size = int(np.round(0.8 * len(X)))
        train_X, train_Y  = Data_X[:train_size, :, :], Data_Y[:train_size, :, :]
        test_X, test_Y = Data_X[train_size:, :, :], Data_Y[train_size:, :, :]

        train_Dataset = SubDataset(train_X ,train_Y)
        test_Dataset = SubDataset(test_X, test_Y)
    else:
        train_Dataset = SubDataset(Data_X, Data_Y)
        test_Dataset = None

    train_Dataloader = DataLoader(train_Dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_Dataloader = DataLoader(test_Dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    """unseen target domain"""
    Unseen_Data = pd.read_csv(unseen_datapath)
    UX = np.array(Unseen_Data.iloc[:, 0:25])
    UY = np.array(Unseen_Data.iloc[:, 25])
    M_SUX = MinMaxScaler()
    M_SUY = MinMaxScaler()
    UX = M_SUX.fit_transform(UX)
    UY = M_SUY.fit_transform(UY.reshape(-1, 1))

    Data_UX, Data_UY = [], []
    for index in range(len(UX) - seq_len):
        Data_UX.append(UX[index:index + seq_len, :])
        Data_UY.append(UY[index:index + seq_len, :])

    Data_UX = np.array(Data_UX)
    Data_UY = np.array(Data_UY)

    unseen_Dataset = SubDataset(Data_UX, Data_UY)
    unseen_Dataloader = DataLoader(unseen_Dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_Dataloader, test_Dataloader, unseen_Dataloader


def dataset_setting_inter(datapath_List, unseen_datapath, seq_len, batch_size, whether_test=False):
    train_LoaderList, test_LoaderList = [], []
    for i in range(len(datapath_List)):
        Data = pd.read_csv(datapath_List[i])
        X = np.array(Data.iloc[:, 0:25])
        Y = np.array(Data.iloc[:, 25])

        """归一化"""
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

        if whether_test == True:
            train_size = int(np.round(0.8 * len(X)))
            train_X, train_Y  = Data_X[:train_size, :, :], Data_Y[:train_size, :, :]
            test_X, test_Y = Data_X[train_size:, :, :], Data_Y[train_size:, :, :]

            train_Dataset = SubDataset(train_X ,train_Y)
            test_Dataset = SubDataset(test_X, test_Y)
        else:
            train_Dataset = SubDataset(Data_X, Data_Y)
            test_Dataset = None

        train_Dataloader = InfiniteDataLoader(dataset=train_Dataset, weights=None, batch_size=batch_size,
                                              num_workers=0)
        test_Dataloader = DataLoader(test_Dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        train_LoaderList.append(train_Dataloader), test_LoaderList.append(test_Dataloader)

    """unseen target domain"""
    Unseen_Data = pd.read_csv(unseen_datapath)
    UX = np.array(Unseen_Data.iloc[:, 0:25])
    UY = np.array(Unseen_Data.iloc[:, 25])
    M_SUX = MinMaxScaler()
    M_SUY = MinMaxScaler()
    UX = M_SUX.fit_transform(UX)
    UY = M_SUY.fit_transform(UY.reshape(-1, 1))

    Data_UX, Data_UY = [], []
    for index in range(len(UX) - seq_len):
        Data_UX.append(UX[index:index + seq_len, :])
        Data_UY.append(UY[index:index + seq_len, :])

    Data_UX = np.array(Data_UX)
    Data_UY = np.array(Data_UY)

    unseen_Dataset = SubDataset(Data_UX, Data_UY)
    unseen_Dataloader = DataLoader(unseen_Dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_LoaderList, test_LoaderList, unseen_Dataloader

def dataset_setting_AdaRNN(datapath, unseen_datapath, seq_len, batch_size):
    Data = pd.read_csv(datapath)
    Unseen_Data = pd.read_csv(unseen_datapath)
    X = np.array(Data.iloc[:, 0:25])
    Y = np.array(Data.iloc[:, 25])
    UX = np.array(Unseen_Data.iloc[:, 0:25])
    UY = np.array(Unseen_Data.iloc[:, 25])

    """归一化"""
    M_SSX = MinMaxScaler()
    M_SSY = MinMaxScaler()
    M_SUX = MinMaxScaler()
    M_SUY = MinMaxScaler()
    SX = M_SSX.fit_transform(X)
    SY = M_SSY.fit_transform(Y.reshape(-1, 1))
    UX = M_SUX.fit_transform(UX)
    UY = M_SUY.fit_transform(UY.reshape(-1, 1))

    # 作用--几乎忽略不计,重复冗余,对于目前的数据集而言 // 但对于别的数据集有用
    split_x_list = get_maxSeg_(data=SX, domain_num=2)
    split_y_list = get_maxSeg_(data=SY, domain_num=2)
    src_x, trg_x = split_x_list[0], split_x_list[1]
    src_y, trg_y = split_y_list[0], split_y_list[1]

    Data_srcX, Data_srcY, Data_trgX, Data_trgY = [], [], [], []
    for index in range(len(src_x) - seq_len):
        Data_srcX.append(src_x[index:index + seq_len, :])
        Data_srcY.append(src_y[index:index + seq_len, :])
    for index in range(len(trg_x) - seq_len):
        Data_trgX.append(trg_x[index:index + seq_len, :])
        Data_trgY.append(trg_y[index:index + seq_len, :])

    Data_srcX = np.array(Data_srcX)
    Data_srcY = np.array(Data_srcY)
    Data_trgX = np.array(Data_trgX)
    Data_trgY = np.array(Data_trgY)
    # print(Data_X.shape)
    # print(Data_Y.shape)

    Data_UX, Data_UY = [], []
    for index in range(len(UX) - seq_len):
        Data_UX.append(UX[index:index + seq_len, :])
        Data_UY.append(UY[index:index + seq_len, :])

    Data_UX = np.array(Data_UX)
    Data_UY = np.array(Data_UY)

    train_Dataset1 = SubDataset(Data_srcX, Data_srcY)
    train_Dataset2 = SubDataset(Data_trgX, Data_trgY)
    unseen_Dataset = SubDataset(Data_UX, Data_UY)

    train_Dataloader1 = DataLoader(train_Dataset1, batch_size=batch_size, shuffle=False, drop_last=True)
    train_Dataloader2 = DataLoader(train_Dataset2, batch_size=batch_size, shuffle=False, drop_last=True)
    unseen_Dataloader = DataLoader(unseen_Dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    train_Dataloader_List = []
    train_Dataloader_List.append(train_Dataloader1)
    train_Dataloader_List.append(train_Dataloader2)

    return train_Dataloader_List, unseen_Dataloader