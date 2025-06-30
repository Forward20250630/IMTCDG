#
import os
import random
import time, datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#
import torch
import torch.nn as nn
import torch.optim as optim
#
from TemporalDistance.AdvSKM import AdvSKM
from GlobalLocalTemporalCL.CLoss import NTXentLoss
from Data_preparation.CL_Augmentation import CoTmixup
from Data_preparation.temp_Dataset import Dataset_setting
from BasicNetwork.A_CNN_TPA_LSTM import A_CNN_TPA_LSTM, A_CNN_TPA_LSTM_2
from GlobalLocalTemporalCL.GLTCL import GlobalTemporalCL, LocalTemporalCL
device = "cuda" if torch.cuda.is_available() else "cpu"

seed_value = 3407
# numpy
np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)
# torch
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)


def train_DomainGeneralization(model, GlobalTemporalHead, LocalTemporalHead, train_dataloader,
                               auxi_dataloader, unseen_dataloader, Lambda, Gamma1, Gamma2, Alpha,
                               Beta, Omega, temp_shift, temperature, DG_fn_List, batch_size, epochs, lr, n_cycle):

    model.to(device)
    GlobalTemporalHead.to(device)
    LocalTemporalHead.to(device)
    DG_fn1, DG_fn2 = DG_fn_List[0], DG_fn_List[1]

    D_INTRA = 0
    D_INTER = 0
    MU = 0
    train_Losses = []
    tMSE_Losses, tDG_Losses, tG_Losses, tL_Losses = [], [], [], []
    test_Losses = []
    test_Metric = []

    best_auc = []
    best_epoch = 0
    best_mse = 100.0

    Loss_fn = nn.MSELoss()
    CCL_Loss_fn = NTXentLoss(batch_size=batch_size, temperature=temperature,
                             use_cosine_similarity=True)
    model_optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    GTH_optimizer = optim.Adam(GlobalTemporalHead.parameters(), lr=2*lr, weight_decay=5e-4)
    LTH_optimizer = optim.Adam(LocalTemporalHead.parameters(), lr=2*lr, weight_decay=5e-4)

    for i in range(epochs):
        model.train()
        GlobalTemporalHead.train()
        LocalTemporalHead.train()
        FULL_dataloader = enumerate(zip(train_dataloader, auxi_dataloader))

        Y_true1, Y_pred1, Y_true2, Y_pred2 = [], [], [], []
        Epoch_Loss = []
        Epoch_MSELoss, Epoch_DGLoss, Epoch_GLoss, Epoch_LLoss = [], [], [], []

        d_intra = 0
        d_inter = 0

        if D_INTRA == 0 and D_INTER == 0 and MU == 0:
            MU = 0.5
        else:
            D_INTRA = D_INTRA / len(train_dataloader)
            D_INTER = D_INTER / len(auxi_dataloader)
            MU = 1 - D_INTRA / (D_INTRA + D_INTER)

        for idx, ((data_x, data_y), (data_ax, data_ay)) in FULL_dataloader:
            data_x, data_y = data_x.to(device), data_y.to(device)
            data_ax, data_ay = data_ax.to(device), data_ay.to(device)

            model_optimizer.zero_grad()
            GTH_optimizer.zero_grad()
            LTH_optimizer.zero_grad()

            data_ax, data_ay = CoTmixup(Mix_ratio=Lambda, Temp_shift=temp_shift,
                                        Main_X=data_x, Main_Y=data_y,
                                        Auxi_X=data_ax, Auxi_Y=data_ay)

            Lstm_x1, Output1 = model(data_x)
            Lstm_x2, Output2 = model(data_ax)

            MSE_Loss1 = Loss_fn(Output1, data_y[:, -1, :])
            MSE_Loss2 = Loss_fn(Output2, data_ay[:, -1, :])
            MSE_Loss = Lambda * MSE_Loss1 + (1 - Lambda) * MSE_Loss2

            DG_Loss = 0
            for n in range(batch_size):
                for m in range(n+1, batch_size):
                    DG_Loss_n = DG_fn1(Lstm_x1[n], Lstm_x1[m])
                    DG_Loss += DG_Loss_n

            DG_Loss1 = DG_Loss / ((batch_size * (batch_size - 1) / 2))
            DG_Loss2 = DG_fn2(Lstm_x1, Lstm_x2)

            DG_Loss = (1 - MU) * DG_Loss1 + MU * DG_Loss2

            GlobalLoss1, G_temp_feat1 = GlobalTemporalHead(Lstm_x1, Lstm_x2)
            GlobalLoss2, G_temp_feat2 = GlobalTemporalHead(Lstm_x2, Lstm_x1)
            LocalLoss1, L_temp_feat1 = LocalTemporalHead(Lstm_x1, Lstm_x2)
            LocalLoss2, L_temp_feat2 = LocalTemporalHead(Lstm_x2, Lstm_x1)

            G_ContCL_Loss = CCL_Loss_fn(G_temp_feat1, G_temp_feat2)
            L_ContCL_Loss = 0
            for k in range(L_temp_feat1.shape[0]):
                L_ContCL_Loss_i = CCL_Loss_fn(L_temp_feat1[k], L_temp_feat2[k])
                L_ContCL_Loss += L_ContCL_Loss_i
            L_ContCL_Loss /= L_temp_feat1.shape[0]

            GlobalLoss = (GlobalLoss1 + GlobalLoss2) + Gamma1 * G_ContCL_Loss
            LocalLoss = (LocalLoss1 + LocalLoss2) + Gamma2 * L_ContCL_Loss

            Loss = MSE_Loss + Omega * DG_Loss + Alpha * GlobalLoss + Beta * LocalLoss

            Loss.backward()
            model_optimizer.step()
            GTH_optimizer.step()
            LTH_optimizer.step()

            Y_pred1 += list(Output1.detach().cpu().numpy())
            Y_true1 += list(data_y[: ,-1, :].detach().cpu().numpy())

            Epoch_Loss.append(Loss.item())
            Epoch_MSELoss.append(MSE_Loss.item())
            Epoch_DGLoss.append(DG_Loss.item())
            Epoch_GLoss.append(GlobalLoss.item())
            Epoch_LLoss.append(LocalLoss.item())

        train_mse = mean_squared_error(Y_true1, Y_pred1)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(Y_true1, Y_pred1)
        train_r2 = r2_score(Y_true1, Y_pred1)

        train_Loss_i, tMSE_Loss_i, tDG_Loss_i, tG_Loss_i, tL_Loss_i = np.mean(Epoch_Loss), \
                                                                      np.mean(Epoch_MSELoss), \
                                                                      np.mean(Epoch_DGLoss), \
                                                                      np.mean(Epoch_GLoss), \
                                                                      np.mean(Epoch_LLoss)

        train_Losses.append(train_Loss_i)
        tMSE_Losses.append(tMSE_Loss_i)
        tDG_Losses.append(tDG_Loss_i)
        tG_Losses.append(tG_Loss_i)
        tL_Losses.append(tL_Loss_i)

        test_auc, test_loss_i = test_DomainGeneralization(model, GlobalTemporalHead,
                                                          LocalTemporalHead, unseen_dataloader)
        test_Metric.append(test_auc)
        test_Losses.append(test_loss_i)

        D_INTRA = np.copy(d_intra).item()
        D_INTER = np.copy(d_inter).item()

        if test_auc[0] < best_mse:
            best_mse = test_auc[0]
            best_epoch = i + 1
            best_auc = test_auc
            torch.save(model, r"./Result/model_files/IMTCDG/IMTCDG3_C{}.pth".format(n_cycle))


        print("Epoch: {}/{}, train loss: {:.5f}\n"
              'train mse: {:.5f}, train rmse: {:.5f}, train mae: {:.5f}, train r2: {:.5f}\n'
              'test mse: {:.5f},  test rmse: {:.5f},  test mae: {:.5f},  test r2: {:.5f}\n'
              .format(i + 1, epochs, train_Loss_i,
                      train_mse, train_rmse, train_mae, train_r2,
                      test_auc[0], test_auc[1], test_auc[2], test_auc[3]))

    # length = np.arange(len(tMSE_Losses))
    # plt.plot(length, tMSE_Losses, label='train_loss')
    # plt.plot(length, test_Losses, label='test_loss')
    # plt.xlabel("epochs")
    # plt.ylabel("loss")
    # plt.legend(loc='upper right')
    # plt.savefig(r'./Result/pictures/Loss_Curve.jpg')
    # plt.show()

    return train_Losses, tMSE_Losses, test_Losses, test_Metric, best_auc

def test_DomainGeneralization(model, GlobalTemporalHead, LocalTemporalHead, unseen_dataloader):
    Loss_fn = nn.MSELoss()
    Y_true, Y_pred = [], []
    Metric_List = []
    model.eval()
    GlobalTemporalHead.eval()
    LocalTemporalHead.eval()

    with torch.no_grad():
        Epoch_Loss = []
        for idx, (data_x, data_y) in enumerate(unseen_dataloader):
            data_x, data_y = data_x.to(device), data_y.to(device)
            _, Output = model(data_x)

            Loss = Loss_fn(Output, data_y[:, -1, :])
            Epoch_Loss.append(Loss.item())

            Y_pred += list(Output.detach().cpu().numpy())
            Y_true += list(data_y[:, -1, :].detach().cpu().numpy())

        test_Loss = np.mean(Epoch_Loss)

        test_mse = mean_squared_error(Y_true, Y_pred)
        test_rmse= np.sqrt(test_mse)
        test_mae = mean_absolute_error(Y_true, Y_pred)
        test_r2 = r2_score(Y_true, Y_pred)

        Metric_List.append(test_mse)
        Metric_List.append(test_rmse)
        Metric_List.append(test_mae)
        Metric_List.append(test_r2)

    return Metric_List, test_Loss

if __name__ == "__main__":
    #
    datapath = f"./TimeSeriesData/.csv"
    auxi_datapath = f"./TimeSeriesData/.csv"
    unseen_datapath = f"./TimeSeriesData/.csv"
    file_out = open(r"./Result/train_log/TemporalDG/IMTCDG3_log.txt", "a")
    #
    lr = 0.001
    seq_len = 25
    batch_size = 32
    #
    num_diretion = 1
    hidden_size = 64
    num_layer = 2
    num_inputs = 25
    num_channels = [16, 32, 64]
    #
    Lambda, mu, Gamma1, Gamma2, Omega, Alpha, Beta = 0.90, 0.5, 0.5, 0.5, 0.1, 0.0005, 0.0005
    temp_shift = 10
    temperature = 0.2
    epochs = 100
    time_step = 15

    run_time = 5
    best_auc_all = np.zeros((run_time, 4))
    time_cost_all = np.zeros((run_time, 1))

    file_out.write("超参数设置--lr:{}, batch_size:{}, Lambda:{}\n".format(lr,
                                                                        batch_size,
                                                                        Omega))
    file_out.flush()
    print("超参数设置--lr:{}, batch_size:{}, Lambda:{}\n".format(lr,
                                                                batch_size,
                                                                Omega))

    for i in range(run_time):
        print("************************* The {}-th time training... *************************".format(i + 1))
        start_time = time.time()

        #
        train_Dataloader, auxi_Dataloader, _, unseen_Dataloader = Dataset_setting(datapath=datapath,
                                                                                  auxi_datapath=auxi_datapath,
                                                                                  unseen_datapath=unseen_datapath,
                                                                                  seq_len=seq_len,
                                                                                  batch_size=batch_size)
        #
        model = A_CNN_TPA_LSTM_2(feature_dim=num_inputs, channel_List=num_channels, seq_len=seq_len,
                                 hidden_size=hidden_size, num_layers=num_layer, num_direction=num_diretion)
        GlobalTemporalHead = GlobalTemporalCL(time_step=time_step)
        LocalTemporalHead = LocalTemporalCL(time_step=time_step)
        DG_fn_List= [AdvSKM(Units_1=hidden_size, Units_2=hidden_size) for i in range(2)]

        train_Loss, train_MSE, test_Loss, test_Metric, best_auc = train_DomainGeneralization(model=model,
                                                                  GlobalTemporalHead=GlobalTemporalHead,
                                                                  LocalTemporalHead=LocalTemporalHead,
                                                                  train_dataloader=train_Dataloader,
                                                                  auxi_dataloader=auxi_Dataloader,
                                                                  unseen_dataloader=unseen_Dataloader,
                                                                  Lambda=Lambda, Gamma1=Gamma1, Gamma2=Gamma2,
                                                                  Alpha=Alpha, Beta=Beta, Omega=Omega,
                                                                  temp_shift=temp_shift, temperature=temperature,
                                                                  DG_fn=DG_fn_List, batch_size=batch_size,
                                                                  epochs=epochs, lr=lr, n_cycle=i+1)

        train_Loss, train_MSE, test_Loss, test_Metric = (np.array(train_Loss).reshape(-1, 1),
                                                         np.array(train_MSE).reshape(-1, 1),
                                                         np.array(test_Loss).reshape(-1, 1),
                                                         np.array(test_Metric).reshape(-1, 4))

        current_date = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
        Loss_matrix = np.hstack((train_Loss, train_MSE, test_Loss, test_Metric))
        Loss_DF = pd.DataFrame(Loss_matrix, columns=["train_Loss", "train_MSE", "test_Loss", "test_mse",
                                                     "test_rmse", "test_mae", "test_r2"])
        csv_Name = "./Result/Loss_curve/IMTCDG/IMTCDG3_" + current_date + "_" + str(i+1) + ".csv"
        Loss_DF.to_csv(csv_Name, index=False, encoding="utf-8")

        end_time = time.time()
        time_cost = end_time - start_time

        for k in range(len(best_auc)):
            best_auc_all[i, k] = best_auc[k]

        time_cost_all[i, 0] = time_cost

        file_out.write("Run time:{}, Recorded mse:{:.5f}, rmse:{:.5f}, mae:{:.5f}, r2:{:.5f}\n".format(i + 1,
                                                                                                   best_auc[0],
                                                                                                   best_auc[1],
                                                                                                   best_auc[2],
                                                                                                   best_auc[3]))
        file_out.flush()

        print("Run time:{}, Recorded mse:{:.5f}, rmse:{:.5f}, mae:{:.5f}, r2:{:.5f}\n".format(i + 1,
                                                                                                  best_auc[0],
                                                                                                  best_auc[1],
                                                                                                  best_auc[2],
                                                                                                  best_auc[3]))

        file_out.write("********************************************************\n")
        file_out.write("\n")
        file_out.flush()

        print("********************************************************\n")
        print("\n")

    mean_best_auc = np.mean(best_auc_all, axis=0)
    std_best_auc = np.std(best_auc_all, axis=0)
    mean_time_cost = np.mean(time_cost_all, axis=0)
    std_time_cost = np.std(time_cost_all, axis=0)

    file_out.write("Best mean recorded mse:{:.5f}, rmse:{:.5f}, mae:{:.5f}, r2:{:.5f}\n".format(mean_best_auc[0],
                                                                                            mean_best_auc[1],
                                                                                            mean_best_auc[2],
                                                                                            mean_best_auc[3]))
    file_out.write("Best std recorded mse:{:.5f}, rmse:{:.5f}, mae:{:.5f}, r2:{:.5f}\n".format(std_best_auc[0],
                                                                                               std_best_auc[1],
                                                                                               std_best_auc[2],
                                                                                               std_best_auc[3]))


    file_out.write("Mean time cost:{:.5f}, std time cost:{:.5f}\n".format(mean_time_cost[0], std_time_cost[0]))
    file_out.flush()

    print("Best mean recorded mse:{:.5f}, rmse:{:.5f}, mae:{:.5f}, r2:{:.5f}\n".format(mean_best_auc[0],
                                                                                       mean_best_auc[1],
                                                                                       mean_best_auc[2],
                                                                                       mean_best_auc[3]))
    print("Best std recorded mse:{:.5f}, rmse:{:.5f}, mae:{:.5f}, r2:{:.5f}\n".format(std_best_auc[0],
                                                                                      std_best_auc[1],
                                                                                      std_best_auc[2],
                                                                                      std_best_auc[3]))

    print("Mean time cost:{:.5f}, std time cost:{:.5f}\n".format(mean_time_cost[0], std_time_cost[0]))
    print("\n********************************************************\n")

