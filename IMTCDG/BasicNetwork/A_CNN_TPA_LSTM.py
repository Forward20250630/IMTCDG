#
import torch
import torch.nn as nn
#
from BasicNetwork.Attention import MultiHeadsAttention, TemporalPatternAttention
device = "cuda" if torch.cuda.is_available() else "cpu"

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=2, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=2, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(8)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=2, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(8)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.bn4(x)
        x = self.relu4(x)
        Output = self.fc2(x)

        return Output


class CNN_LSTM(nn.Module):
    def __init__(self, feature_dim, seq_len, hidden_size, num_layers, num_direction):
        super(CNN_LSTM, self).__init__()
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_direction = num_direction

        self.conv1 = nn.Conv1d(in_channels=self.feature_dim, out_channels=32, kernel_size=2, padding=1, stride=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.MTHA1 = MultiHeadsAttention(Num_Heads=4, Num_Hidden=100, Q_size=13, K_size=13, V_size=13)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, padding=1, stride=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.MTHA2 = MultiHeadsAttention(Num_Heads=4, Num_Hidden=100, Q_size=26, K_size=26, V_size=26)
        # self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2, padding=1, stride=2)
        # self.bn3 = nn.BatchNorm1d(32)
        # self.relu3 = nn.ReLU()
        # self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.lstm = nn.LSTM(input_size=64, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_size, int(self.hidden_size/2))
        self.bn4 = nn.BatchNorm1d(50)
        # self.bn4 = nn.BatchNorm1d(int(self.hidden_size/2))
        # print(self.hidden_size/2)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(int(self.hidden_size/2), 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # print("x1.shape", x.shape)
        x = self.MTHA1(x, x, x)
        x = self.maxpool1(x)
        # print("MTHA", x.shape)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        # print("x2.shape", x.shape)
        x = self.MTHA2(x, x, x)
        x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.relu3(x)
        # x = self.maxpool3(x)

        x = x.permute(0, 2, 1)
        h_0 = torch.randn(self.num_direction * self.num_layers, x.size(0), self.hidden_size).to(device)
        c_0 = torch.randn(self.num_direction * self.num_layers, x.size(0), self.hidden_size).to(device)

        t_x, _ = self.lstm(x, (h_0, c_0))
        x = self.fc1(t_x)
        x = self.bn4(x)
        x = self.relu4(x)
        Output = self.fc2(x)
        Output = Output[:, -1, :]

        return t_x, Output


class A_CNN_TPA_LSTM(nn.Module):
    def __init__(self, feature_dim, channel_List, seq_len, hidden_size, num_layers, num_direction):
        super(A_CNN_TPA_LSTM, self).__init__()
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.channel_List = channel_List
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_direction = num_direction
        # self.MTHA_param = [13, 4]
        # self.filter_num = 4
        # self.filter_size = 1

        self.conv1 = nn.Conv1d(in_channels=self.feature_dim, out_channels=self.channel_List[0],
                               kernel_size=2, padding=1, stride=1)
        # self.MTHA1 = MultiHeadsAttention(Num_Heads=1, Num_Hidden=self.MTHA_param[0], Q_size=self.MTHA_param[0],
        #                                  K_size=self.MTHA_param[0], V_size=self.MTHA_param[0])
        self.bn1 = nn.BatchNorm1d(self.channel_List[0])
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.conv2 = nn.Conv1d(in_channels=self.channel_List[0], out_channels=self.channel_List[1],
                               kernel_size=2, padding=1, stride=1)
        # self.MTHA2 = MultiHeadsAttention(Num_Heads=2, Num_Hidden=self.MTHA_param[1], Q_size=self.MTHA_param[1],
        #                                  K_size=self.MTHA_param[1], V_size=self.MTHA_param[1])
        self.bn2 = nn.BatchNorm1d(self.channel_List[1])
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.conv3 = nn.Conv1d(in_channels=self.channel_List[1], out_channels=self.channel_List[2],
                               kernel_size=2, padding=1, stride=1)
        # self.MTHA3 = MultiHeadsAttention(Num_Heads=2, Num_Hidden=self.MTHA_param[2], Q_size=self.MTHA_param[2],
        #                                  K_size=self.MTHA_param[2], V_size=self.MTHA_param[2])
        self.bn3 = nn.BatchNorm1d(self.channel_List[2])
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=1)

        self.lstm = nn.LSTM(input_size=self.channel_List[-1], hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)
        # self.TPA = TemporalPatternAttention(self.filter_size, self.filter_num, hidden_size)
        self.fc1 = nn.Linear(self.hidden_size, int(self.hidden_size/2))
        self.bn4 = nn.BatchNorm1d(25)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(int(self.hidden_size/2), 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        # print("a1", x.shape)
        # x = self.MTHA1(x, x, x)
        # print("1.after :", x.shape)
        x = self.bn1(x)
        x = self.relu1(x)
        # print("x1.shape", x.shape)
        x = self.maxpool1(x)
        x = self.conv2(x)
        # print("a2", x.shape)
        # x = self.MTHA2(x, x, x)
        # print("2.after :", x.shape)
        x = self.bn2(x)
        x = self.relu2(x)
        # print("x2.shape", x.shape)
        x = self.maxpool2(x)
        x = self.conv3(x)
        # print("a3", x.shape)
        # x = self.MTHA3(x, x, x)
        # print("3.after :", x.shape)
        x = self.bn3(x)
        x = self.relu3(x)
        # print("x2.shape", x.shape)
        x = self.maxpool3(x)

        x = x.permute(0, 2, 1)
        # print("x", x.shape)
        # batch_size, seq_len, _ = c_x.shape
        # print("seq_len", seq_len)
        # H = torch.zeros(batch_size, seq_len-1, self.channel_List[-1]).to(device)
        h_0 = torch.randn(self.num_direction * self.num_layers, x.size(0), self.hidden_size).to(device)
        c_0 = torch.randn(self.num_direction * self.num_layers, x.size(0), self.hidden_size).to(device)
        Lstm_x, _ = self.lstm(x, (h_0, c_0))

        # for t in range(seq_len):
            # print("x.shape", x.shape)
            # xt = c_x[:, t, :].view(batch_size, 1, -1)
            # Lstm_x, (h_0, c_0) = self.lstm(xt, (h_0, c_0))
            # htt = h_0[-1, :, :]
            # if t != seq_len-1:
            #     H[:, t, :] = htt
        # H = self.bn5(H)
        # H = self.relu5(H)
        # H = H.view(batch_size, 1, seq_len-1, self.channel_List[-1])
        # x_H = self.TPA(H, htt)
        # print("x_H", x_H.shape)

        x = self.fc1(Lstm_x)
        # x = self.fc1(x_H)
        # x = self.bn4(x)
        x = self.relu4(x)
        Output = self.fc2(x)
        # print("Output", Output.shape)
        Output = Output[:, -1, :]
        # print("output.shape", output.shape)

        # return Lstm_x, Output
        return Lstm_x, Output

class A_CNN_TPA_LSTM_2(nn.Module):
    def __init__(self, feature_dim, channel_List, seq_len, hidden_size, num_layers, num_direction):
        super(A_CNN_TPA_LSTM_2, self).__init__()
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.channel_List = channel_List
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_direction = num_direction
        self.MTHA_param = [25, 25]


        self.conv1 = nn.Conv1d(in_channels=self.feature_dim, out_channels=self.channel_List[0],
                               kernel_size=2, padding=1, stride=1)
        self.MTHA1 = MultiHeadsAttention(Num_Heads=5, Num_Hidden=self.MTHA_param[0], Q_size=self.MTHA_param[0],
                                         K_size=self.MTHA_param[0], V_size=self.MTHA_param[0])
        self.bn1 = nn.BatchNorm1d(self.channel_List[0])
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.conv2 = nn.Conv1d(in_channels=self.channel_List[0], out_channels=self.channel_List[1],
                               kernel_size=2, padding=1, stride=1)
        self.MTHA2 = MultiHeadsAttention(Num_Heads=5, Num_Hidden=self.MTHA_param[1], Q_size=self.MTHA_param[1],
                                         K_size=self.MTHA_param[1], V_size=self.MTHA_param[1])
        self.bn2 = nn.BatchNorm1d(self.channel_List[1])
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.conv3 = nn.Conv1d(in_channels=self.channel_List[1], out_channels=self.channel_List[2],
                               kernel_size=2, padding=1, stride=1)
        # self.MTHA3 = MultiHeadsAttention(Num_Heads=5, Num_Hidden=self.MTHA_param[2], Q_size=self.MTHA_param[2],
        #                                  K_size=self.MTHA_param[2], V_size=self.MTHA_param[2])
        self.bn3 = nn.BatchNorm1d(self.channel_List[2])
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=1)

        self.lstm = nn.LSTM(input_size=sum(self.channel_List), hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)
        # self.TPA = TemporalPatternAttention(self.filter_size, self.filter_num, hidden_size)
        self.fc1 = nn.Linear(self.hidden_size, int(self.hidden_size/2))
        self.bn4 = nn.BatchNorm1d(25)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(int(self.hidden_size/2), 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        # print("a1", x.shape)
        # print("conv1.after :", x.shape)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # print("max1.after :", x.shape)
        MTHA_x1 = self.MTHA1(x, x, x)
        x = self.conv2(x)
        # print("a2", x.shape)
        # print("conv2.after :", x.shape)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        # print("max2.after :", x.shape)
        MTHA_x2 = self.MTHA2(x, x, x)
        x = self.conv3(x)
        # print("a3", x.shape)
        # print("conv3.after :", x.shape)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        # print("max2.after :", x.shape)

        x = torch.cat((MTHA_x1.permute(0, 2, 1),
                       MTHA_x2.permute(0, 2, 1),
                       x.permute(0, 2, 1)), dim=-1)
        # print("cat.after", x.shape)
        # batch_size, seq_len, _ = c_x.shape
        # print("seq_len", seq_len)
        # H = torch.zeros(batch_size, seq_len-1, self.channel_List[-1]).to(device)
        h_0 = torch.randn(self.num_direction * self.num_layers, x.size(0), self.hidden_size).to(device)
        c_0 = torch.randn(self.num_direction * self.num_layers, x.size(0), self.hidden_size).to(device)
        Lstm_x, _ = self.lstm(x, (h_0, c_0))
        # print("Lstm_x", Lstm_x.shape)

        # for t in range(seq_len):
            # print("x.shape", x.shape)
            # xt = c_x[:, t, :].view(batch_size, 1, -1)
            # Lstm_x, (h_0, c_0) = self.lstm(xt, (h_0, c_0))
            # htt = h_0[-1, :, :]
            # if t != seq_len-1:
            #     H[:, t, :] = htt
        # H = self.bn5(H)
        # H = self.relu5(H)
        # H = H.view(batch_size, 1, seq_len-1, self.channel_List[-1])
        # x_H = self.TPA(H, htt)
        # print("x_H", x_H.shape)

        x = self.fc1(Lstm_x)
        # x = self.fc1(x_H)
        # x = self.bn4(x)
        x = self.relu4(x)
        Output = self.fc2(x)
        # print("Output", Output.shape)
        Output = Output[:, -1, :]
        # print("output.shape", output.shape)

        # return Lstm_x, Output
        return Lstm_x, Output