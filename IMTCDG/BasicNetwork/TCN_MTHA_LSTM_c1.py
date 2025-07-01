#
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
#
from BasicNetwork.Attention import MultiHeadsAttention
from TemporalDistance.AdvSKM import AdvSKM
device = "cuda" if torch.cuda.is_available() else "cpu"


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.bn3 = nn.BatchNorm1d(n_outputs)
        self.relu = nn.ReLU()
        self.init_weights()

        self.TCNblock = nn.Sequential(self.conv1,
                                      self.chomp1,
                                      self.bn1,
                                      self.relu1,
                                      self.dropout1,
                                      self.conv2,
                                      self.chomp2,
                                      self.bn2,
                                      self.relu2,
                                      self.dropout2)

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        output1 = self.TCNblock(x)
        output2 = x if self.downsample is None else self.downsample(x)
        output = self.bn3(output1 + output2)
        return self.relu(output)


class TemporalBlock_A(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding,
                 Num_heads, Num_hidden, QKV_size, dropout=0.2):
        super(TemporalBlock_A, self).__init__()
        # temporal convolution block with attention
        self.conv1 = weight_norm(
                     nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
                     nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.bn3 = nn.BatchNorm1d(n_outputs)
        self.relu = nn.ReLU()
        self.init_weights()

        self.TCNblock = nn.Sequential(self.conv1,
                                      self.chomp1,
                                      self.bn1,
                                      self.relu1,
                                      self.dropout1,
                                      self.conv2,
                                      self.chomp2,
                                      self.bn2,
                                      self.relu2,
                                      self.dropout2)

        self.MTHA = MultiHeadsAttention(Num_Heads=Num_heads, Num_Hidden=Num_hidden, dropout=dropout,
                                        Q_size=QKV_size, K_size=QKV_size, V_size=QKV_size)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)


    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # print("run_time")
        output1 = self.TCNblock(x)
        output2 = x if self.downsample is None else self.downsample(x)
        output = output1 + output2
        # print("output.shape", output.shape)
        output = self.MTHA(output, output, output)
        output = self.bn3(output)
        # print("output1.shape", output1.shape)
        # print("output2.shape", output2.shape)
        output = self.relu(output)
        # print(x.shape)
        output = self.maxpool(output)

        return output


class TemporalConvNet(nn.Module):
    def __init__(self, seq_len, num_direction, hidden_size, num_layers, num_inputs, num_channels,
                 kernel_size=2, dropout=0.2, attention=True):
        super(TemporalConvNet, self).__init__()
        self.seq_len = seq_len
        self.num_direction = num_direction
        self.hidden_size = hidden_size
        self.num_layers = 2
        self.AdvSKM = AdvSKM()
        self.layers = []
        # self.atten_param = [[4, 100, 25],
        #                     [4, 100, 100],
        #                     [4, 100, 100]]
        self.atten_param = [[4, 100, 25],
                            [4, 100, 50],
                            [4, 100, 50]]
        self.num_levels = len(num_channels)
        for i in range(self.num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            if attention:
                self.layers += [TemporalBlock_A(in_channels, out_channels, kernel_size, stride=1,
                                                dilation=dilation_size, padding=(kernel_size - 1) * dilation_size,
                                                dropout=dropout, Num_heads=self.atten_param[i][0],
                                                Num_hidden=self.atten_param[i][1], QKV_size=self.atten_param[i][2])]
            else:
                self.layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                              dilation=dilation_size, padding=(kernel_size - 1) * dilation_size,
                                              dropout=dropout)]

        self.network = nn.Sequential(*self.layers)

        # Lstm + Fc regression
        self.lstm = nn.LSTM(input_size=num_channels[-1], hidden_size=self.hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_size, int(self.hidden_size/2))
        # self.bn1 = nn.BatchNorm1d(int(hidden_size/2))
        self.bn1 = nn.BatchNorm1d(self.atten_param[-1][-1])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(int(self.hidden_size/2), 1)
        self.Regressor = nn.Linear(int(self.hidden_size/2), 1)


    def forward(self, x):
        x = x.permute(0, 2, 1)
        temp_feat = self.network(x)
        # print("CL.shape", temp_feat.shape)

        x = temp_feat.permute(0, 2, 1)
        # print("CL_shape", x.shape)
        h_0 = torch.randn(self.num_direction * self.num_layers, x.size(0), self.hidden_size).to(device)
        c_0 = torch.randn(self.num_direction * self.num_layers, x.size(0), self.hidden_size).to(device)
        Lstm_x, _ = self.lstm(x, (h_0, c_0))
        # print("Lstm_x.shape", Lstm_x.shape)

        x = self.fc1(Lstm_x)
        x = self.bn1(x)
        x = self.relu1(x)
        Output = self.fc2(x)
        Output = Output[:, -1, :]

        return Lstm_x, Output
