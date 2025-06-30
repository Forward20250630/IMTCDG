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
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.bn = nn.BatchNorm1d(n_outputs)
        self.relu = nn.ReLU()
        self.init_weights()

        self.TCNblock = nn.Sequential(self.conv1,
                                      self.chomp1,
                                      self.relu1,
                                      self.dropout1,
                                      self.conv2,
                                      self.chomp2,
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
        output = self.bn(output1+output2)
        output = self.relu(output)
        return output


class TemporalBlock_A(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding,
                 Num_heads, Num_hidden, QKV_size, dropout=0.2):
        super(TemporalBlock_A, self).__init__()
        # temporal convolution block with attention
        self.conv1 = weight_norm(
                     nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
                     nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.bn = nn.BatchNorm1d(n_outputs)
        self.relu = nn.ReLU()
        self.init_weights()

        self.TCNblock = nn.Sequential(self.conv1,
                                      self.chomp1,
                                      self.relu1,
                                      self.dropout1,
                                      self.conv2,
                                      self.chomp2,
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
        x = self.bn(output1 + output2)
        x = self.relu(x)
        # print(x.shape)
        # x = self.MTHA(x, x, x)
        # print("tun_time_after")
        output = self.maxpool(x)

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
        self.atten_param = [[4, 100, 10],
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
        self.bn1 = nn.BatchNorm1d(self.atten_param[2][2])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(int(self.hidden_size/2), 1)
        self.Regressor = nn.Linear(int(self.hidden_size/2), 1)

        self.Gate1 = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size),
                                   nn.BatchNorm1d(self.hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_size,int(self.hidden_size / 2)),
                                   nn.BatchNorm1d(int(self.hidden_size/2)),
                                   nn.ReLU(),
                                   nn.Linear(int(self.hidden_size/2), 1),
                                   nn.Sigmoid())
        # self.Gate2 = nn.Sequential(nn.Linear(seq_len * 2, seq_len),
        #                            nn.BatchNorm1d(seq_len),
        #                            nn.Sigmoid())

        self.softmax1 = nn.Softmax(dim=0)
        # self.softmax2 = nn.Softmax(dim=-1)


    def forward(self, x, Weight_mat=None,  whether_train=True,
                whether_pretrain=True, whether_main=True):
        x = x.permute(0, 2, 1)
        temp_feat = self.network(x)
        # print("CL.shape", temp_feat.shape)

        x = temp_feat.permute(0, 2, 1)
        # print("CL_shape", x.shape)
        h_0 = torch.randn(self.num_direction * self.num_layers, x.size(0), self.hidden_size).to(device)
        c_0 = torch.randn(self.num_direction * self.num_layers, x.size(0), self.hidden_size).to(device)
        Lstm_x, _ = self.lstm(x, (h_0, c_0))

        # Weight_mat, Distance_mat = None, None
        if whether_pretrain and whether_train:
            m_DGLoss, Weight_mat, Distance_mat = self.forward_pretrain(Lstm_x,
                                                                       whether_main=whether_main)
            # print("pretrain weight:", Weight_mat)
        elif whether_train:
            m_DGLoss, Weight_mat, Distance_mat = self.forward_Boosting(Lstm_x, Weight_mat,
                                                                       whether_main=whether_main)
            # print("train weight:", Weight_mat)
        else:
            m_DGLoss, Weight_mat, Distance_mat = None, None, None

        x = self.fc1(Lstm_x)
        # print("x.shape", x.shape)
        x = self.bn1(x)
        x = self.relu1(x)
        Output = self.fc2(x)
        Output = Output[:, -1, :]

        return Lstm_x, Output, m_DGLoss, Weight_mat, Distance_mat

    def cal_AdvSKM(self, Lstm_x):
        # Domain gernerlization without dynamic weight
        batch_size = Lstm_x.shape[0]
        DG_Loss = torch.zeros((1,)).to(device)

        for i in range(batch_size):
            for j in range(i+1, batch_size):
                DG_Loss_i = self.AdvSKM(Lstm_x[i], Lstm_x[j])
                DG_Loss += DG_Loss_i

        m_DGLoss = DG_Loss / ((batch_size * (batch_size - 1) / 2))

        return m_DGLoss


    def get_Weight(self, Lstm_x, batch_size):
        Weight_mat = torch.zeros(batch_size, batch_size).to(device)
        for i in range(batch_size):
            for j in range(i+1, batch_size):
                # print("Lstm_i.shape", Lstm_x[i].shape)
                x = torch.cat((Lstm_x[i], Lstm_x[j]), -1)
                x = x.view(Lstm_x.shape[1], -1)
                weight = self.Gate1(x)
                weight = torch.mean(weight, dim=0)
                weight = self.softmax1(weight).squeeze()
                # print("weight.shape", weight.shape)
                Weight_mat[i][j] = weight

        return Weight_mat

    def Update_Weight_Boosting(self, Weight_mat, Distance_Old, Distance_New,  seq_len):
        Epsilon = 1e-12
        Distance_O = Distance_Old.detach()
        Distance_N = Distance_New.detach()
        index = Distance_N > Distance_O + Epsilon
        Weight_mat[index] = Weight_mat[index] * (1 + torch.sigmoid(Distance_N[index] - Distance_O[index]))
        Weight_Norm = torch.norm(Weight_mat, dim=1, p=1)
        Weight_mat = Weight_mat / Weight_Norm.t().unsqueeze(1).repeat(1, seq_len)

        return Weight_mat

    def forward_pretrain(self, Lstm_x, whether_main = True):

        if whether_main:
            batch_size = Lstm_x.shape[0]
            # W_size = batch_size * (batch_size - 1) / 2
            # Weight_mat = torch.zeros(W_size, 1)
            Distance_mat = torch.zeros(batch_size, batch_size).to(device)
            Weight_mat = self.get_Weight(Lstm_x, batch_size)

            DG_Loss = torch.zeros((1,)).to(device)
            for i in range(batch_size):
                for j in range(i + 1, batch_size):
                    DG_Loss_i = Weight_mat[i][j] * self.AdvSKM(Lstm_x[i], Lstm_x[j])
                    print("DG loss i:", DG_Loss_i)
                    Distance_mat[i][j] = DG_Loss_i
                    DG_Loss += DG_Loss_i
            m_DGLoss = DG_Loss / ((batch_size * (batch_size - 1) / 2))
            return m_DGLoss, Weight_mat, Distance_mat
        else:
            return None, None, None

    def forward_Boosting(self, Lstm_x, Weight_mat=None, whether_main=True):

        if whether_main:
            batch_size = Lstm_x.shape[0]
            Distance_mat = torch.zeros(batch_size, batch_size).to(device)

            DG_Loss = torch.zeros((1,)).to(device)
            for i in range(batch_size):
                for j in range(i + 1, batch_size):
                    DG_Loss_i = Weight_mat[i][j] * self.AdvSKM(Lstm_x[i], Lstm_x[j])
                    Distance_mat[i][j] = DG_Loss_i
                    DG_Loss += DG_Loss_i
            m_DGLoss = DG_Loss / ((batch_size * (batch_size - 1) / 2))
            return m_DGLoss, Weight_mat, Distance_mat
        else:
            return None, None, None