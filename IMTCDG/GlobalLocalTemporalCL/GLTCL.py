#
import torch
import torch.nn as nn
#
from BasicNetwork.Attention import MultiHeadsAttention
from BasicNetwork.Transformer import LSTM_Transformer, Seq_Transformer
device = "cuda" if torch.cuda.is_available() else "cpu"


class NonLinear_Head(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(NonLinear_Head, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(output_dim, output_dim //2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

class GloblTemporalProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, num_layer):
        super(GloblTemporalProjectionHead, self).__init__()
        self.Lstm1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim1,
                             num_layer=num_layer, batch_first=True)
        self.MTHA1 = MultiHeadsAttention(Num_Heads=4, Num_Hidden=64, Q_size=hidden_dim1,
                                         K_size=hidden_dim1, V_size=hidden_dim1)
        self.Lstm2 = nn.LSTM(input_size=hidden_dim1, hidden_size=hidden_dim2,
                             num_layer=num_layer, batch_first=True)
        self.MTHA2 = MultiHeadsAttention(Num_Heads=4, Num_Hidden=64, Q_size=hidden_dim2,
                                         K_size=hidden_dim2, V_size=hidden_dim2)

    def forward(self, x):
        x = self.Lstm1(x)
        x = self.MTHA1(x)
        x = self.Lstm2(x)
        x = self.MTHA2(x)

        return x

class LocalTemporalProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, num_layer):
        super(LocalTemporalProjectionHead).__init__()
        self.Aavgpool = nn.AdaptiveAvgPool1d(64)
        self.Lstm1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim1,
                             num_layer=num_layer, batch_first=True)
        self.MTHA1 = MultiHeadsAttention(Num_Heads=4, Num_Hidden=64, Q_size=hidden_dim1,
                                         K_size=hidden_dim1, V_size=hidden_dim1)
        self.Lstm2 = nn.LSTM(input_size=hidden_dim1, hidden_size=hidden_dim2,
                             num_layer=num_layer, batch_first=True)
        self.MTHA2 = MultiHeadsAttention(Num_Heads=4, Num_Hidden=64, Q_size=hidden_dim2,
                                         K_size=hidden_dim2, V_size=hidden_dim2)

    def forward(self, x):
        x = self.Aavgpool(x)
        x = self.Lstm1(x)
        x = self.MTHA1(x)
        x = self.Lstm2(x)
        x = self.MTHA2(x)

        return x


class GlobalTemporalCL(nn.Module):
    def __init__(self, time_step=8, in_channels=64, hidden_dim=64, type="LSTM"):
        super(GlobalTemporalCL, self).__init__()
        # type->the network type to process data feature of time series, i.e.:LSTM, transformer...
        self.Head_type = type
        self.timestep = time_step
        self.in_channels = in_channels # 64
        self.hidden_dim = hidden_dim
        # self.fc_dim = 32
        self.depth = 2
        self.num_Head = 4
        self.NHidden_dim = 64
        self.NOutput_dim = 32
        self.LSoftmax = nn.LogSoftmax(dim=0)

        if self.Head_type == "LSTM":
            self.GTProjectionHead = LSTM_Transformer(patch_size=self.in_channels, dim=self.hidden_dim,
                                                     depth=self.depth, heads=self.num_Head, fc_dim=32)
        else:
            self.GTProjectionHead = Seq_Transformer(patch_size=self.in_channels, dim=self.hidden_dim,
                                                     depth=self.depth, heads=self.num_Head, fc_dim=32)

        self.MapLayerPool = nn.ModuleList([nn.Linear(self.hidden_dim, self.in_channels)
                                          for i in range(self.timestep)])

        self.NonLinearHead = NonLinear_Head(hidden_dim=self.NHidden_dim, output_dim=self.NOutput_dim)


    #def GlobalTemporalContrastiveLoss(self):
    def forward(self, Main_x, Auxi_x):
        batch_size = Main_x.shape[0]
        seq_len = Main_x.shape[1]

        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(device)
        encoding_samples = torch.empty((self.timestep, batch_size, self.in_channels)).float().to(device)

        for i in range(1, self.timestep + 1):
            encoding_samples[i - 1] = Auxi_x[:, t_samples + i, :].view(batch_size, self.in_channels)
        forward_x = Main_x[:, :t_samples + 1, :]

        Nce_Loss = 0
        IM_Cx = self.GTProjectionHead(forward_x)
        Pred_x = torch.empty((self.timestep, batch_size, self.in_channels)).float().to(device)
        for n in range(self.timestep):
            Fc_n = self.MapLayerPool[n]
            Pred_x[n] = Fc_n(IM_Cx)

        for m in range(self.timestep):
            total = torch.mm(encoding_samples[m], torch.transpose(Pred_x[m], 0, 1))
            Nce_Loss += torch.sum(torch.diag(self.LSoftmax(total)))
        Nce_Loss /= -1. * batch_size * self.timestep
        # print("Global Nce_loss:", Nce_Loss)

        FM_Cx = self.NonLinearHead(IM_Cx)

        return Nce_Loss, FM_Cx

class LocalTemporalCL(nn.Module):
    def __init__(self, time_step=8, AAPool_dim=8, in_channels=1, hidden_dim=64,type="LSTM"):
        super(LocalTemporalCL, self).__init__()
        self.Head_type = type
        self.AAPool_dim = AAPool_dim
        self.timestep = time_step
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        # self.fc_dim = 32
        self.depth = 2
        self.num_Head = 4
        self.NHidden_dim = 64
        self.NOutput_dim = 32
        self.LSoftmax = nn.LogSoftmax(dim=0)

        self.LocalMapLayer = nn.AdaptiveAvgPool1d(self.AAPool_dim)

        if self.Head_type == "LSTM":
            self.LTProjectionHead = LSTM_Transformer(patch_size=self.in_channels, dim=self.hidden_dim,
                                                      depth=self.depth, heads=self.num_Head, fc_dim=32)
        else:
            self.LTProjectionHead = Seq_Transformer(patch_size=self.in_channels, dim=self.hidden_dim,
                                                     depth=self.depth, heads=self.num_Head, fc_dim=32)

        self.MapLayerPool_k = nn.ModuleList([nn.Linear(self.hidden_dim, self.in_channels)
                                            for i in range(self.timestep)])
        self.MapLayerPool = [self.MapLayerPool_k for k in range(self.AAPool_dim)]

        self.NonLinearHead = NonLinear_Head(hidden_dim=self.NHidden_dim, output_dim=self.NOutput_dim)

    def Temporal_CL(self, x_u, x_w, batch_size, num_th):
        x_u, x_w = x_u.unsqueeze(-1), x_w.unsqueeze(-1)
        seq_len = x_w.shape[1]

        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(device)
        encoding_samples = torch.empty((self.timestep, batch_size, self.in_channels)).float().to(device)

        for i in range(1, self.timestep+1):
            encoding_samples[i-1] = x_w[:, t_samples+i, :].view(batch_size, 1)
        forward_x = x_u[:, :t_samples + 1, :]

        Nce_Loss = 0
        IM_Cx = self.LTProjectionHead(forward_x)
        Pred_x = torch.empty((self.timestep, batch_size, self.in_channels)).float().to(device)
        for n in range(self.timestep):
            Fc_n = self.MapLayerPool[num_th][n]
            Pred_x[n] = Fc_n(IM_Cx)

        for m in range(self.timestep):
            total = torch.mm(encoding_samples[m], torch.transpose(Pred_x[m], 0, 1))
            Nce_Loss += torch.sum(torch.diag(self.LSoftmax(total)))
        Nce_Loss /= -1. * batch_size * self.timestep

        FM_Cx = self.NonLinearHead(IM_Cx)

        return Nce_Loss, FM_Cx

    def forward(self, Main_x, Auxi_x):
        batch_size = Main_x.shape[0]

        IM_x = self.LocalMapLayer(Main_x)
        IA_x = self.LocalMapLayer(Auxi_x)

        # fine-grained level contrastive learning
        Nce_Loss = 0
        FM_Cx = torch.zeros(IM_x.shape[2], batch_size, self.NOutput_dim//2).to(device)
        for i in range(IM_x.shape[2]):
            Nce_Loss_i, FM_Cx_i = self.Temporal_CL(IM_x[:, :, i], IA_x[:, :, i], batch_size, i)
            Nce_Loss += Nce_Loss_i
            FM_Cx[i] = FM_Cx_i

        Nce_Loss /= IM_x.shape[2]
        # print("Local Nce_loss:", Nce_Loss)

        return Nce_Loss, FM_Cx