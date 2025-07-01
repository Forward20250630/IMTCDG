#
import torch
import torch.nn as nn
import torch.optim as optim
#
from TemporalDistance.distance import MMD_loss
device = "cuda" if torch.cuda.is_available() else "cpu"

class Cosine_Activation(nn.Module):
    # cosine activation function
    def __init__(self):
        super(Cosine_Activation, self).__init__()

    def forward(self, X):
        return torch.cos(X)


class SpectralMapBlock(nn.Module):
    def __init__(self, Units):
        super(SpectralMapBlock, self).__init__()
        self.Units = Units
        self.C = 25
        self.fc1 = nn.Linear(self.Units, int(self.Units//2))
        self.fc12 = nn.Linear(int(self.Units//2), int(self.Units//2))
        self.fc2 = nn.Linear(self.Units, int(self.Units//2))
        self.fc22 = nn.Linear(int(self.Units//2), int(self.Units//2))
        self.bn1 = nn.BatchNorm1d(self.C)
        self.bn2 = nn.BatchNorm1d(self.C)
        self.cos = Cosine_Activation()
        self.relu = nn.ReLU()

    def forward(self, X):
        X1 = self.fc1(X)
        X12 = self.fc12(X1)
        X2 = self.fc2(X)
        X22 = self.fc22(X2)
        X1 = self.bn1(X12)
        X2 = self.bn2(X22)
        Output1 = self.cos(X1)
        Output2 = self.relu(X2)
        Output = torch.cat((Output1, Output2), dim=-1)
        return Output

#method 1
class DSKN1(nn.Module):
    def __init__(self, Units_1=64, Units_2=32):
        super(DSKN1, self).__init__()
        self.block1 = SpectralMapBlock(Units=Units_1)
        self.block2 = SpectralMapBlock(Units=Units_2)

    def forward(self, X):
        X = self.block1(X)
        X = self.block2(X)
        return X

# method 2
class DSKN2(nn.Module):
    def __init__(self, Units):
        super(DSKN2, self).__init__()
        self.Uints=Units
        self.cos=Cosine_Activation()
        self.Spec_layer = nn.Sequential(
            nn.Linear(self.Uints, int(self.Uints//2)),
            nn.Linear(int(self.Uints), int(self.Uints//2)),
            nn.BatchNorm1d(int(self.Uints//2)),
            nn.ReLU(),
            nn.Linear(int(self.Uints//2), int(self.Uints//4)),
            nn.Linear(int(self.Uints//4), int(self.Uints//4)),
            nn.BatchNorm1d(int(self.Uints//4)),
            nn.ReLU()
        )
        self.Arc_layer = nn.Sequential(
            nn.Linear(self.Uints, int(self.Uints//2)),
            nn.Linear(int(self.Uints), int(self.Uints//2)),
            nn.BatchNorm1d(int(self.Uints//2)),
            self.cos,
            nn.Linear(int(self.Uints//2), int(self.Uints//4)),
            nn.Linear(int(self.Uints//4), int(self.Uints//4)),
            nn.BatchNorm1d(int(self.Uints//4)),
            self.cos
        )

    def forward(self, X):
        Spec_X = self.Spec_layer(X)
        Arc_X = self.Arc_layer(X)
        Output = torch.cat((Spec_X, Arc_X))
        return Output


class AdvSKM_in(nn.Module):
    def __init__(self, Units_1=64, Units_2=64, Adv_lr=0.001, Adv_step=1, kernel_type="rbf"):
        super(AdvSKM_in, self).__init__()
        self.Adv_step = Adv_step
        self.DSKN1 = DSKN1(Units_1=Units_1, Units_2=Units_2).to(device)
        # self.DSKN2 = DSKN2(Units=256)
        self.mmd_Loss = MMD_loss(kernel_type=kernel_type)
        self.Adv_Optimizer = optim.Adam(self.DSKN1.parameters(), lr=Adv_lr, weight_decay=2e-5, betas=(0.5, 0.99))

    # def Compute_Loss(self, emb_fX, emb_sX):
    #     Adv_mmd_Loss = -self.mmd_Loss(emb_fX, emb_sX)
    #     return Adv_mmd_Loss

    def Update_AdvSKM(self, fX, sX):
        emb_fX = self.DSKN1(fX)
        emb_sX = self.DSKN1(sX)
        mmd_Loss = -self.mmd_Loss(emb_fX.reshape(emb_fX.shape[0], -1),
                                  emb_sX.reshape(emb_sX.shape[0], -1))
        mmd_Loss.requires_grad = True
        self.Adv_Optimizer.zero_grad()
        mmd_Loss.backward()
        self.Adv_Optimizer.step()
        return None

    def forward(self, fX, sX, training_Flag=True):
        # print("fx.shape", fX.shape)
        # AdvSKM_Loss = 0
        for i in range(self.Adv_step):
            if training_Flag == True:
                self.Update_AdvSKM(fX, sX)
        emb_fX = self.DSKN1(fX)
        emb_sX = self.DSKN1(sX)
        AdvSKM_Loss = self.mmd_Loss(emb_fX.reshape(emb_fX.shape[0], -1),
                                    emb_sX.reshape(emb_sX.shape[0], -1))
        AdvSKM_Loss.requires_grad = True

        return AdvSKM_Loss