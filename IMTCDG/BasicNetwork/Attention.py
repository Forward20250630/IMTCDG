#
import math
#
import torch
import torch.nn as nn
#
device = "cuda" if torch.cuda.is_available() else "cpu"

class MultiHeadsAttention(nn.Module):
    def __init__(self, Num_Heads=8, Num_Hidden=8, dropout=0.2, Q_size=8, K_size=8, V_size=8, QKV_bias=False):
        super(MultiHeadsAttention, self).__init__()
        assert Num_Hidden % Num_Heads == 0
        # if Num_Hidden % Num_Heads !=0:
        #   raise ValueError("Num_Hidden % Num_Heads !=0, please reset the numbers of the two objects.")
        self.Num_Heads = Num_Heads
        self.dropout = nn.Dropout(p=dropout)
        self.W_Q = nn.Linear(Q_size, Num_Hidden, bias=QKV_bias)
        self.W_K = nn.Linear(K_size, Num_Hidden, bias=QKV_bias)
        self.W_V = nn.Linear(V_size, Num_Hidden, bias=QKV_bias)
        self.W_O = nn.Linear(Num_Hidden, Num_Hidden, bias=QKV_bias)
        self.Scale = torch.sqrt(torch.FloatTensor([Num_Hidden // Num_Heads])).to(device)

    def transpose_Output(self, X, Num_Heads):
        X = X.view(-1, Num_Heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        X = X.reshape(X.shape[0], X.shape[1], -1)
        return X

    def transpose_QKV(self, X, Num_Heads):
        X = X.reshape(X.shape[0], X.shape[1], Num_Heads, -1)
        X = X.permute(0, 2, 1, 3)
        X = X.reshape(-1, X.shape[2], X.shape[3])
        return X

    def forward(self, Q, K, V):
        # print("Q.shape", Q.shape)
        Q = self.transpose_QKV(self.W_Q(Q), self.Num_Heads)
        K = self.transpose_QKV(self.W_K(K), self.Num_Heads)
        V = self.transpose_QKV(self.W_V(V), self.Num_Heads)
        Attention = torch.matmul(Q, K.transpose(1, 2)) / self.Scale
        Attention = torch.softmax(Attention, dim=-1)
        X = torch.matmul(self.dropout(Attention), V)
        X_Concat = self.transpose_Output(X, self.Num_Heads)
        Output = self.W_O(X_Concat)
        return Output

class TemporalPatternAttention(nn.Module):
    def __init__(self, filter_size, filter_num, attn_size, attn_len=1):
        super(TemporalPatternAttention, self).__init__()
        self.filter_size = filter_size  # 1
        self.filter_num = filter_num
        self.feat_size = attn_size - self.filter_size + 1  # hidden_size
        self.conv = nn.Conv2d(1, filter_num, (attn_len, filter_size))
        self.fc1 = nn.Linear(attn_size, filter_num)
        self.fc2 = nn.Linear(attn_size + self.filter_num, attn_size)
        self.bn = nn.BatchNorm1d(self.feat_size)
        self.relu = nn.ReLU()
        # self.bn2 = nn.BatchNorm1d(self.feat_size + self.filter_num)
        self.sigmoid = nn.Sigmoid()

    def forward(self, H, ht):
        # print("H.shape", H.shape)
        _, channels, _, attn_size, = H.size()
        conv_vecs = self.conv(H)
        conv_vecs = conv_vecs.view(-1, self.feat_size, self.filter_num)
        conv_vecs = self.bn(conv_vecs)
        conv_vecs = self.relu(conv_vecs)

        htt = self.fc1(ht)
        htt = htt.view(-1, self.filter_num, 1)
        s = torch.bmm(conv_vecs, htt)
        alpha = self.sigmoid(s)
        v = torch.bmm(conv_vecs.view(-1, self.filter_num, attn_size), alpha).view(-1, self.filter_num) 

        x_concat = torch.cat([ht, v], dim=1)
        # x_concat = self.bn2(x_concat)
        output = self.fc2(x_concat)

        return output



"""残差自注意力机制--residual self-attention"""
class RSA_Block(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(RSA_Block).__init__()


    def forward(self, x):



        return x


if __name__ == "__main__":
    MTHS = MultiHeadsAttention(Num_Heads=4, Num_Hidden=8, Q_size=8, K_size=8, V_size=8)
    MTHS.to(device)
    X = torch.ones((32, 64, 8)).to(device)
    # print(X)
    output = MTHS(X, X, X)
    print(output.shape)
    print(output)