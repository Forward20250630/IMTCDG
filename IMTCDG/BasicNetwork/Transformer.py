import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
device = "cuda" if torch.cuda.is_available() else "cpu"


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        # dim: 100
        self.norm = nn.LayerNorm(dim)
        self.fn = fn # Multi-Heads Attention

    def forward(self, x, **kwargs):
        # print("PreNorm", x.shape)
        return self.fn(self.norm(x), **kwargs)

class LSTM_PerNorm(nn.Module):
    def __init__(self):
        super(LSTM_PerNorm).__init__()


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        # HAR:[dim:100, hidden_dim:64]
        self.net = nn.Sequential(
                                 nn.Linear(dim, hidden_dim),
                                 # nn.BatchNorm1d(hidden_dim),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, dim),
                                 nn.Dropout(dropout)
                                )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
                                    nn.Linear(dim, dim),
                                    nn.Dropout(dropout)
                                   )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, fc_dim, dropout):
        super().__init__()
        # HAR:[dim:100, depth:4, heads:4, mlp_dim:64, dropout:0.1]
        self.layers = nn.ModuleList([])
        # depth:4
        for _ in range(depth):
            self.layers.append(
                 nn.ModuleList([
                                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                                Residual(PreNorm(dim, FeedForward(dim, fc_dim, dropout=dropout)))
                               ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class Seq_Transformer(nn.Module):
    def __init__(self, *, patch_size, dim, depth, heads, fc_dim, channels=1, dropout=0.1):
        super().__init__()
        # HAR:[path_size:128, dim:100, depth:4, heads:4, mlp_dim:64]
        patch_dim = channels * patch_size # HAR:128
        self.patch_to_embedding = nn.Linear(patch_dim, dim) # HAR:[128, 100]
        self.c_token = nn.Parameter(torch.randn(1, 1, dim)) # HAR:[1, 1, 100]
        self.transformer = Transformer(dim, depth, heads, fc_dim, dropout) #HAR:[100, 4, 4, 64, 0.1]
        self.to_c_token = nn.Identity()


    def forward(self, forward_seq):
        x = self.patch_to_embedding(forward_seq)
        b, n, _ = x.shape
        c_tokens = repeat(self.c_token, '() n d -> b n d', b=b) # HAR:shape[128, 1, 100]
        x = torch.cat((c_tokens, x), dim=1) # HAR:shape[128, 7, 100]
        x = self.transformer(x)
        c_t = self.to_c_token(x[:, 0])
        return c_t

class LSTM_Transformer(nn.Module):
    def __init__(self, *, patch_size, dim, depth, heads, fc_dim, channels=1, dropout=0.1):
        super(LSTM_Transformer, self).__init__()
        patch_dim = channels * patch_size # HAR:128
        # self.patch_to_embedding = nn.Linear(patch_dim, dim) # HAR:[128, 100]
        self.num_direction = 1
        self.num_layers = 1
        self.hidden_size = dim
        self.temporal_embdding = nn.LSTM(input_size=patch_dim, hidden_size=self.hidden_size,
                                         num_layers=1, batch_first=True)
        self.c_token = nn.Parameter(torch.randn(1, 1, dim)) # HAR:[1, 1, 100]
        self.transformer = Transformer(dim, depth, heads, fc_dim, dropout) #HAR:[100, 4, 4, 64, 0.1]
        self.to_c_token = nn.Identity()

    def forward(self, forward_seq):

        h_0 = torch.randn(self.num_direction * self.num_layers,
                          forward_seq.size(0), self.hidden_size).to(device)
        c_0 = torch.randn(self.num_direction * self.num_layers,
                          forward_seq.size(0), self.hidden_size).to(device)

        # print("forward_seq.shape", forward_seq.shape)
        # x = self.patch_to_embedding(forward_seq)
        x, _ = self.temporal_embdding(forward_seq, (h_0, c_0))
        # print("Lstm_x.", type(x))
        # print("x", x)
        b, n, _ = x.shape
        c_tokens = repeat(self.c_token, '() n d -> b n d', b=b) # HAR:shape[128, 1, 100]
        x = torch.cat((c_tokens, x), dim=1) # HAR:shape[128, 7, 100]
        x = self.transformer(x)
        c_t = self.to_c_token(x[:, 0])
        return c_t