import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

"""CORAL"""
def CORAL(source, target):
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)

    # source covariance
    tmp_s = torch.ones((1, ns)).cuda() @ source
    cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)

    # target covariance
    tmp_t = torch.ones((1, nt)).cuda() @ target
    ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)

    # frobenius norm
    loss = (cs - ct).pow(2).sum()
    loss = loss / (4 * d * d)

    return loss

"""MMD"""
class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd(self, X, Y):
        delta = X.mean(axis=0) - Y.mean(axis=0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
            return loss

"""Cosine"""
def Cosine(source, target):
    source, target = source.mean(0), target.mean(0)
    cos = nn.CosineSimilarity(dim=0)
    loss = cos(source, target)
    return loss.mean()

"""Adversarial"""
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class Discriminator(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=1024, bn_size=25, num_domains=4):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        layers = [
            nn.Linear(input_dim * bn_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(hidden_dim//2)),
            nn.BatchNorm1d(int(hidden_dim//2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim//2), num_domains),
        ]
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# class Discriminator(nn.Module):
#     def __init__(self, input_dim=256, hidden_dim=128):
#         super(Discriminator, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_dim, 1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = self.relu1(x)
#         x = self.fc2(x)
#         x = self.sigmoid(x)
#         return x
#
#
# def Adv(source, target, input_dim=64, hidden_dim=32):
#     domain_loss = nn.BCELoss()
#     adv_net = Discriminator(input_dim, hidden_dim).cuda()
#     domain_src = torch.ones(len(source)).cuda()
#     domain_tar = torch.zeros(len(target)).cuda()
#     domain_src, domain_tar = domain_src.view(domain_src.shape[0], 1), domain_tar.view(domain_tar.shape[0], 1)
#     reverse_src = ReverseLayerF.apply(source, 1)
#     reverse_tar = ReverseLayerF.apply(target, 1)
#     pred_src = adv_net(reverse_src)
#     pred_tar = adv_net(reverse_tar)
#     loss_s, loss_t = domain_loss(pred_src, domain_src), domain_loss(pred_tar, domain_tar)
#     loss = loss_s + loss_t
#     return loss

"""KL Divergence"""
def KL_Div(source, target):
    if len(source) < len(target):
        target = target[:len(source)]
    elif len(source) > len(target):
        source = source[:len(target)]
    criterion = nn.KLDivLoss(reduction='batchmean')
    loss = criterion(source.log(), target)
    return loss

"""JS Divergence"""
def JS_Div(source, target):
    if len(source) < len(target):
        target = target[:len(source)]
    elif len(source) > len(target):
        source = source[:len(target)]
    M = .5 * (source + target)
    loss_1, loss_2 = KL_Div(source, M), KL_Div(target, M)
    return .5 * (loss_1 + loss_2)