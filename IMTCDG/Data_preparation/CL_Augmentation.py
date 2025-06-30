#
import numpy as np
#
import torch


def DataTransform(sample, w_jitter_scale_ratio, s_max_seg, s_jitter_ratio):
    weak_aug = scaling(sample, w_jitter_scale_ratio, )
    strong_aug = jitter(permutation(sample, max_segments=s_max_seg),s_jitter_ratio)

    return weak_aug, strong_aug


def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)


def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0,warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)

def CoTmixup(Mix_ratio, Temp_shift, Main_X, Main_Y, Auxi_X, Auxi_Y):
    # Mix_ratio = 0.9, 0.79, Temp_shift = 10, 20, 25
    Mix_ratio = round(Mix_ratio, 2)
    h = Temp_shift // 2

    MainCL_X = Mix_ratio * Main_X + (1 - Mix_ratio) * \
               torch.mean(torch.stack([torch.roll(Auxi_X, -i, 2) for i in range(-h, h)], 2), 2)
    MainCL_Y = Mix_ratio * Main_Y + (1 - Mix_ratio) * \
               torch.mean(torch.stack([torch.roll(Auxi_Y, -i, 2) for i in range(-h, h)], 2), 2)

    AuxiCL_X = Mix_ratio * Auxi_X + (1 - Mix_ratio) * \
               torch.mean(torch.stack([torch.roll(Main_X, -i, 2) for i in range(-h, h)], 2), 2)
    AuxiCL_Y = Mix_ratio * Auxi_Y + (1 - Mix_ratio) * \
               torch.mean(torch.stack([torch.roll(Auxi_Y, -i, 2) for i in range(-h, h)], 2), 2)

    return MainCL_X, MainCL_Y