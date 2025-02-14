# Copyright (c) Nanxin Chen, Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This implementation is modified from https://github.com/zcaceres/spec_augment

MIT License

Copyright (c) 2019 Zach Caceres, Jenny Cai
"""
import time

import numpy as np

import torch


def specaug(spec, W=80, F=27, T=70, num_freq_masks=2, num_time_masks=2, p=0.2, replace_with_zero=False,
            pw=None, torch=False):
    """SpecAugment

    Reference: SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
        (https://arxiv.org/pdf/1904.08779.pdf)

    This implementation modified from https://github.com/zcaceres/spec_augment

    Args:
        spec (numpy.ndarray): input tensor of the shape `(T, dim)`
        W (int): time warp parameter
        F (int): maximum width of each freq mask
        T (int): maximum width of each time mask
        num_freq_masks (int): number of frequency masks
        num_time_masks (int): number of time masks. In negative, number will be adaptive /
         proportional to the spec length
        p (int): time mask width shouldn't exeed this times num of frames
        replace_with_zero (bool): if True, masked parts will be filled with 0, if False, filled with mean

    Returns:
        output (numpy.ndarray): resultant matrix of shape `(T, dim)`
    """
    spec = spec if torch else torch.from_numpy(spec)
    spec = spec.cuda().transpose(0, 1)
    if replace_with_zero:
        pad_value = 0.
    else:
        pad_value = spec.mean()

    timing = dict()
    twrp_start = time.time()
    time_warped = time_warp(spec, W=W, pw=pw)
    timing['aug_twrp'] = time.time() - twrp_start

    fmsk_start = time.time()
    freq_masked = freq_mask(time_warped, F=F, num_masks=num_freq_masks, pad_value=pad_value)
    timing['aug_fmsk'] = time.time() - fmsk_start

    tmsk_start = time.time()
    time_masked = time_mask(freq_masked, T=T, num_masks=num_time_masks, p=p, pad_value=pad_value)
    timing['aug_tmsk'] = time.time() - tmsk_start

    return time_masked.transpose(0, 1) if torch else time_masked.transpose(0, 1).numpy(), timing


def time_warp(spec, W=5, pw=None):
    """Time warping

    Args:
        spec (torch.Tensor): input tensor of shape `(dim, T)`
        W (int): time warp parameter. Skip if not positive
        pw (float): increase / decrease rate (recommended no more than 0.2)

    Returns:
        time warpped tensor (torch.Tensor): output tensor of shape `(dim, T)`
    """
    t = spec.size(1)
    W = W if pw is None else int(t * pw)
    if W <= 0 or t - W <= W + 1:
        return spec
    center = np.random.randint(W + 1, t - W)
    warped = np.random.randint(center - W, center + W + 1)
    if warped == center:
        return spec
    spec = spec.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():  # to make the results deterministic
        left = torch.nn.functional.interpolate(
            spec[:, :, :, :center], size=(spec.size(2), warped),
            mode="bicubic", align_corners=False,
        )
        right = torch.nn.functional.interpolate(
            spec[:, :, :, center:], size=(spec.size(2), t - warped),
            mode="bicubic", align_corners=False,
        )
    return torch.cat((left, right), dim=-1).squeeze(0).squeeze(0)


def freq_mask(spec, F=30, num_masks=1, pad_value=0.):
    """Frequency masking

    Args:
        spec (torch.Tensor): input tensor of shape `(dim, T)`
        F (int): maximum width of each mask
        num_masks (int): number of masks
        pad_value (float): value for padding

     Returns:
         freq masked tensor (torch.Tensor): output tensor of shape `(dim, T)`
    """
    cloned = spec.clone()
    num_mel_channels = cloned.size(0)

    for i in range(num_masks):
        f = np.random.randint(0, F + 1)
        f_zero = np.random.randint(0, num_mel_channels - f + 1)

        if f == 0:
            continue
        cloned[f_zero:f_zero + f] = pad_value
    return cloned


def time_mask(spec, T=40, num_masks=1, p=0.2, pad_value=0.):
    """Time masking

    Args:
        spec (torch.Tensor): input tensor of shape `(dim, T)`
        T (int): maximum width of each mask
        num_masks (int): number of masks
        p (float): time mask width shouldn't exeed this times num of frames
        pad_value (float): value for padding

    Returns:
        time masked tensor (torch.Tensor): output tensor of shape `(dim, T)`
    """
    cloned = spec.clone()
    len_spectro = cloned.size(1)
    T = min(T, int(len_spectro * p))
    if T == 0 or num_masks == 0:
        return cloned
    num_masks = num_masks if num_masks > 0 else int(len_spectro * p / T)

    for i in range(num_masks):
        t = np.random.randint(0, T + 1)
        t_zero = np.random.randint(0, len_spectro - t + 1)

        if t == 0:
            continue
        cloned[:, t_zero:t_zero + t] = pad_value
    return cloned
