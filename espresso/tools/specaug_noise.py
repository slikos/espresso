# Copyright (c) slikos

import numpy as np
import torch
from torch import Tensor


class AddSpecNoise:
    """Add noise spec to signal spec"""
    def __init__(self, noises_npz_path=None, noise_multiplier_range=None):
        self.noise_multiplier_range = noise_multiplier_range or [1., 1.]
        self.noises = np.load(noises_npz_path, allow_pickle=True) if noises_npz_path is not None else None

    def __call__(self, spec: Tensor):
        """Add noise spec to signal spec
        Args:
            spec (torch.Tensor): input tensor of shape `(T, dim)`
        Returns:
            noised tensor (torch.Tensor): output tensor of shape `(T, dim)`
        """
        if self.noises is None:
            return spec

        cloned = spec.clone()
        spec_duration = cloned.size(0)

        noises_start = np.random.randint(0, self.noises.shape[0] - spec_duration)
        noise = self.noises[noises_start:noises_start+spec_duration, :]
        noise = torch.from_numpy(noise).to(cloned.device)
        noise_multiplier = np.random.uniform(self.noise_multiplier_range[0],
                                             self.noise_multiplier_range[1])
        cloned += noise * noise_multiplier
        return cloned
