# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

def replace_inf_to_zero(val):
    val[val == float('inf')] = 0.0
    return val

class SCS:
    """
    SAT in FreeMatch
    """

    def __init__(self, num_classes, momentum=0.999, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.m = momentum

        self.local_ = torch.ones((self.num_classes))  / self.num_classes
        self.global_ = self.local_.mean()

    @torch.no_grad()
    def update(self, config, probs, y):
        given_idx=y.long()
        given_probs=probs[torch.arange(len(given_idx)),given_idx]

        if config.use_quantile:
            self.global_ = self.global_ * self.m + (1 - self.m) * torch.quantile(given_probs, 0.8)  # * given_probs.mean()
        else:
            self.global_ = self.global_ * self.m + (1 - self.m) * given_probs.mean()

        if config.clip_thresh:
            self.global_ = torch.clip(self.global_, 0.0, 0.95)

        self.local_ = self.local_ * self.m + (1 - self.m) * probs.mean(dim=0)

    @torch.no_grad()
    def forward(self, config, logits_x, y, *args, **kwargs):
        if not self.local_.is_cuda:
            self.local_ = self.local_.to(logits_x.device)
        if not self.global_.is_cuda:
            self.global_ = self.global_.to(logits_x.device)

        probs = logits_x.detach()

        self.update(config, probs, y)

        given_idx = y.long()
        given_probs = probs[torch.arange(len(given_idx)),given_idx]

        mod = self.local_ / torch.max(self.local_, dim=-1)[0]
        mask = given_probs.ge(self.global_ * mod[given_idx]).to(given_probs.dtype)
        mask_idx = mask.nonzero()
        mask_noise_idx = (mask - 1).nonzero()
        self.mask = mask
        return mask_idx,mask_noise_idx
