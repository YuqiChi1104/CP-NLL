# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

def replace_inf_to_zero(val):
    val[val == float('inf')] = 0.0
    return val
def min_max(x):
    return (x - x.min())/(x.max() - x.min())
class MASK_Loss:
    """
    SAT in FreeMatch
    """

    def __init__(self, num_classes, momentum=0.999, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.m = momentum

        self.p_model = torch.ones((self.num_classes))  / self.num_classes
        self.time_p = self.p_model.mean()

    @torch.no_grad()
    def update(self, config, loss,y):
        # loss =  min_max(loss.cpu().numpy())

        if config.use_quantile:
            self.time_p = self.time_p * self.m + (1 - self.m) * loss.mean()  # * given_probs.mean()
        else:
            self.time_p = self.time_p * self.m + (1 - self.m) * loss.mean()

        if config.clip_thresh:
            self.time_p = torch.clip(self.time_p, 0.0, 0.95)

        for i in range(self.num_classes):
            ind = ( y == i)
            self.p_model[i] = self.p_model[i] * self.m + (1 - self.m) * loss[ind].mean()

    @torch.no_grad()
    def masking(self, config, loss,y,y_true,softmax_x_ulb=False, *args, **kwargs):
        if not self.p_model.is_cuda:
            self.p_model = self.p_model.to(loss.device)
        if not self.time_p.is_cuda:
            self.time_p = self.time_p.to(loss.device)

        # loss = loss.detach()
        loss = min_max(loss.cpu())
        loss = loss.detach()
        self.update(config, loss, y)

        # given_probs, given_idx = probs.max(dim=-1)

        mod = self.p_model / torch.max(self.p_model, dim=-1)[0]
        mask = torch.zeros(len(y))
        for i in range(len(y)):
            mask[i] = loss[i].le(self.time_p * mod[y[i].long()]).to(loss.dtype)
        mask_idx = mask.nonzero()
        mask_noise_idx = (mask - 1).nonzero()
        # print(mask)
        cont = sum(y[mask_idx] == y_true[mask_idx])
        # print(y[mask_idx])
        # print(y_true[mask_idx])
        # print(cont)
        print("The pura ratio of clean subset is {}/{} = {} ".format(cont,mask.sum(),float(cont/(mask.sum()))))
        self.mask = mask
        return mask_idx,mask_noise_idx

    def entropy_loss(self, logits_s, logits_w, index):
        # select samples

        prob_s = logits_s.softmax(dim=-1)
        _, pred_label_s = torch.max(prob_s, dim=-1)

        hist_s = torch.bincount(pred_label_s, minlength=logits_s.shape[1]).to(logits_w.dtype)
        hist_s = hist_s / hist_s.sum()

        # modulate prob model
        p_model= self.p_model.reshape(1, -1)
        label_hist = self.label_hist.reshape(1, -1)
        # prob_model_scaler = torch.nan_to_num(1 / label_hist, nan=0.0, posinf=0.0, neginf=0.0).detach()
        prob_model_scaler = replace_inf_to_zero(1 / label_hist).detach()
        mod_prob_model =p_model* prob_model_scaler
        mod_prob_model = mod_prob_model / mod_prob_model.sum(dim=-1, keepdim=True)

        # modulate mean prob
        mean_prob_scaler_s = replace_inf_to_zero(1 / hist_s).detach()
        # mean_prob_scaler_s = torch.nan_to_num(1 / hist_s, nan=0.0, posinf=0.0, neginf=0.0).detach()
        mod_mean_prob_s = prob_s.mean(dim=0, keepdim=True) * mean_prob_scaler_s
        mod_mean_prob_s = mod_mean_prob_s / mod_mean_prob_s.sum(dim=-1, keepdim=True)

        loss = mod_prob_model * torch.log(mod_mean_prob_s + 1e-12)
        loss = loss.sum(dim=1)
        return loss.mean(), hist_s.mean()



