import torch

class SCR:

    def __init__(self, num_classes, n_sigma=2, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.n_sigma = n_sigma
        self.m = momentum

        # initialize Gaussian mean and variance
        self.prob_max_mu_t = torch.ones((self.num_classes)) / self.num_classes
        self.prob_max_var_t = torch.ones((self.num_classes))

    @torch.no_grad()
    def update(self, probs):
        max_probs, max_idx = probs.max(dim=-1)
        prob_max_mu_t = torch.zeros_like(self.prob_max_mu_t)
        prob_max_var_t = torch.ones_like(self.prob_max_var_t)
        for i in range(self.num_classes):
            prob = max_probs[max_idx == i]
            if len(prob) > 1:
                prob_max_mu_t[i] = torch.mean(prob)
                prob_max_var_t[i] = torch.var(prob, unbiased=True)
        self.prob_max_mu_t = self.m * self.prob_max_mu_t + (1 - self.m) * prob_max_mu_t
        self.prob_max_var_t = self.m * self.prob_max_var_t + (1 - self.m) * prob_max_var_t
        return max_probs, max_idx

    @torch.no_grad()
    def forward(self, logits,*args, **kwargs):
        if not self.prob_max_mu_t.is_cuda:
            self.prob_max_mu_t = self.prob_max_mu_t.to(logits.device)
        if not self.prob_max_var_t.is_cuda:
            self.prob_max_var_t = self.prob_max_var_t.to(logits.device)

        probs = logits.detach()
        self.update(probs)
        max_probs, max_idx = logits.max(dim=-1)

        mu = self.prob_max_mu_t[max_idx]
        var = self.prob_max_var_t[max_idx]
        mask = torch.exp(-((torch.clamp(max_probs - mu, max=0.0) ** 2) / (2 * var / (self.n_sigma ** 2))))
        return mask