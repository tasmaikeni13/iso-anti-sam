import torch
from torch.optim.optimizer import Optimizer

class SAM(Optimizer):
    """
    Standard Vanilla Sharpness-Aware Minimization (SAM).
    Reference: Foret et al., 2020 (https://arxiv.org/abs/2010.01412).

    Simultaneously minimizes loss value and loss sharpness:
      min_w max_{||eps|| <= rho} L(w + eps)

    Update rule:
      1. eps = rho * (grad / (||grad||_2 + eps_floor))
      2. w_pert = w + eps
      3. Compute grad at w_pert
      4. Restore original w: w <- w_pert - eps
      5. Apply base optimizer step (e.g. AdamW or SGD) on grad(w_pert).
    """
    def __init__(self, params, base_optimizer_cls=torch.optim.AdamW, rho=0.05,
                 adaptive=False, **kwargs):
        if rho < 0.0:
            raise ValueError(f"Invalid perturbation radius rho: {rho}")
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        Computes the ascent perturbation eps = rho * grad / ||grad||_2
        and updates w <- w + eps. Saves original weights in state['old_p'].
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            rho = group["rho"]
            adaptive = group.get("adaptive", False)
            scale = rho / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                if adaptive:
                    e_w = (torch.pow(p, 2) if p.dtype.is_floating_point else p) * p.grad * scale
                else:
                    e_w = p.grad * scale
                p.add_(e_w)
                
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Restores original weights w from state['old_p'] and executes the base optimizer
        update using gradients evaluated at the perturbed position.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if "old_p" in self.state[p]:
                    p.data.copy_(self.state[p]["old_p"])
                    del self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def zero_grad(self, set_to_none=False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def _grad_norm(self):
        """
        Computes the global L2 gradient norm across all parameter groups in FP32.
        """
        device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            adaptive = group.get("adaptive", False)
            for p in group["params"]:
                if p.grad is not None:
                    if adaptive:
                        g = torch.abs(p) * p.grad
                    else:
                        g = p.grad
                    norms.append(torch.linalg.vector_norm(g, 2).to(device))
        if not norms:
            return torch.tensor(0.0, device=device)
        return torch.linalg.vector_norm(torch.stack(norms), 2)
