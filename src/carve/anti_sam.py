import torch
from torch.optim.optimizer import Optimizer

class AntiSAM(Optimizer):
    """
    Standard Anti-SAM Optimizer:
      min_w ( min_{||eps|| <= rho} L(w + eps) )

    Computes:
      eps = -rho * (grad / ||grad||)
      w_step = w - lr * grad(w + eps)

    NOTE: Demonstrates rapid training loss descent, but suffers from
    phase-space caustic collapse and sharp-needle overfitting.
    Use Carve for generalization.
    """
    def __init__(self, params, base_optimizer_cls=torch.optim.SGD, rho=0.05, **kwargs):
        if rho < 0.0:
            raise ValueError(f"Invalid perturbation radius rho: {rho}")
        defaults = dict(rho=rho, **kwargs)
        super(AntiSAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """Computes eps = -rho * (grad / ||grad||) and updates w <- w + eps."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = -group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Restores w and applies base optimizer update on grad(w + eps)."""
        for group in self.param_groups:
            for p in group["params"]:
                if "old_p" in self.state[p]:
                    p.data.copy_(self.state[p]["old_p"])
                    del self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = [
            torch.linalg.vector_norm(p.grad, 2).to(shared_device)
            for group in self.param_groups for p in group["params"]
            if p.grad is not None
        ]
        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.linalg.vector_norm(torch.stack(norms), 2)
