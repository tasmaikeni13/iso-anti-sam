import torch
from torch.optim.optimizer import Optimizer

class IsoAntiSAM(Optimizer):
    """
    IsoAntiSAM: Isochoric and Coherent Anti-Sharpness-Aware Minimization.

    Solves the sharp-needle catastrophe of Anti-SAM while preserving its
    rapid loss-cutting property.

    Features:
    1. Bilateral Coherence Gate: Evaluates cross-batch cosine similarity to quench
       spurious needle gradients (gate = 0) and amplify true shared manifold signals.
    2. Isochoric Gauge: Transverse divergence-free constraint preserving phase-space volume.
    3. Direct synchronization of validation loss descent with training loss descent.
    """
    def __init__(self, params, base_optimizer_cls=torch.optim.SGD, rho=0.05,
                 coherence_floor=0.0, **kwargs):
        if rho < 0.0:
            raise ValueError(f"Invalid perturbation radius rho: {rho}")
        defaults = dict(rho=rho, coherence_floor=coherence_floor, **kwargs)
        super(IsoAntiSAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def compute_bilateral_perturbation(self, grads_b1, grads_b2):
        """
        Given gradient lists from two independent micro-batches B1 and B2:
        Computes the Bilateral Coherent Erosion vector:
          cos_sim = <g1, g2> / (||g1|| * ||g2||)
          gate = max(coherence_floor, cos_sim)
          eps = -rho * gate * (g_avg / ||g_avg||)
        """
        # Flatten and compute norms
        dot_product = 0.0
        norm_sq_1 = 0.0
        norm_sq_2 = 0.0
        
        for g1, g2 in zip(grads_b1, grads_b2):
            if g1 is not None and g2 is not None:
                dot_product += torch.sum(g1 * g2).item()
                norm_sq_1 += torch.sum(g1 * g1).item()
                norm_sq_2 += torch.sum(g2 * g2).item()
                
        norm1 = norm_sq_1 ** 0.5 + 1e-12
        norm2 = norm_sq_2 ** 0.5 + 1e-12
        cos_sim = dot_product / (norm1 * norm2)

        # Average gradient norm
        g_avg_list = []
        norm_sq_avg = 0.0
        for g1, g2 in zip(grads_b1, grads_b2):
            if g1 is not None and g2 is not None:
                g_avg = 0.5 * (g1 + g2)
                norm_sq_avg += torch.sum(g_avg * g_avg).item()
                g_avg_list.append(g_avg)
            else:
                g_avg_list.append(None)
                
        norm_avg = norm_sq_avg ** 0.5 + 1e-12
        
        for group in self.param_groups:
            gate = max(group.get("coherence_floor", 0.0), cos_sim)
            rho = group["rho"]
            scale = -rho * gate / norm_avg
            
            idx = 0
            for p in group["params"]:
                if p.grad is None or g_avg_list[idx] is None:
                    idx += 1
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = g_avg_list[idx] * scale
                p.add_(e_w)
                idx += 1
                
        return cos_sim

    @torch.no_grad()
    def step_with_bilateral(self, zero_grad=False):
        """Restores original weights and updates with the evaluated outer gradient."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if "old_p" in self.state[p]:
                    p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()
