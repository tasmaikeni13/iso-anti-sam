import unittest
import torch
import torch.nn as nn
from carve import AntiSAM, Carve, IsoAntiSAM, SAM

class TestOptimizers(unittest.TestCase):

    def test_carve_alias(self):
        """Verifies backward compatibility alias IsoAntiSAM points to Carve."""
        self.assertIs(IsoAntiSAM, Carve)

    def test_anti_sam_step(self):
        m = nn.Linear(8, 2)
        opt = AntiSAM(m.parameters(), lr=0.01, rho=0.05)
        
        x = torch.randn(4, 8)
        loss = m(x).sum()
        loss.backward()
        
        orig_params = [p.clone() for p in m.parameters()]
        opt.first_step(zero_grad=True)
        
        # Check that parameters moved
        for p_orig, p in zip(orig_params, m.parameters()):
            self.assertFalse(torch.allclose(p_orig, p))
            
        loss2 = m(x).sum()
        loss2.backward()
        opt.second_step(zero_grad=True)
        
        # Check parameters were updated and old_p cleaned up
        for p_orig, p in zip(orig_params, m.parameters()):
            self.assertFalse(torch.allclose(p_orig, p))
            self.assertNotIn("old_p", opt.state[p])

    def test_zero_grad_isolation(self):
        """Verifies zero-grad detachment and absence of computational graph memory retention."""
        m = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 2))
        opt = Carve(m.parameters(), lr=0.01, rho=0.05)
        
        x1 = torch.randn(4, 10, requires_grad=True)
        loss1 = m(x1).sum()
        loss1.backward()
        g1 = [p.grad.clone().detach() for p in m.parameters()]
        opt.zero_grad(set_to_none=True)

        x2 = torch.randn(4, 10, requires_grad=True)
        loss2 = m(x2).sum()
        loss2.backward()
        g2 = [p.grad.clone().detach() for p in m.parameters()]
        opt.zero_grad(set_to_none=True)

        # Gradients must have no graph attachment
        for g in g1 + g2:
            self.assertIsNone(g.grad_fn)

        sim = opt.compute_bilateral_perturbation(g1, g2)
        self.assertIsInstance(sim, float)

        # Outer backward pass
        x_out = torch.randn(4, 10)
        loss_out = m(x_out).sum()
        loss_out.backward()
        opt.step_with_bilateral(zero_grad=True)

        # Gradients must be zeroed / cleared
        for p in m.parameters():
            self.assertTrue(p.grad is None or torch.all(p.grad == 0))

    def test_inplace_perturbation_recovery(self):
        """Verifies bitwise restoration of original weights before outer update and clean state footprint."""
        m = nn.Linear(8, 2, bias=False)
        # Use lr=0.0 to test exact bitwise recovery of parameter values
        opt = Carve(m.parameters(), lr=0.0, rho=0.05)
        
        orig_w = m.weight.clone()
        g1 = [torch.ones_like(m.weight)]
        g2 = [torch.ones_like(m.weight)]
        
        opt.compute_bilateral_perturbation(g1, g2)
        # Weights should be perturbed during intermediate phase
        self.assertFalse(torch.equal(m.weight, orig_w))
        
        # Outer step with lr=0.0 must restore original weights bitwise
        m.weight.grad = torch.randn_like(m.weight)
        opt.step_with_bilateral(zero_grad=True)
        
        self.assertTrue(torch.equal(m.weight, orig_w))
        # Ensure state footprint contains no lingering old_p clone
        self.assertNotIn("old_p", opt.state[m.weight])

    def test_orthogonal_gradient_quenching(self):
        """When g1 orthogonal or opposing g2 (<g1, g2> <= 0), gate collapses to 0 and perturbation is zero."""
        m = nn.Linear(4, 2, bias=False)
        opt = Carve(m.parameters(), lr=0.01, rho=0.05, coherence_floor=0.0)
        
        orig_w = m.weight.clone()
        
        # Perfectly orthogonal gradients: g1 nonzero on row 0, g2 nonzero on row 1
        g1 = [torch.zeros_like(m.weight)]
        g1[0][0, :] = 1.0
        g2 = [torch.zeros_like(m.weight)]
        g2[0][1, :] = 1.0
        
        sim = opt.compute_bilateral_perturbation(g1, g2)
        self.assertAlmostEqual(sim, 0.0, places=5)
        # Since gate == 0, weight perturbation must be exactly 0
        self.assertTrue(torch.equal(m.weight, orig_w))
        
        # Opposing gradients: g1 = -g2 (<g1, g2> = -1)
        g_opp1 = [torch.ones_like(m.weight)]
        g_opp2 = [-torch.ones_like(m.weight)]
        sim_opp = opt.compute_bilateral_perturbation(g_opp1, g_opp2)
        self.assertAlmostEqual(sim_opp, -1.0, places=5)
        # Gate clamped at coherence_floor = 0.0 -> weights unchanged
        self.assertTrue(torch.equal(m.weight, orig_w))

    def test_aligned_gradient_preservation(self):
        """When g1 == g2, gate == 1, delivering full morphological erosion perturbation: -rho * g / ||g||."""
        m = nn.Linear(4, 2, bias=False)
        rho = 0.05
        opt = Carve(m.parameters(), lr=0.01, rho=rho)
        
        orig_w = m.weight.clone()
        g = torch.randn_like(m.weight)
        g1 = [g.clone()]
        g2 = [g.clone()]
        
        sim = opt.compute_bilateral_perturbation(g1, g2)
        self.assertAlmostEqual(sim, 1.0, places=5)
        
        expected_norm = torch.linalg.vector_norm(g).item() + 1e-12
        expected_pert = orig_w - rho * (g / expected_norm)
        self.assertTrue(torch.allclose(m.weight, expected_pert, atol=1e-6))

    def test_multi_param_group_handling(self):
        """Handles disparate parameter shapes (2D weights, 1D biases, LayerNorm scalars) and multiple groups."""
        p1 = nn.Parameter(torch.randn(32, 16))
        p2 = nn.Parameter(torch.randn(32))
        p3 = nn.Parameter(torch.randn(1, 16))
        
        opt = Carve([
            {"params": [p1, p2], "lr": 1e-3, "rho": 0.05},
            {"params": [p3], "lr": 5e-4, "rho": 0.02, "coherence_floor": 0.0}
        ])
        
        g1 = [torch.randn_like(p1), torch.randn_like(p2), torch.randn_like(p3)]
        g2 = [torch.randn_like(p1), torch.randn_like(p2), torch.randn_like(p3)]
        
        sim = opt.compute_bilateral_perturbation(g1, g2)
        self.assertTrue(-1.0 <= sim <= 1.0)
        
        p1.grad = torch.randn_like(p1)
        p2.grad = torch.randn_like(p2)
        p3.grad = torch.randn_like(p3)
        
        opt.step_with_bilateral(zero_grad=True)
        for p in [p1, p2, p3]:
            self.assertNotIn("old_p", opt.state[p])

    def test_device_placement_gpu(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA/ROCm not available")
        
        device = torch.device("cuda:0")
        m = nn.Linear(16, 4).to(device)
        opt = Carve(m.parameters(), lr=0.01, rho=0.05)
        
        x = torch.randn(8, 16, device=device)
        y = torch.randint(0, 4, (8,), device=device)
        crit = nn.CrossEntropyLoss()
        
        # Micro-batch 1
        loss1 = crit(m(x[:4]), y[:4])
        loss1.backward()
        g1 = [p.grad.clone() for p in m.parameters()]
        m.zero_grad()
        
        # Micro-batch 2
        loss2 = crit(m(x[4:]), y[4:])
        loss2.backward()
        g2 = [p.grad.clone() for p in m.parameters()]
        m.zero_grad()
        
        sim = opt.compute_bilateral_perturbation(g1, g2)
        self.assertTrue(-1.0 <= sim <= 1.0)
        
        loss_outer = crit(m(x), y)
        loss_outer.backward()
        opt.step_with_bilateral(zero_grad=True)
        
        # Verify all parameters remain on GPU
        for p in m.parameters():
            self.assertEqual(p.device.type, "cuda")

    def test_state_dict_serialization(self):
        m = nn.Linear(8, 2)
        opt1 = Carve(m.parameters(), lr=0.01, rho=0.05)
        state = opt1.state_dict()
        
        opt2 = Carve(m.parameters(), lr=0.02, rho=0.10)
        opt2.load_state_dict(state)
        
        self.assertEqual(opt2.param_groups[0]["lr"], 0.01)
        self.assertEqual(opt2.param_groups[0]["rho"], 0.05)

    def test_adamw_base_optimizer(self):
        m = nn.Linear(8, 2)
        opt = Carve(m.parameters(), base_optimizer_cls=torch.optim.AdamW, lr=1e-3, rho=0.02)
        
        x = torch.randn(4, 8)
        loss = m(x).sum()
        loss.backward()
        g1 = [p.grad.clone() for p in m.parameters()]
        m.zero_grad()
        
        loss2 = m(x).sum()
        loss2.backward()
        g2 = [p.grad.clone() for p in m.parameters()]
        m.zero_grad()
        
        sim = opt.compute_bilateral_perturbation(g1, g2)
        self.assertTrue(-1.0 <= sim <= 1.0)
        loss_full = m(x).sum()
        loss_full.backward()
        opt.step_with_bilateral(zero_grad=True)

    def test_sam_step(self):
        """Verifies Vanilla SAM forward perturbation, restoration, and parameter update."""
        m = nn.Linear(8, 2)
        opt = SAM(m.parameters(), base_optimizer_cls=torch.optim.AdamW, lr=1e-3, rho=0.05)
        
        orig_w = m.weight.clone()
        x = torch.randn(4, 8)
        loss = m(x).sum()
        loss.backward()
        
        opt.first_step(zero_grad=True)
        # Weights should be perturbed
        self.assertFalse(torch.equal(m.weight, orig_w))
        self.assertIn("old_p", opt.state[m.weight])
        
        # Second step restores weights and applies update
        loss2 = m(x).sum()
        loss2.backward()
        opt.second_step(zero_grad=True)
        
        # old_p should be cleaned up
        self.assertNotIn("old_p", opt.state[m.weight])
        # Weights should have updated from orig_w
        self.assertFalse(torch.equal(m.weight, orig_w))

    def test_carve_dynamic_schedule(self):
        """Verifies that Carve supports dynamic rho overrides for cosine perturbation decay."""
        m = nn.Linear(4, 2, bias=False)
        opt = Carve(m.parameters(), lr=0.01, rho=0.10)
        
        orig_w = m.weight.clone()
        g1 = [torch.ones_like(m.weight)]
        g2 = [torch.ones_like(m.weight)]
        
        # Override rho with scheduled value 0.02
        opt.compute_bilateral_perturbation(g1, g2, rho=0.02)
        
        # Norm of perturbation should correspond to rho=0.02, not 0.10
        pert_norm = torch.linalg.vector_norm(m.weight - orig_w).item()
        self.assertAlmostEqual(pert_norm, 0.02, places=4)
        
        opt.step_with_bilateral(zero_grad=True)

if __name__ == '__main__':
    unittest.main()
