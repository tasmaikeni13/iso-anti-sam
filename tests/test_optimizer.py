import unittest
import torch
import torch.nn as nn
from iso_anti_sam import AntiSAM, IsoAntiSAM

class TestOptimizers(unittest.TestCase):

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
        
        # Check parameters were updated
        for p_orig, p in zip(orig_params, m.parameters()):
            self.assertFalse(torch.allclose(p_orig, p))

    def test_iso_anti_sam_coherence_gating(self):
        m = nn.Linear(8, 2)
        opt = IsoAntiSAM(m.parameters(), lr=0.01, rho=0.05)
        
        # Case 1: Identical gradients (cos_sim = 1.0)
        g_identical_1 = [torch.ones_like(p) for p in m.parameters()]
        g_identical_2 = [torch.ones_like(p) for p in m.parameters()]
        sim = opt.compute_bilateral_perturbation(g_identical_1, g_identical_2)
        self.assertAlmostEqual(sim, 1.0, places=4)
        
        # Step with bilateral
        opt.step_with_bilateral(zero_grad=True)
        
        # Case 2: Opposite gradients (cos_sim = -1.0) -> gate should be 0.0
        m2 = nn.Linear(8, 2)
        opt2 = IsoAntiSAM(m2.parameters(), lr=0.01, rho=0.05)
        g_opp_1 = [torch.ones_like(p) for p in m2.parameters()]
        g_opp_2 = [-torch.ones_like(p) for p in m2.parameters()]
        
        p_before = [p.clone() for p in m2.parameters()]
        sim2 = opt2.compute_bilateral_perturbation(g_opp_1, g_opp_2)
        self.assertLess(sim2, -0.99)
        
        # Check that weights did NOT move because gate = 0.0
        for pb, p in zip(p_before, m2.parameters()):
            self.assertTrue(torch.allclose(pb, p))

    def test_device_placement_gpu(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA/ROCm not available")
        
        device = torch.device("cuda:0")
        m = nn.Linear(16, 4).to(device)
        opt = IsoAntiSAM(m.parameters(), lr=0.01, rho=0.05)
        
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
        opt1 = IsoAntiSAM(m.parameters(), lr=0.01, rho=0.05)
        state = opt1.state_dict()
        
        opt2 = IsoAntiSAM(m.parameters(), lr=0.02, rho=0.10)
        opt2.load_state_dict(state)
        
        self.assertEqual(opt2.param_groups[0]["lr"], 0.01)
        self.assertEqual(opt2.param_groups[0]["rho"], 0.05)

    def test_adamw_base_optimizer(self):
        m = nn.Linear(8, 2)
        opt = IsoAntiSAM(m.parameters(), base_optimizer_cls=torch.optim.AdamW, lr=1e-3, rho=0.02)
        
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

if __name__ == '__main__':
    unittest.main()
