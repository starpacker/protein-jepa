import torch
import sys, math
sys.path.insert(0, '.')
from protein_jepa_frame.models.lspwm import calibrated_drift_matching_loss

class DummyPC:
    enabled = True
    log_mobility_scale = torch.nn.Parameter(torch.tensor(math.log(10000.0)))
    log_free_energy_scale = torch.nn.Parameter(torch.tensor(0.0))
    def regularization_loss(self): return torch.tensor(0.0)
    @property
    def mobility_scale(self): return 10000.0

class DummyWM:
    physical_calibration = DummyPC()
    def calibrated_transition(self, q, cond, dt, temp, history_features=None):
        return {'mean': q.detach()}

z = torch.randn(4, 256)
t = z + 0.03 * torch.randn(4, 256)
wm = DummyWM()

loss_norm = calibrated_drift_matching_loss(wm, z, t, None, 8000.0, None, normalize=True)
loss_raw = calibrated_drift_matching_loss(wm, z, t, None, 8000.0, None, normalize=False)
print(f'normalized: {loss_norm.item():.4f}  (expect ~1)')
print(f'raw: {loss_raw.item():.6f}  (expect ~0.0009)')

class DummyWM2:
    physical_calibration = DummyPC()
    def calibrated_transition(self, q, cond, dt, temp, history_features=None):
        return {'mean': t}
wm2 = DummyWM2()
loss_perf = calibrated_drift_matching_loss(wm2, z, t, None, 8000.0, None, normalize=True)
print(f'perfect: {loss_perf.item():.4e}  (expect ~0)')
print('ALL OK')