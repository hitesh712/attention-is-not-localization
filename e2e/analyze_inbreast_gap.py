"""INbreast: oracle vs end-to-end gap, split by whether the detector found the mass; paired tests."""
import numpy as np
from scipy.stats import wilcoxon

z = np.load('G:/My Drive/VS_CC/models/e2e/inbreast_e2e.npz', allow_pickle=True)
L = {k[4:]: z[k] for k in z.files if k.startswith('les_')}


def row(name, m):
    return (f"{name:<22} n={m.sum():>3} | oracle {L['d_oracle'][m].mean():.3f} | det {L['d_det'][m].mean():.3f} | "
            f"e2e-top1 {L['d_top1'][m].mean():.3f} | e2e-all {L['d_all'][m].mean():.3f} "
            f"(>0.5 {np.mean(L['d_all'][m] > 0.5) * 100:4.1f}%)")


det = L['detected'].astype(bool)
print(row('all masses', np.ones_like(det)))
print(row('detected', det))
print(row('missed', ~det))
q = np.quantile(L['side'], [1 / 3, 2 / 3])
for lo, hi, nm in ((0, q[0], f'small (<{q[0]:.0f}px)'), (q[0], q[1], 'medium'), (q[1], 1e9, f'large (>{q[1]:.0f}px)')):
    m = (L['side'] >= lo) & (L['side'] < hi)
    print(row(nm, m) + f" | detected {det[m].mean() * 100:.1f}%")
print('\nper-lesion e2e-all vs detector (Wilcoxon):', wilcoxon(L['d_all'], L['d_det']))
print('per-lesion e2e-all vs detector, detected only:', wilcoxon(L['d_all'][det], L['d_det'][det]))
