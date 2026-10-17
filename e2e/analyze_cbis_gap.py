"""Decompose the oracle vs end-to-end gap on CBIS: detection vs delineation, by type and size; paired tests."""
import numpy as np
from scipy.stats import wilcoxon

z = np.load('G:/My Drive/VS_CC/models/e2e/cbis_e2e.npz', allow_pickle=True)
L = {k[4:]: z[k] for k in z.files if k.startswith('les_')}
I = {k[4:]: z[k] for k in z.files if k.startswith('img_')}


def row(name, m):
    return (f"{name:<26} n={m.sum():>3} | oracle {L['d_oracle'][m].mean():.3f} | det {L['d_det'][m].mean():.3f} | "
            f"e2e-all {L['d_all'][m].mean():.3f} (>0.5 {np.mean(L['d_all'][m] > 0.5) * 100:4.1f}%) | "
            f"detected {L['detected'][m].mean() * 100:5.1f}%")


det = L['detected'].astype(bool); abn = L['abn'].astype(str)
print('--- is the gap detection or delineation? ---')
print(row('all lesions', np.ones_like(det)))
print(row('detected by the detector', det))
print(row('missed by the detector', ~det))
print('\n--- by abnormality type ---')
for a in sorted(set(abn)):
    print(row(a, abn == a))
print('\n--- by lesion size (longest side at full resolution, terciles) ---')
q = np.quantile(L['side'], [1 / 3, 2 / 3])
for lo, hi, nm in ((0, q[0], f'small (<{q[0]:.0f}px)'), (q[0], q[1], 'medium'), (q[1], 1e9, f'large (>{q[1]:.0f}px)')):
    print(row(nm, (L['side'] >= lo) & (L['side'] < hi)))
print('\n--- detected lesions only, by type ---')
for a in sorted(set(abn)):
    print(row(a + ' & detected', (abn == a) & det))
print('\n--- paired tests (Wilcoxon signed-rank on the same lesions / images) ---')
print('per-lesion  e2e-all  vs detector:', wilcoxon(L['d_all'], L['d_det']))
print('per-lesion  e2e-all  vs oracle  :', wilcoxon(L['d_all'], L['d_oracle']))
print('per-image   e2e-top1 vs detector:', wilcoxon(I['dF_top1'], I['dF_det']))
print('per-image   e2e-all  vs detector:', wilcoxon(I['dF_all'], I['dF_det']))
print(f"\nimages with no detection >= 0.5: {int(I['fallback'].sum())}/{len(I['fallback'])} | "
      f"mean detections per image {I['n_det'].mean():.2f}")
