"""Convert nnU-Net v2 probability outputs into the detector-map format read by the end-to-end scripts.
  python nnunet_maps.py          CBIS-DDSM test set + INbreast   (e2e_cbis.py / e2e_inbreast.py)
  python nnunet_maps.py vindr    VinDr-Mammo                     (e2e_vindr_dicom.py)
"""
import os, sys, glob
import numpy as np

ROOT = 'D:/nnunet'


def lesion_prob(path):
    p = np.load(path)['probabilities']            # (classes, 1, H, W) for 2D natural images
    return np.asarray(p[1], np.float32).squeeze()


if 'vindr' in sys.argv[1:]:
    vd = {os.path.basename(f)[4:-4]: lesion_prob(f) for f in glob.glob(f'{ROOT}/pred_vindr/vdr_*.npz')}
    np.savez_compressed(f'{ROOT}/P_vindr.npz', **vd)
    print(f'VinDr-Mammo maps {len(vd)}')
else:
    cb = sorted(glob.glob(f'{ROOT}/pred_cbis/cbis_te_*.npz'))
    P = np.stack([lesion_prob(f) for f in cb]); assert P.shape == (399, 256, 256), P.shape
    np.save(f'{ROOT}/P_cbis_test.npy', P)
    inb = {os.path.basename(f)[4:-4]: lesion_prob(f) for f in glob.glob(f'{ROOT}/pred_inbreast/inb_*.npz')}
    np.savez_compressed(f'{ROOT}/P_inbreast.npz', **inb)
    print(f'CBIS maps {P.shape} | INbreast maps {len(inb)}')
