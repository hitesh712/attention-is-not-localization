"""VinDr-Mammo detector inputs: every selected DICOM -> the 256x256 CLAHE image the full-image detectors were trained on
(identical to full_input() in e2e_common.py and to the INbreast export of nnunet_prep.py).

Main variant: display values from the DICOM's own VOI LUT (vindr_data.display_pixels). Sensitivity variant, written
alongside in the same pass: plain min-max of the stored pixel values, the protocol used for INbreast.

Writes  D:/vindr_dicom/cache/X256.npy, X256_minmax.npy (N, 256, 256 uint8) + ids.json     (Attention U-Net, SegFormer)
        <nnU-Net raw>/imagesTs_vindr/vdr_<image id>_0000.png                              (nnU-Net; main variant)
        D:/vindr_dicom/cache/prep_summary.json                                            (scanner census for the paper)
Run with any environment that has cv2 + pydicom (e.g. the Mammo-CLIP venv).
"""
import os, sys, json, time, glob, collections
import numpy as np, cv2, pydicom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vindr_data as VD

CACHE = VD.ROOT + '/cache'; NN = 'D:/nnunet/nnUNet_raw/Dataset501_CBISFull/imagesTs_vindr'
os.makedirs(CACHE, exist_ok=True); os.makedirs(NN, exist_ok=True)
T0 = time.time(); log = lambda *a: print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)
_clahe = cv2.createCLAHE(2.0, (8, 8))
to8 = lambda a: ((a - a.min()) / (a.max() - a.min() + 1e-6) * 255).astype(np.uint8)
band = lambda g: np.diff(np.percentile(g[g > 5], [1, 99]))[0] / 255 if (g > 5).any() else 0.0   # range used by the breast

sel = VD.selection(); A = VD.annotations()
missing = len(VD.selection(existing_only=False)) - len(sel)
log(f'{len(sel)} images on disk' + (f' ({missing} of the selection not downloaded yet)' if missing else ''))
for f in glob.glob(NN + '/vdr_*_0000.png'): os.remove(f)                    # the folder must hold exactly this export
X = np.zeros((len(sel), 256, 256), np.uint8); Xm = np.zeros_like(X); census = collections.Counter(); bad = []; bands = []
for k, r in enumerate(sel):
    d = pydicom.dcmread(VD.path(r['study_id'], r['image_id']))
    g = to8(VD.display_pixels(d, voi=True)); gm = to8(VD.display_pixels(d, voi=False))
    if g.shape != (A[r['image_id']]['H'], A[r['image_id']]['W']): bad.append(r['image_id'])
    X[k] = _clahe.apply(cv2.resize(g, (256, 256))); Xm[k] = _clahe.apply(cv2.resize(gm, (256, 256)))
    cv2.imwrite(f'{NN}/vdr_{r["image_id"]}_0000.png', X[k])
    bands.append((band(g[::8, ::8]), band(gm[::8, ::8])))
    sp = getattr(d, 'ImagerPixelSpacing', None) or getattr(d, 'PixelSpacing', None)
    voi = 'VOI LUT sequence' if 'VOILUTSequence' in d else 'window' if 'WindowCenter' in d else 'none'
    census[(str(getattr(d, 'Manufacturer', '?')).strip().split()[0], f'{float(sp[0]):.3f}' if sp else '?',
            str(getattr(d, 'PhotometricInterpretation', '')).upper(), voi)] += 1
    if (k + 1) % 100 == 0 or k + 1 == len(sel): log(f'{k + 1}/{len(sel)}')
np.save(CACHE + '/X256.npy', X); np.save(CACHE + '/X256_minmax.npy', Xm)
json.dump([[r['study_id'], r['image_id'], r['set']] for r in sel], open(CACHE + '/ids.json', 'w'))
bands = np.array(bands)
S = dict(n=len(sel), n_missing=missing, sets=dict(collections.Counter(r['set'] for r in sel)), size_mismatch_with_annotation=bad,
         breast_intensity_band=dict(voi_lut=float(bands[:, 0].mean()), minmax=float(bands[:, 1].mean())),
         census=[dict(manufacturer=m, pixel_mm=p, photometric=ph, voi=v, n=n) for (m, p, ph, v), n in census.most_common()])
json.dump(S, open(CACHE + '/prep_summary.json', 'w'), indent=2)
log(json.dumps({k: v for k, v in S.items() if k != 'census'})); [log(c) for c in S['census']]
