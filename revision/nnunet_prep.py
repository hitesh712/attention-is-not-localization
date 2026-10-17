"""Prepare an nnU-Net v2 dataset for the full-image detector comparison (Reviewer 2 #3).

Same images and resolution as the Attention U-Net detector: the cached 256x256 CLAHE full mammograms
(2,408 training, 399 test) and their union lesion masks; INbreast mass and BI-RADS 1 images are exported with
the identical full_input() preprocessing for external testing. Fold 0 of splits_final.json is the detector's own
patient-grouped train/validation split, so nnU-Net never validates on a training patient.
"""
import os, sys, json, glob, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, 'G:/My Drive/VS_CC/e2e')
import numpy as np, cv2, pandas as pd, pydicom
from sklearn.model_selection import GroupShuffleSplit
import cbis_data as D

ROOT = 'D:/nnunet'; DS = 'Dataset501_CBISFull'
RAW = f'{ROOT}/nnUNet_raw/{DS}'; PRE = f'{ROOT}/nnUNet_preprocessed/{DS}'
for d in ('imagesTr', 'labelsTr', 'imagesTs', 'imagesTs_inbreast'): os.makedirs(f'{RAW}/{d}', exist_ok=True)
os.makedirs(PRE, exist_ok=True)
_clahe = cv2.createCLAHE(2.0, (8, 8))

Xtr = np.load(D.CACHE + '/tr_X_256.npy', mmap_mode='r'); Ytr = np.load(D.CACHE + '/tr_Y_256.npy', mmap_mode='r')
Xte = np.load(D.CACHE + '/te_X_256.npy', mmap_mode='r')
tr = D.train_rows(); d = tr[tr['mask_path'].notna()]
pats = np.array([rows.iloc[0]['patient_id'] for _, rows in d.groupby('image_key')]); assert len(pats) == len(Xtr)
names = [f'cbis_tr_{i:04d}' for i in range(len(Xtr))]
for i, n in enumerate(names):
    cv2.imwrite(f'{RAW}/imagesTr/{n}_0000.png', np.asarray(Xtr[i])[..., 0])
    cv2.imwrite(f'{RAW}/labelsTr/{n}.png', np.asarray(Ytr[i])[..., 0].astype(np.uint8))
for i in range(len(Xte)): cv2.imwrite(f'{RAW}/imagesTs/cbis_te_{i:04d}_0000.png', np.asarray(Xte[i])[..., 0])

trn, va = next(GroupShuffleSplit(1, test_size=0.15, random_state=2021).split(np.arange(len(Xtr)), groups=pats))
json.dump([{'train': [names[i] for i in sorted(trn)], 'val': [names[i] for i in sorted(va)]}], open(f'{PRE}/splits_final.json', 'w'))
json.dump({'channel_names': {'0': 'mammogram'}, 'labels': {'background': 0, 'lesion': 1},
           'numTraining': len(names), 'file_ending': '.png'}, open(f'{RAW}/dataset.json', 'w'), indent=2)

# INbreast external images (mass images + BI-RADS 1 normals), identical preprocessing to the detector
INB = 'D:/INbreast Release 1.0'
dcms = {m.group(1): f for f in glob.glob(INB + '/AllDICOMs/*.dcm') for m in [re.match(r'(\d+)_', os.path.basename(f))] if m}
meta = pd.read_csv(INB + '/INbreast.csv', sep=None, engine='python')
ids = sorted({os.path.splitext(os.path.basename(x))[0] for x in glob.glob(INB + '/AllXML/*.xml')} & set(dcms))
ids += [str(f) for f in meta.loc[meta['Bi-Rads'].astype(str).str.strip() == '1', 'File Name'] if str(f) in dcms]
for iid in sorted(set(ids)):
    a = pydicom.dcmread(dcms[iid]).pixel_array.astype(np.float32)
    g = ((a - a.min()) / (a.max() - a.min() + 1e-6) * 255).astype(np.uint8)
    cv2.imwrite(f'{RAW}/imagesTs_inbreast/inb_{iid}_0000.png', _clahe.apply(cv2.resize(g, (256, 256))))
print(f'train {len(names)} (fit {len(trn)}, val {len(va)}) | CBIS test {len(Xte)} | INbreast {len(set(ids))}')
