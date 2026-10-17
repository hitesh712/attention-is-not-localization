"""CBIS-DDSM path resolution and the patient-disjoint split, verbatim from the training notebooks.

test_images(): the 399 masked test mammograms in the exact order used by every published evaluation
               (rows with a resolvable ROI mask, grouped by image, sorted by image key).
test_rows():   one row per ROI (the 422-lesion list of cbis_roi_seg.ipynb), with metadata.
train_rows():  training rows with the 31 patients shared with the official test split removed.
"""
import os
import numpy as np, pandas as pd, cv2

DATA_ROOT = 'D:/cbis-ddsm'; JPEG_ROOT = DATA_ROOT + '/jpeg'; CSV_DIR = DATA_ROOT + '/csv'
DICOM_INFO = CSV_DIR + '/dicom_info.csv'; CACHE = DATA_ROOT + '/_seg_cache'
TRAIN_CSVS = ['mass_case_description_train_set.csv', 'calc_case_description_train_set.csv']
TEST_CSVS = ['mass_case_description_test_set.csv', 'calc_case_description_test_set.csv']
MASK_THRESH = 127


def _to_local(p):
    p = str(p).replace('\\', '/'); i = p.find('jpeg/')
    return os.path.join(JPEG_ROOT, p[i + 5:]) if i >= 0 else p


def _uid_maps():
    di = pd.read_csv(DICOM_INFO); di['SeriesDescription'] = di['SeriesDescription'].fillna('')
    di['local'] = di['image_path'].apply(_to_local); di['uid'] = di['SeriesInstanceUID'].astype(str)
    full, mask, allf = {}, {}, {}
    for _, r in di.iterrows():
        allf.setdefault(r['uid'], []).append(r['local']); d = r['SeriesDescription'].lower()
        if 'full' in d: full.setdefault(r['uid'], r['local'])
        elif 'roi mask' in d: mask.setdefault(r['uid'], r['local'])
    return full, mask, allf


FULL_MAP, MASK_MAP, ANY_MAP = _uid_maps()


def series_uid(p):
    q = str(p).replace('\\', '/').strip().split('/'); return q[-2] if len(q) >= 2 else None


def is_mask_like(g):
    return g is not None and np.mean((g <= 10) | (g >= 245)) > 0.90 and len(np.unique(g)) <= 64


def resolve_mask(p):
    u = series_uid(p)
    if u in MASK_MAP: return MASK_MAP[u]
    for c in ANY_MAP.get(u, []):
        if is_mask_like(cv2.imread(c, cv2.IMREAD_GRAYSCALE)): return c
    return None


def _tables(names):
    df = pd.concat([pd.read_csv(os.path.join(CSV_DIR, n)) for n in names
                    if os.path.exists(os.path.join(CSV_DIR, n))], ignore_index=True)
    df.columns = [c.strip().replace(' ', '_').lower() for c in df.columns]
    df['label'] = df['abnormality_type'].astype(str).str.upper().str.contains('MASS').astype(int)   # 1 = mass
    df['full_path'] = df['image_file_path'].apply(lambda p: FULL_MAP.get(series_uid(p)))
    df['mask_path'] = df['roi_mask_file_path'].apply(resolve_mask)
    df['patient_id'] = df['patient_id'].astype(str); df['image_key'] = df['image_file_path'].astype(str)
    return df.dropna(subset=['full_path']).reset_index(drop=True)


_TEST = _tables(TEST_CSVS)


def test_rows():
    return _TEST.dropna(subset=['mask_path']).reset_index(drop=True)


def test_images():
    """[(image_key, full_path, [mask_paths], patient_id, label_of_first_row)] in the published order."""
    d = _TEST[_TEST['mask_path'].notna()]
    return [(k, r.iloc[0]['full_path'], list(r['mask_path']), r.iloc[0]['patient_id'], int(r.iloc[0]['label']))
            for k, r in d.groupby('image_key')]


def train_rows():
    tr = _tables(TRAIN_CSVS)
    return tr[~tr['patient_id'].isin(set(_TEST['patient_id']))].reset_index(drop=True)


def read_mask(path, shape):
    """Binary mask at the image's native resolution (None if the file is not mask-like)."""
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE) if path else None
    if not is_mask_like(m): return None
    if m.shape != shape: m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m >= MASK_THRESH
