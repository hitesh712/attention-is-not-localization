"""Model-free alignment test #2. A correctly placed mass annotation should be locally denser than
same-sized boxes placed at random elsewhere in the same breast. For each lesion we compute the
percentile of its local contrast among 30 random same-size boxes in the same image:
aligned annotations -> percentiles well above 0.5; misplaced annotations -> ~uniform (mean ~0.5)."""
import os, glob, re, plistlib
import numpy as np, cv2, pandas as pd, pydicom

BG = 10; NRAND = 30; rs = np.random.RandomState(0)


def contrast(img, x0, y0, x1, y1):
    H, W = img.shape
    a, b, c, d = max(0, int(y0)), min(H, int(np.ceil(y1)) + 1), max(0, int(x0)), min(W, int(np.ceil(x1)) + 1)
    if b - a < 2 or d - c < 2: return None
    ph, pw = max(2, (b - a) // 2), max(2, (d - c) // 2)
    A, B, C, D = max(0, a - ph), min(H, b + ph), max(0, c - pw), min(W, d + pw)
    ring = np.ones((B - A, D - C), bool); ring[a - A:b - A, c - C:d - C] = False
    box, rv = img[a:b, c:d].astype(np.float32), img[A:B, C:D].astype(np.float32)[ring]
    box, rv = box[box > BG], rv[rv > BG]
    if box.size < 4 or rv.size < 4: return None
    return (box.mean() - rv.mean()) / (rv.std() + 1e-6)


def percentile(img, x0, y0, x1, y1):
    c0 = contrast(img, x0, y0, x1, y1)
    if c0 is None: return None
    H, W = img.shape; bw, bh = x1 - x0, y1 - y0
    ys, xs = np.where(img > BG)
    null = []
    for _ in range(NRAND * 4):
        k = rs.randint(len(ys)); cy, cx = ys[k], xs[k]
        nx0, ny0 = cx - bw / 2, cy - bh / 2
        if nx0 < 0 or ny0 < 0 or nx0 + bw >= W or ny0 + bh >= H: continue
        if np.mean(img[int(ny0):int(ny0 + bh), int(nx0):int(nx0 + bw)] > BG) < 0.95: continue
        v = contrast(img, nx0, ny0, nx0 + bw, ny0 + bh)
        if v is not None: null.append(v)
        if len(null) >= NRAND: break
    return float(np.mean(np.array(null) < c0)) if len(null) >= 10 else None


def summarise(name, p):
    p = np.array([q for q in p if q is not None])
    print(f'{name:<22} n={len(p):>4} | mean percentile {p.mean():.2f} | median {np.median(p):.2f} | '
          f'top-decile {np.mean(p >= 0.9) * 100:5.1f}% | bottom-half {np.mean(p < 0.5) * 100:5.1f}%   (random placement: 0.50 / 10% / 50%)')


# INbreast masses (aligned control)
INB = 'D:/INbreast Release 1.0'
dcms = {m.group(1): f for f in glob.glob(INB + '/AllDICOMs/*.dcm') for m in [re.match(r'(\d+)_', os.path.basename(f))] if m}
pi = []
for xp in sorted(glob.glob(INB + '/AllXML/*.xml')):
    iid = os.path.splitext(os.path.basename(xp))[0]
    if iid not in dcms: continue
    with open(xp, 'rb') as fh: pl = plistlib.load(fh, fmt=plistlib.FMT_XML)
    polys = []
    for roi in pl['Images'][0]['ROIs']:
        if str(roi.get('Name', '')).strip().lower() == 'mass':
            pts = [tuple(map(float, re.findall(r'[-\d.]+', s)[:2])) for s in roi.get('Point_px', [])]
            pts = [q for q in pts if len(q) == 2]
            if len(pts) >= 3: polys.append(np.array(pts))
    if not polys: continue
    a = pydicom.dcmread(dcms[iid]).pixel_array.astype(np.float32)
    g = ((a - a.min()) / (a.max() - a.min() + 1e-6) * 255).astype(np.uint8)
    g = cv2.resize(g, (g.shape[1] // 4, g.shape[0] // 4), interpolation=cv2.INTER_AREA)   # speed; geometry scaled below
    for q in polys:
        pi.append(percentile(g, q[:, 0].min() / 4, q[:, 1].min() / 4, q[:, 0].max() / 4, q[:, 1].max() / 4))
summarise('INbreast masses (ctrl)', pi)

# CBIS test masses (aligned control): bbox of the ROI mask on the full mammogram
CB = 'D:/cbis-ddsm'
di = pd.read_csv(CB + '/csv/dicom_info.csv'); di['SeriesDescription'] = di['SeriesDescription'].fillna('')
loc = lambda p: os.path.join(CB + '/jpeg', str(p).replace('\\', '/').split('jpeg/')[-1])
full = {str(r.SeriesInstanceUID): loc(r.image_path) for r in di.itertuples() if 'full' in r.SeriesDescription.lower()}
roim = {str(r.SeriesInstanceUID): loc(r.image_path) for r in di.itertuples() if 'roi mask' in r.SeriesDescription.lower()}
uid = lambda p: str(p).replace('\\', '/').strip().split('/')[-2]
t = pd.read_csv(CB + '/csv/mass_case_description_test_set.csv')
pc = []
for r in t.sample(150, random_state=0).itertuples(index=False):
    fp, mp = full.get(uid(r[11])), roim.get(uid(r[13]))
    if not fp or not mp: continue
    g = cv2.imread(fp, cv2.IMREAD_GRAYSCALE); m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
    if g is None or m is None: continue
    if not (np.mean((m <= 10) | (m >= 245)) > 0.90 and len(np.unique(m)) <= 64): continue   # is_mask_like guard
    if m.shape != g.shape: m = cv2.resize(m, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(m >= 127)
    if not len(ys): continue
    g = cv2.resize(g, (g.shape[1] // 4, g.shape[0] // 4), interpolation=cv2.INTER_AREA)
    pc.append(percentile(g, xs.min() / 4, ys.min() / 4, xs.max() / 4, ys.max() / 4))
summarise('CBIS masses (ctrl)', pc)

# VinDr masses: boxes scaled from original DICOM dims to the 1520x912 PNG (the original protocol)
VROOT = 'D:/pipeline2_mammogram/pipeline2_mammogram/data/vindrmammo'
df = pd.read_csv(VROOT + '/finding_annotations.csv')
mass = df[df['finding_categories'].astype(str).str.contains('Mass') & df['xmin'].notna()]
paths = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(VROOT + '/images_png/*/*.png')}
pv = []
for r in mass.itertuples(index=False):
    if r.image_id not in paths: continue
    g = cv2.imread(paths[r.image_id], cv2.IMREAD_GRAYSCALE); H, W = g.shape; sx, sy = W / int(r.width), H / int(r.height)
    pv.append(percentile(g, r.xmin * sx, r.ymin * sy, r.xmax * sx, r.ymax * sy))
summarise('VinDr masses', pv)
