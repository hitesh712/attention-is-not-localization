"""How much of the oracle-crop result is explained by the crop geometry alone?

Every oracle crop is centred on the lesion and scaled so its longest side spans ~1/1.8 of the crop.
A predictor that ignores the image and outputs a fixed centred disk (diameter 256/1.8) measures what
that geometry gives for free. Also renders VinDr boxes on the full PNGs to check box alignment.
"""
import os, glob, re, plistlib
import numpy as np, cv2, pandas as pd, pydicom
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

SZ, MARGIN, MASK_THRESH = 256, 1.8, 127
OUT = 'G:/My Drive/VS_CC/models/e2e'
yy, xx = np.mgrid[:SZ, :SZ]
DISK = (yy - (SZ - 1) / 2) ** 2 + (xx - (SZ - 1) / 2) ** 2 <= (SZ / MARGIN / 2) ** 2   # pure geometric prior


def dice(gt, pm):
    i = np.count_nonzero(gt & pm); s = np.count_nonzero(gt) + np.count_nonzero(pm)
    return 2 * i / s if s else 1.0


def crop_window(H, W, r0, r1, c0, c1, margin=MARGIN):
    cy, cx = (int(r0) + int(r1)) // 2, (int(c0) + int(c1)) // 2
    half = int(max(int(r1) - int(r0), int(c1) - int(c0)) * margin / 2) + 8
    return max(0, cy - half), min(H, cy + half), max(0, cx - half), min(W, cx + half)


def crop_mask(mask, win):
    ri0, ri1, ci0, ci1 = win
    return cv2.resize(mask[ri0:ri1, ci0:ci1].astype(np.uint8) * 255, (SZ, SZ), interpolation=cv2.INTER_NEAREST) >= 127


def report(name, gts, model_dice=None):
    d = np.array([dice(g, DISK) for g in gts]); pg = np.array([g[SZ // 2, SZ // 2] for g in gts])
    print(f'{name:<9} n={len(gts):>4} | centred-disk prior: Dice {d.mean():.3f} (median {np.median(d):.3f}, '
          f'>0.5 {np.mean(d > 0.5) * 100:.1f}%) | centre-pixel pointing {pg.mean() * 100:.1f}%'
          + (f' | model {model_dice}' if model_dice else ''))
    return d


# ---- CBIS: cached oracle GT crops
Y = np.load('D:/cbis-ddsm/_seg_cache/roite_Y_256.npy', mmap_mode='r'); v = np.load('D:/cbis-ddsm/_seg_cache/roite_valid_256.npy')
d_cbis = report('CBIS', [np.asarray(Y[j])[..., 0].astype(bool) for j in np.where(v)[0]], 'Dice 0.871, PG 98.8%')

# ---- INbreast: rebuild the oracle GT crops (union of masses in the window, as published)
INB = 'D:/INbreast Release 1.0'
dcms = {re.match(r'(\d+)_', os.path.basename(f)).group(1): f for f in glob.glob(INB + '/AllDICOMs/*.dcm')
        if re.match(r'(\d+)_', os.path.basename(f))}


def parse_masses(xml_path):
    with open(xml_path, 'rb') as fh: pl = plistlib.load(fh, fmt=plistlib.FMT_XML)
    out = []
    for r in pl['Images'][0]['ROIs']:
        if str(r.get('Name', '')).strip().lower() == 'mass':
            pts = [tuple(map(float, re.findall(r'[-\d.]+', s)[:2])) for s in r.get('Point_px', [])]
            pts = [p for p in pts if len(p) == 2]
            if len(pts) >= 3: out.append(np.array(pts, np.int32))
    return out


gts = []
for xp in sorted(glob.glob(INB + '/AllXML/*.xml')):
    iid = os.path.splitext(os.path.basename(xp))[0]
    if iid not in dcms: continue
    try: masses = parse_masses(xp)
    except Exception: continue
    if not masses: continue
    ds = pydicom.dcmread(dcms[iid], stop_before_pixels=True); H, W = int(ds.Rows), int(ds.Columns)
    full = np.zeros((H, W), np.uint8)
    for poly in masses: cv2.fillPoly(full, [poly], 1)
    for poly in masses:
        g = crop_mask(full, crop_window(H, W, poly[:, 1].min(), poly[:, 1].max(), poly[:, 0].min(), poly[:, 0].max()))
        if g.any(): gts.append(g)
d_inb = report('INbreast', gts, 'Dice 0.797, PG 100%')

# ---- VinDr: box-level metrics of the centred disk in the published oracle windows
VROOT = 'D:/pipeline2_mammogram/pipeline2_mammogram/data/vindrmammo'
df = pd.read_csv(VROOT + '/finding_annotations.csv')
mass = df[df['finding_categories'].astype(str).str.contains('Mass') & df['xmin'].notna()].reset_index(drop=True)
paths = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(VROOT + '/images_png/*/*.png')}
shape = {}
pg, cont, biou = [], [], []
dy, dx = np.where(DISK); dbox = (dx.min(), dy.min(), dx.max(), dy.max())


def box_iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1]); ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


for r in mass.itertuples(index=False):
    if r.image_id not in paths: continue
    if r.image_id not in shape: shape[r.image_id] = cv2.imread(paths[r.image_id], cv2.IMREAD_GRAYSCALE).shape
    H, W = shape[r.image_id]; sx, sy = W / int(r.width), H / int(r.height)
    x0, y0, x1, y1 = r.xmin * sx, r.ymin * sy, r.xmax * sx, r.ymax * sy
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; half = int(max(x1 - x0, y1 - y0) * MARGIN / 2) + 8
    ri0, ri1 = max(0, int(cy - half)), min(H, int(cy + half)); ci0, ci1 = max(0, int(cx - half)), min(W, int(cx + half))
    ch, cw = ri1 - ri0, ci1 - ci0
    if ch <= 0 or cw <= 0: continue
    bm = np.zeros((SZ, SZ), np.uint8)
    cv2.rectangle(bm, (int((x0 - ci0) / cw * SZ), int((y0 - ri0) / ch * SZ)), (int((x1 - ci0) / cw * SZ), int((y1 - ri0) / ch * SZ)), 1, -1)
    bm = bm.astype(bool)
    if not bm.any(): continue
    ys, xs = np.where(bm)
    pg.append(bm[SZ // 2, SZ // 2]); cont.append((DISK & bm).sum() / DISK.sum())
    biou.append(box_iou(dbox, (xs.min(), ys.min(), xs.max(), ys.max())))
print(f'VinDr     n={len(pg):>4} | centred-disk prior: pointing {np.mean(pg) * 100:.1f}% | containment {np.mean(cont):.3f} | '
      f'box-IoU {np.mean(biou):.3f} (>0.5 {np.mean(np.array(biou) > 0.5) * 100:.1f}%) | model PG 89.6%, cont 0.797, box-IoU 0.587')
print('VinDr PNG shapes seen:', sorted(set(shape.values()))[:5], '... unique', len(set(shape.values())))

# ---- VinDr alignment check: BI-RADS 4/5 masses, box drawn on the full PNG + zoom
sel = mass[mass['finding_birads'].astype(str).str.contains('4|5') & mass['image_id'].isin(paths.keys())]
sel = sel.iloc[np.random.RandomState(1).choice(len(sel), 6, replace=False)]
fig, ax = plt.subplots(2, 6, figsize=(16, 7), dpi=110)
for k, r in enumerate(sel.itertuples(index=False)):
    g = cv2.imread(paths[r.image_id], cv2.IMREAD_GRAYSCALE); H, W = g.shape; sx, sy = W / int(r.width), H / int(r.height)
    x0, y0, x1, y1 = r.xmin * sx, r.ymin * sy, r.xmax * sx, r.ymax * sy
    ax[0, k].imshow(g, cmap='gray'); ax[0, k].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color='lime', lw=1.5))
    ax[0, k].set_title(f'{r.laterality}-{r.view_position} {r.finding_birads}\norig {int(r.height)}x{int(r.width)}', fontsize=7)
    m = int(max(x1 - x0, y1 - y0) * 1.5)
    a0, a1 = max(0, int(y0) - m), min(H, int(y1) + m); b0, b1 = max(0, int(x0) - m), min(W, int(x1) + m)
    ax[1, k].imshow(g[a0:a1, b0:b1], cmap='gray')
    ax[1, k].add_patch(plt.Rectangle((x0 - b0, y0 - a0), x1 - x0, y1 - y0, fill=False, color='lime', lw=1.5))
    for a in ax[:, k]: a.axis('off')
plt.tight_layout(); plt.savefig(OUT + '/diag_vindr_alignment.png', bbox_inches='tight')
print('saved alignment panel')
