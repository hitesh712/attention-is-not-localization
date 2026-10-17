"""Model-free alignment test: how much of each annotated lesion falls on image background (outside the breast)?
Correctly registered mass annotations should essentially never cover background."""
import os, glob, re, plistlib
import numpy as np, cv2, pandas as pd, pydicom

BG = 10   # 8-bit intensity treated as background


def frac_bg(img, x0, y0, x1, y1):
    H, W = img.shape
    a, b = max(0, int(y0)), min(H, int(np.ceil(y1)) + 1); c, d = max(0, int(x0)), min(W, int(np.ceil(x1)) + 1)
    if b <= a or d <= c: return 1.0                        # box entirely outside the image
    return float(np.mean(img[a:b, c:d] <= BG))


VROOT = 'D:/pipeline2_mammogram/pipeline2_mammogram/data/vindrmammo'
df = pd.read_csv(VROOT + '/finding_annotations.csv')
mass = df[df['finding_categories'].astype(str).str.contains('Mass') & df['xmin'].notna()]
paths = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(VROOT + '/images_png/*/*.png')}
fv, cache = [], {}
for r in mass.itertuples(index=False):
    if r.image_id not in paths: continue
    if r.image_id not in cache:
        cache = {r.image_id: cv2.imread(paths[r.image_id], cv2.IMREAD_GRAYSCALE)}
    g = cache[r.image_id]; H, W = g.shape; sx, sy = W / int(r.width), H / int(r.height)
    fv.append(frac_bg(g, r.xmin * sx, r.ymin * sy, r.xmax * sx, r.ymax * sy))
fv = np.array(fv)

INB = 'D:/INbreast Release 1.0'
dcms = {m.group(1): f for f in glob.glob(INB + '/AllDICOMs/*.dcm') for m in [re.match(r'(\d+)_', os.path.basename(f))] if m}
fi = []
for xp in sorted(glob.glob(INB + '/AllXML/*.xml')):
    iid = os.path.splitext(os.path.basename(xp))[0]
    if iid not in dcms: continue
    with open(xp, 'rb') as fh: pl = plistlib.load(fh, fmt=plistlib.FMT_XML)
    polys = []
    for roi in pl['Images'][0]['ROIs']:
        if str(roi.get('Name', '')).strip().lower() == 'mass':
            pts = [tuple(map(float, re.findall(r'[-\d.]+', s)[:2])) for s in roi.get('Point_px', [])]
            pts = [p for p in pts if len(p) == 2]
            if len(pts) >= 3: polys.append(np.array(pts))
    if not polys: continue
    a = pydicom.dcmread(dcms[iid]).pixel_array.astype(np.float32)
    g = ((a - a.min()) / (a.max() - a.min() + 1e-6) * 255).astype(np.uint8)
    for p in polys:
        fi.append(frac_bg(g, p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()))
fi = np.array(fi)

for name, f in (('INbreast (control)', fi), ('VinDr', fv)):
    print(f'{name:<19} n={len(f):>4} | lesion boxes with >10% background: {np.mean(f > 0.10) * 100:5.1f}% | '
          f'>50% background: {np.mean(f > 0.50) * 100:5.1f}% | mean background fraction {f.mean():.3f}')
