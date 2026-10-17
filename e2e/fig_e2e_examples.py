"""End-to-end examples chosen by rule (not by eye), CBIS-DDSM (top) and INbreast (bottom):
  (a) the detected lesion whose end-to-end Dice is closest to the median of detected lesions,
  (b) the missed lesion of median size,
  (c) the first image (CBIS: test order; INbreast: first BI-RADS 1 normal by file id) with a false-positive detection.
Green: ground truth; yellow: detector boxes; red: end-to-end mask. TF venv (uses e2e_common).
"""
import os, sys, glob, re, plistlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, 'G:/My Drive/VS_CC/revision')
from e2e_common import *
import pydicom, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import cbis_data as D

E2E = MODELS + '/e2e'; FIG = 'D:/Research 9.9.26/current research/IEEE_Access/figures'
roi = build_model(W_ROI); det = build_model(W_DET)


PEAKS = {}


def run_e2e(gray, p):
    H, W = gray.shape; canvas = np.zeros((H, W), bool); boxes = []; peaks = []
    for c in candidates(p):
        b = grid_to_full_bbox(c['bbox'], H, W); win = crop_window(H, W, *b); boxes.append(win)
        peaks.append(grid_to_full_pt(*c['peak'], H, W))
        paste(canvas, predict(roi, crop_input(gray, win)[None])[0], win)
    PEAKS['last'] = peaks                                     # detection peaks: the points the evaluation matches
    return canvas, boxes


def panel(ax, gray, gt, canvas, boxes, focus, title):
    H, W = gray.shape
    r0, r1, c0, c1 = focus if focus else (0, H, 0, W)
    ax.imshow(gray[r0:r1, c0:c1], cmap='gray'); ax.set_title(title, fontsize=7.5); ax.axis('off')
    for m, col in ((gt, 'lime'), (canvas, 'red')):
        if m is not None and m[r0:r1, c0:c1].any(): ax.contour(m[r0:r1, c0:c1].astype(float), levels=[0.5], colors=col, linewidths=0.8)
    for (a, b, c, d) in boxes:
        ax.add_patch(plt.Rectangle((c - c0, a - r0), d - c, b - a, fill=False, ec='yellow', lw=0.7, ls='--'))


def breast_box(gray, pad=40):
    ys, xs = np.where(gray > np.percentile(gray, 60)); H, W = gray.shape
    return max(0, ys.min() - pad), min(H, ys.max() + pad), max(0, xs.min() - pad), min(W, xs.max() + pad)


fig, axs = plt.subplots(2, 3, figsize=(7.1, 5.2))
# ------------------------------------------------------------------ CBIS
Z = np.load(E2E + '/cbis_e2e_pub.npz'); imgs = D.test_images(); P = np.load(D.CACHE + '/e2e_te_P_256.npy', mmap_mode='r')
dt = Z['les_detected'].astype(bool); dall = Z['les_d_all']; li = Z['les_i'].astype(int)
a = np.where(dt)[0][np.argmin(np.abs(dall[dt] - np.median(dall[dt])))]
ms = np.where(~dt)[0]; b = ms[np.argsort(Z['les_area'][ms])[len(ms) // 2]]
def cbis_case(i):
    key, fp, mps, pid, lab = imgs[i]; gray = cv2.imread(fp, cv2.IMREAD_GRAYSCALE); H, W = gray.shape
    gt = np.zeros((H, W), bool)
    for mp in mps:
        m = D.read_mask(mp, (H, W))
        if m is not None: gt |= m
    canvas, boxes = run_e2e(gray, np.asarray(P[i], np.float32)); return gray, gt, canvas, boxes
for col, (k, tag) in enumerate(((a, 'detected'), (b, 'missed'))):
    gray, gt, canvas, boxes = cbis_case(li[k])
    panel(axs[0, col], gray, gt, canvas, boxes, breast_box(gray), f'CBIS-DDSM, {tag} lesion (Dice {dall[k]:.2f})')
for i in range(len(imgs)):                                   # first image with a false-positive detection
    gray, gt, canvas, boxes = cbis_case(i)
    fps = [pk for pk in PEAKS['last'] if not gt[pk]]          # same pointing criterion as the evaluation
    if fps and len(boxes) > 1:
        panel(axs[0, 2], gray, gt, canvas, boxes, breast_box(gray), f'CBIS-DDSM, false positive ({len(fps)} of {len(boxes)} boxes)'); break
# ------------------------------------------------------------------ INbreast
INB = 'D:/INbreast Release 1.0'
dcms = {m.group(1): f for f in glob.glob(INB + '/AllDICOMs/*.dcm') for m in [re.match(r'(\d+)_', os.path.basename(f))] if m}
def read_dcm(iid):
    a_ = pydicom.dcmread(dcms[iid]).pixel_array.astype(np.float32); return ((a_ - a_.min()) / (a_.max() - a_.min() + 1e-6) * 255).astype(np.uint8)
def masses(iid):
    with open(f'{INB}/AllXML/{iid}.xml', 'rb') as fh: pl = plistlib.load(fh, fmt=plistlib.FMT_XML)
    out = []
    for r in pl['Images'][0]['ROIs']:
        if str(r.get('Name', '')).strip().lower() == 'mass':
            pts = [tuple(map(float, re.findall(r'[-\d.]+', s)[:2])) for s in r.get('Point_px', [])]; pts = [q for q in pts if len(q) == 2]
            if len(pts) >= 3: out.append(np.array(pts, np.int32))
    return out
ZI = np.load(E2E + '/inbreast_e2e_pub.npz'); dti = ZI['les_detected'].astype(bool); dalli = ZI['les_d_all']; liid = ZI['les_iid'].astype(str)
ai = np.where(dti)[0][np.argmin(np.abs(dalli[dti] - np.median(dalli[dti])))]
msi = np.where(~dti)[0]; bi = msi[np.argsort(ZI['les_area'][msi])[len(msi) // 2]]
def inb_case(iid, with_gt=True):
    gray = read_dcm(iid); H, W = gray.shape; gt = np.zeros((H, W), np.uint8)
    if with_gt: cv2.fillPoly(gt, masses(iid), 1)
    canvas, boxes = run_e2e(gray, predict(det, full_input(gray)[None])[0]); return gray, gt.astype(bool), canvas, boxes
for col, (k, tag) in enumerate(((ai, 'detected'), (bi, 'missed'))):
    gray, gt, canvas, boxes = inb_case(liid[k])
    panel(axs[1, col], gray, gt, canvas, boxes, breast_box(gray), f'INbreast, {tag} mass (Dice {dalli[k]:.2f})')
meta = pd.read_csv(INB + '/INbreast.csv', sep=None, engine='python')
normals = sorted(str(f) for f in meta.loc[meta['Bi-Rads'].astype(str).str.strip() == '1', 'File Name'] if str(f) in dcms)
for iid in normals:
    gray, gt, canvas, boxes = inb_case(iid, with_gt=False)
    if canvas.any():
        panel(axs[1, 2], gray, None, canvas, boxes, breast_box(gray), f'INbreast, BI-RADS 1 normal ({len(boxes)} false positive{"s" if len(boxes) > 1 else ""})'); break
fig.tight_layout(h_pad=0.6, w_pad=0.4)
fig.savefig(FIG + '/FIG_e2e_examples.pdf', bbox_inches='tight', dpi=300); fig.savefig(FIG + '/FIG_e2e_examples.png', bbox_inches='tight', dpi=200)
print('saved FIG_e2e_examples')
