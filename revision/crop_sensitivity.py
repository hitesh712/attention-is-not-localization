"""Crop-size sensitivity (Reviewer 2 #8): conditional delineation, the geometric prior and the end-to-end pipeline
at crop margins 1.4, 1.8 (original), 2.4 and 3.0, for the original and the pipeline-aware delineator.

Conditional + prior are scored in each margin's own oracle crop (as a conditional protocol would report them);
end-to-end is always scored in the fixed 1.8 oracle window so the numbers are comparable with the main results.
"""
import os, sys, json, glob, re, plistlib, time, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, 'G:/My Drive/VS_CC/e2e')
from e2e_common import *
import pydicom
import cbis_data as D

MARGINS = (1.4, 1.8, 2.4, 3.0); MODELS_ = 'G:/My Drive/VS_CC/models'; OUT = MODELS_ + '/revision'
ROIS = {'original': W_ROI, 'pipeline_aware': MODELS_ + '/cbis_roi_seg_jitter_s2021.weights.h5'}
ROIS = {k: v for k, v in ROIS.items() if os.path.exists(v)}
T0 = time.time(); log = lambda *a: print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)
yy, xx = np.mgrid[:SZ, :SZ]
prior = {m: (yy - (SZ - 1) / 2) ** 2 + (xx - (SZ - 1) / 2) ** 2 <= (SZ / m / 2) ** 2 for m in MARGINS}


def run(cases, read_image, detector_maps, models):
    """cases: list of (key, [lesion masks fn]) ; returns per-margin means."""
    R = {m: {k: [] for k in ['prior'] + [f'cond_{n}' for n in models] + [f'e2e_{n}' for n in models]} for m in MARGINS}
    for key, lesion_masks in cases:
        gray = read_image(key); H, W = gray.shape; p = detector_maps[key]
        lm = [m for m in lesion_masks(H, W) if m.any()]
        if not lm: continue
        gt = np.logical_or.reduce(lm)
        cands = candidates(p)[:10]
        for m in MARGINS:
            dwins = [crop_window(H, W, *grid_to_full_bbox(c['bbox'], H, W), margin=m) for c in cands]
            owins = [crop_window(H, W, *mask_bbox(l), margin=m) for l in lm]
            evals = [crop_window(H, W, *mask_bbox(l), margin=1.8) for l in lm]
            X = np.stack([crop_input(gray, w) for w in dwins + owins])
            for name, mdl in models.items():
                pr = predict(mdl, X); canvas = np.zeros((H, W), bool)
                for k, w in enumerate(dwins): paste(canvas, pr[k], w)
                for q, (ow, ew, l) in enumerate(zip(owins, evals, lm)):
                    gtc = crop_mask(l, ow)
                    R[m][f'cond_{name}'].append(dice_iou(gtc, pr[len(dwins) + q] >= THR)[0])
                    R[m][f'e2e_{name}'].append(dice_iou(crop_mask(l, ew), crop_mask(canvas, ew))[0])
                    if name == next(iter(models)): R[m]['prior'].append(dice_iou(gtc, prior[m])[0])
        del gray; gc.collect()
    return {str(m): {k: float(np.mean(v)) for k, v in R[m].items()} | {'n': len(R[m]['prior'])} for m in MARGINS}


models = {n: build_model(w) for n, w in ROIS.items()}
det = build_model(W_DET); log('models:', list(ROIS))

# ---- CBIS test (per image; each lesion is scored with its own mask, as in the per-lesion analysis)
Xte = np.load(D.CACHE + '/te_X_256.npy', mmap_mode='r')
imgs = D.test_images(); Pc = {k: predict(det, Xte[i:i + 1])[0] for i, (k, *_r) in enumerate(imgs)}
rows = D.test_rows(); by_img = rows.groupby('image_key')['mask_path'].apply(list).to_dict()
fp_of = {k: fp for k, fp, *_r in imgs}
cases = [(k, (lambda k: lambda H, W: [m for m in (D.read_mask(mp, (H, W)) for mp in by_img[k]) if m is not None])(k)) for k, *_r in imgs]
res = {'cbis': run(cases, lambda k: cv2.imread(fp_of[k], cv2.IMREAD_GRAYSCALE), Pc, models)}
log('CBIS done', json.dumps(res['cbis']))

# ---- INbreast masses
INB = 'D:/INbreast Release 1.0'
dcms = {m_.group(1): f for f in glob.glob(INB + '/AllDICOMs/*.dcm') for m_ in [re.match(r'(\d+)_', os.path.basename(f))] if m_}
def read_dcm(iid):
    a = pydicom.dcmread(dcms[iid]).pixel_array.astype(np.float32); return ((a - a.min()) / (a.max() - a.min() + 1e-6) * 255).astype(np.uint8)
def polys(xp):
    with open(xp, 'rb') as fh: pl = plistlib.load(fh, fmt=plistlib.FMT_XML)
    out = []
    for r in pl['Images'][0]['ROIs']:
        if str(r.get('Name', '')).strip().lower() == 'mass':
            pts = [tuple(map(float, re.findall(r'[-\d.]+', s)[:2])) for s in r.get('Point_px', [])]; pts = [q for q in pts if len(q) == 2]
            if len(pts) >= 3: out.append(np.array(pts, np.int32))
    return out
icases, Pi = [], {}
for xp in sorted(glob.glob(INB + '/AllXML/*.xml')):
    iid = os.path.splitext(os.path.basename(xp))[0]
    if iid not in dcms: continue
    try: ps = polys(xp)
    except Exception: continue
    if not ps: continue
    Pi[iid] = predict(det, full_input(read_dcm(iid))[None])[0]
    def lmasks(H, W, ps=ps):
        out = []
        for q in ps:
            m_ = np.zeros((H, W), np.uint8); cv2.fillPoly(m_, [q], 1); out.append(m_.astype(bool))
        return out
    icases.append((iid, lmasks))
res['inbreast'] = run(icases, read_dcm, Pi, models)
log('INbreast done', json.dumps(res['inbreast']))
json.dump(res, open(OUT + '/crop_sensitivity.json', 'w'), indent=2)
