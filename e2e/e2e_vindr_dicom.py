"""Detection and box-level end-to-end evaluation on VinDr-Mammo (second external cohort, original DICOMs from
PhysioNet; CBIS-trained models, no retraining). Replaces the withdrawn PNG-based analysis (e2e_vindr.py).

VinDr-Mammo provides bounding boxes, not outlines, so everything is scored at box level:
  * detection: a detection whose peak lies inside the box of a mass or suspicious calcification is a true positive
    (pointing criterion, greedy by score; duplicates ignored); a detection inside the box of any OTHER annotated
    finding (asymmetry, distortion, ...) is ignored, because it is neither a target nor a false alarm; every other
    detection is a false positive. Operating point (threshold 0.5), local-maximum FROC and CPM as for CBIS / INbreast.
  * 300 BI-RADS 1 images of the official test split: false positives per normal mammogram.
  * pointing per image: the detector's global maximum inside a target box.
  * with --roi, the pipeline at box level, per lesion and in the lesion's evaluation window (annotation-centred,
    margin 1.8, as for the Dice analyses): IoU between the annotated box and the bounding box of the prediction, for
    the detector alone, the end-to-end pipeline, conditional delineation (annotation-centred crop) and the geometric
    prior (centred disk); plus the fraction of the predicted pixels inside the box (containment).
Detector inputs come from vindr_prep.py (256x256 CLAHE, as trained); DICOMs are read only for the delineator.
"""
import os, sys, json, time, gc, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, 'G:/My Drive/VS_CC/revision')
from e2e_common import *                       # sets DML_VISIBLE_DEVICES before importing TF
import vindr_data as VD

ap = argparse.ArgumentParser()
ap.add_argument('--roi', default='', help="delineator weights; '' = detection only (no DICOM is read)")
ap.add_argument('--det', default=W_DET, help='detector weights (default: the published v3 model)')
ap.add_argument('--pmap', default='', help='precomputed detector probability maps: .npz keyed by VinDr image id')
ap.add_argument('--tag', default='pub')
ap.add_argument('--limit', type=int, default=0, help='first N lesion images and N normals (smoke test)')
ap.add_argument('--minmax', action='store_true', help='sensitivity analysis: plain min-max of the stored pixel values '
                                                       '(the INbreast protocol) instead of the DICOM VOI LUT')
args = ap.parse_args()
if args.minmax: VD.VOI = False
OUT = MODELS + '/e2e'; KMAX = 10
T0 = time.time(); log = lambda *a: print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)
yy_, xx_ = np.mgrid[:SZ, :SZ]; PRIOR = (yy_ - (SZ - 1) / 2) ** 2 + (xx_ - (SZ - 1) / 2) ** 2 <= (SZ / MARGIN / 2) ** 2

ids = json.load(open(VD.ROOT + '/cache/ids.json'))
X = np.load(VD.ROOT + '/cache/X256' + ('_minmax' if args.minmax else '') + '.npy', mmap_mode='r')
row_of = {i: k for k, (_, i, _) in enumerate(ids)}; A = VD.annotations()
lesion = [(s, i, st) for s, i, st in ids if st in VD.LESION_SETS and any(b['target'] for b in A[i]['boxes'])]
normals = [(s, i, st) for s, i, st in ids if st == VD.NORMAL_SET]
if args.limit: lesion, normals = lesion[:args.limit], normals[:args.limit]
log(f'lesion images {len(lesion)} | BI-RADS 1 normal images {len(normals)}')

# ------------------------------------------------------------------ detector maps (256x256)
todo = [i for _, i, _ in lesion + normals]; P = {}
if args.pmap:
    Zp = np.load(args.pmap)
    for i in todo: P[i] = np.asarray(Zp[i], np.float32)
    log('detector probability maps loaded from', os.path.basename(args.pmap))
else:
    det = build_model(args.det); log('detector weights:', os.path.basename(args.det))
    for b in range(0, len(todo), 32):
        ch = todo[b:b + 32]; pr = predict(det, np.stack([np.repeat(np.asarray(X[row_of[i]])[..., None], 3, -1) for i in ch]))
        for i, q in zip(ch, pr): P[i] = q
    del det; gc.collect()
log('detector maps ready')


def match(points, scores, tboxes, oboxes):
    """Greedy pointing-criterion matching by score against target boxes; points inside another annotated finding are ignored."""
    hit = [False] * len(tboxes); rec = []
    for (r, c), s in sorted(zip(points, scores), key=lambda z: -z[1]):
        ins = [k for k, b in enumerate(tboxes) if VD.inside(b, r, c)]
        if ins:
            free = [k for k in ins if not hit[k]]
            if free: hit[free[0]] = True; rec.append((float(s), True))
        elif any(VD.inside(b, r, c) for b in oboxes): continue
        else: rec.append((float(s), False))
    return rec, hit


def box_scores(gtc, pm):
    """(IoU of the two bounding boxes, fraction of predicted pixels inside the annotated box, fraction of the box covered)."""
    if not pm.any(): return 0.0, np.nan, 0.0
    a0, a1, b0, b1 = mask_bbox(gtc); c0, c1, d0, d1 = mask_bbox(pm)
    ih = max(0, min(a1, c1) - max(a0, c0) + 1); iw = max(0, min(b1, d1) - max(b0, d0) + 1); inter = ih * iw
    union = (a1 - a0 + 1) * (b1 - b0 + 1) + (c1 - c0 + 1) * (d1 - d0 + 1) - inter
    both = np.count_nonzero(pm & gtc)
    return inter / union, both / np.count_nonzero(pm), both / np.count_nonzero(gtc)


# ------------------------------------------------------------------ normals: every detection is a false positive
NRM_n, NRM_peaks = [], []
for s, i, st in normals:
    NRM_n.append(sum(not c['fallback'] for c in candidates(P[i]))); NRM_peaks += list(local_peaks(P[i])[1])

# ------------------------------------------------------------------ lesion images
roi = build_model(args.roi) if args.roi else None
if roi is not None: log('delineator weights:', os.path.basename(args.roi))
SRC = ('oracle', 'prior', 'det', 'all')
IMG = {k: [] for k in ('image', 'study', 'set', 'pg', 'pg_any', 'n_det', 'fallback', 'has_mass', 'has_calc')}
LES = {k: [] for k in ['image', 'study', 'set', 'mass', 'calc', 'side', 'frac', 'detected', 'pg_oracle']
       + [f'{m}_{n}' for m in ('biou', 'cont', 'cov') for n in SRC]}
FR_s, FR_t, OP_s, OP_t = [], [], [], []
for n_done, (s, i, st) in enumerate(lesion):
    a = A[i]; H, W = a['H'], a['W']; p = P[i]
    tb = [b for b in a['boxes'] if b['target']]; tboxes = [b['box'] for b in tb]; ob = [b['box'] for b in a['boxes'] if not b['target']]
    cands = candidates(p); real = [c for c in cands if not c['fallback']]
    rec, hit = match([grid_to_full_pt(*c['peak'], H, W) for c in real], [c['score'] for c in real], tboxes, ob)
    OP_s += [r[0] for r in rec]; OP_t += [r[1] for r in rec]
    pts, vals = local_peaks(p)
    rec, _ = match([grid_to_full_pt(y, x, H, W) for y, x in pts], vals, tboxes, ob)
    FR_s += [r[0] for r in rec]; FR_t += [r[1] for r in rec]
    yy, xx = np.unravel_index(int(np.argmax(p)), p.shape); pk = grid_to_full_pt(yy, xx, H, W)
    for k, v in (('image', i), ('study', s), ('set', st), ('pg', int(any(VD.inside(b, *pk) for b in tboxes))),
                 ('pg_any', int(any(VD.inside(b['box'], *pk) for b in a['boxes']))), ('n_det', len(real)),
                 ('fallback', int(cands[0]['fallback'])), ('has_mass', int(any(b['mass'] for b in tb))), ('has_calc', int(any(b['calc'] for b in tb)))):
        IMG[k].append(v)
    rows = [dict(image=i, study=s, set=st, mass=int(b['mass']), calc=int(b['calc']), side=int(max(b['box'][1] - b['box'][0], b['box'][3] - b['box'][2])),
                 frac=float((b['box'][1] - b['box'][0]) * (b['box'][3] - b['box'][2]) / (H * W)), detected=int(hit[k]), pg_oracle=-1,
                 **{f'{m}_{n}': np.nan for m in ('biou', 'cont', 'cov') for n in SRC}) for k, b in enumerate(tb)]
    if roi is not None:
        gray = VD.read_gray(s, i); assert gray.shape == (H, W), (i, gray.shape, H, W)
        wins = [crop_window(H, W, *grid_to_full_bbox(c['bbox'], H, W)) for c in cands[:KMAX]]
        lwin = [crop_window(H, W, y0, y1 - 1, x0, x1 - 1) for (y0, y1, x0, x1) in tboxes]
        probs = predict(roi, np.stack([crop_input(gray, w) for w in wins + lwin]))
        e2e = np.zeros((H, W), bool)
        for k, w in enumerate(wins): paste(e2e, probs[k], w)
        det_full = cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR) >= THR
        for k, (box, win) in enumerate(zip(tboxes, lwin)):
            gtc = crop_mask(VD.box_mask((H, W), [box]), win)
            if not gtc.any(): continue
            po = probs[len(wins) + k]; yy, xx = np.unravel_index(int(np.argmax(po)), po.shape); rows[k]['pg_oracle'] = int(gtc[yy, xx])
            for n, pm in (('oracle', po >= THR), ('prior', PRIOR), ('det', crop_mask(det_full, win)), ('all', crop_mask(e2e, win))):
                rows[k]['biou_' + n], rows[k]['cont_' + n], rows[k]['cov_' + n] = box_scores(gtc, pm)
        del gray, e2e, det_full; gc.collect()
    for r in rows:
        for k in LES: LES[k].append(r[k])
    if (n_done + 1) % 100 == 0 or n_done + 1 == len(lesion): log(f'{n_done + 1}/{len(lesion)} lesion images')

# ------------------------------------------------------------------ save + summary
IMG = {k: np.asarray(v) for k, v in IMG.items()}; LES = {k: np.asarray(v) for k, v in LES.items()}
n_img, n_les = len(lesion), len(LES['image'])
fr_at, fr_cpm, fpi, sens = froc(FR_s, FR_t, n_les, n_img)
s_tp = np.sort(np.asarray(FR_s)[np.asarray(FR_t, bool)])[::-1]; s_fp = np.sort(np.asarray(NRM_peaks))[::-1]; nrm_at = {}
for f in FROC_FPI:                                   # sensitivity on lesion images against false positives per NORMAL image
    k = int(np.floor(f * max(len(normals), 1))); t = s_fp[k] if k < len(s_fp) else -np.inf
    nrm_at[str(f)] = float(np.sum(s_tp > t) / max(n_les, 1))
op_tp = int(np.sum(OP_t)); op_fp = int(len(OP_t) - op_tp)
np.savez_compressed(OUT + f'/vindrdcm_e2e_{args.tag}.npz', **{'img_' + k: v for k, v in IMG.items()}, **{'les_' + k: v for k, v in LES.items()},
                    froc_scores=np.asarray(FR_s), froc_tp=np.asarray(FR_t), op_scores=np.asarray(OP_s), op_tp=np.asarray(OP_t),
                    normal_n_det=np.asarray(NRM_n), normal_peaks=np.asarray(NRM_peaks), normal_image=np.asarray([i for _, i, _ in normals]),
                    n_lesions_unique=n_les, n_images=n_img, n_normals=len(normals))
m = lambda v: float(np.nanmean(v)) if len(v) and np.isfinite(v).any() else None
ms, cs = LES['mass'] == 1, LES['calc'] == 1; dt = LES['detected'] == 1
S = dict(
    n_lesion_images=n_img, n_lesions=n_les, n_masses=int(ms.sum()), n_calcifications=int(cs.sum()), n_normals=len(normals),
    detector=os.path.basename(args.pmap or args.det), delineator=os.path.basename(args.roi) if args.roi else None,
    pointing=dict(all=m(IMG['pg']), mass_images=m(IMG['pg'][IMG['has_mass'] == 1]), calc_only_images=m(IMG['pg'][IMG['has_mass'] == 0]),
                  any_finding=m(IMG['pg_any']), official_test_split=m(IMG['pg'][np.char.endswith(IMG['set'].astype(str), '_test')])),
    operating_point_thr05=dict(sensitivity=op_tp / max(n_les, 1), sensitivity_mass=m(LES['detected'][ms]), sensitivity_calc=m(LES['detected'][cs]),
                               fp_per_image=op_fp / max(n_img, 1), fp_per_normal_image=float(np.mean(NRM_n)) if normals else None,
                               normals_with_any_detection=float(np.mean(np.asarray(NRM_n) > 0)) if normals else None),
    detections_per_image=dict(mean=m(IMG['n_det']), fallback_images=int(IMG['fallback'].sum())),
    froc_local_maxima=dict(sens_at_fp_per_image={str(k): v for k, v in fr_at.items()}, cpm=fr_cpm, sens_at_fp_per_normal_image=nrm_at,
                           end_fp_per_image=float(fpi[-1]) if len(fpi) else 0.0, end_sensitivity=float(sens[-1]) if len(sens) else 0.0),
    box_level=None if roi is None else dict(
        {n: dict(box_iou=m(LES['biou_' + n]), box_iou_gt05=m((LES['biou_' + n] > 0.5).astype(float)), containment=m(LES['cont_' + n]),
                 coverage=m(LES['cov_' + n])) for n in SRC},
        oracle_pointing=m(LES['pg_oracle'][LES['pg_oracle'] >= 0].astype(float)),
        detected_only={n: m(LES['biou_' + n][dt]) for n in SRC}, missed_only={n: m(LES['biou_' + n][~dt]) for n in SRC}),
)
json.dump(S, open(OUT + f'/vindrdcm_e2e_{args.tag}.json', 'w'), indent=2)
log('SUMMARY\n' + json.dumps(S, indent=2))
