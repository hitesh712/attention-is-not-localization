"""End-to-end detect-then-delineate on INbreast (external cohort, CBIS-trained models, no retraining).

  1. reproduce the published conditional result (ROI delineator on ground-truth-centred crops: Dice 0.797)
  2. end-to-end: CBIS detector on the whole INbreast mammogram -> its detections -> same delineator
     scored on the same 116 masses, same windows, same resolution
  3. detector alone on the 67 BI-RADS 1 (normal) images -> false positives per normal mammogram
"""
import os, sys, json, time, gc, glob, re, plistlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2e_common import *                       # sets DML_VISIBLE_DEVICES before importing TF
import pandas as pd, pydicom

import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--roi', default=W_ROI, help='delineator weights (default: the published ROI model)')
ap.add_argument('--det', default=W_DET, help='detector weights (default: the published v3 model)')
ap.add_argument('--pmap', default='', help='precomputed detector probability maps: .npz keyed by INbreast file id')
ap.add_argument('--tag', default='', help='suffix for the output files')
args = ap.parse_args(); SUF = f'_{args.tag}' if args.tag else ''
INB = 'D:/INbreast Release 1.0'; XMLDIR = INB + '/AllXML'; DCMDIR = INB + '/AllDICOMs'
OUT = MODELS + '/e2e'; os.makedirs(OUT, exist_ok=True)
KMAX = 10
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


# ------------------------------------------------------------------ loaders (verbatim from the INbreast notebook)
dcms = {}
for f in glob.glob(DCMDIR + '/*.dcm'):
    m = re.match(r'(\d+)_', os.path.basename(f))
    if m: dcms[m.group(1)] = f


def read_dcm(path):
    a = pydicom.dcmread(path).pixel_array.astype(np.float32)
    return ((a - a.min()) / (a.max() - a.min() + 1e-6) * 255).astype(np.uint8)


def parse_masses(xml_path):
    with open(xml_path, 'rb') as fh: pl = plistlib.load(fh, fmt=plistlib.FMT_XML)
    out = []
    for r in pl['Images'][0]['ROIs']:
        if str(r.get('Name', '')).strip().lower() == 'mass':
            pts = [tuple(map(float, re.findall(r'[-\d.]+', s)[:2])) for s in r.get('Point_px', [])]
            pts = [p for p in pts if len(p) == 2]
            if len(pts) >= 3: out.append(np.array(pts, np.int32))
    return out


cases = []
for xp in sorted(glob.glob(XMLDIR + '/*.xml')):
    iid = os.path.splitext(os.path.basename(xp))[0]
    if iid not in dcms: continue
    try: masses = parse_masses(xp)
    except Exception: continue
    if masses: cases.append((iid, masses))
meta = pd.read_csv(INB + '/INbreast.csv', sep=None, engine='python')
normals = [str(f) for f in meta.loc[meta['Bi-Rads'].astype(str).str.strip() == '1', 'File Name'] if str(f) in dcms]
log(f'mass images {len(cases)} | BI-RADS 1 normal images {len(normals)}')

# ------------------------------------------------------------------ pass 1: detector on whole mammograms
P = {}
if args.pmap:                                    # another detector's maps (e.g. nnU-Net), same downstream code
    Zp = np.load(args.pmap)
    for iid in [c[0] for c in cases] + normals: P[iid] = (np.asarray(Zp[iid], np.float32), None)
    log('detector probability maps loaded from', os.path.basename(args.pmap))
else:
    det = build_model(args.det)
    log('detector weights:', os.path.basename(args.det))
    for iid in [c[0] for c in cases] + normals:
        g = read_dcm(dcms[iid]); P[iid] = (predict(det, full_input(g)[None])[0], g.shape); del g
    del det
gc.collect()
log('pass 1 (detector) done')

# normals: every detection is a false positive
NRM = dict(n_det=[], peaks=[], sweep_fp={t: 0 for t in SWEEP})
for iid in normals:
    p, _ = P[iid]
    NRM['n_det'].append(sum(not c['fallback'] for c in candidates(p)))
    NRM['peaks'] += list(local_peaks(p)[1])
    for t in SWEEP: NRM['sweep_fp'][t] += cv2.connectedComponents((p >= t).astype(np.uint8), connectivity=8)[0] - 1

# ------------------------------------------------------------------ pass 2: delineator, oracle vs end-to-end
roi = build_model(args.roi)
log('delineator weights:', os.path.basename(args.roi))
NRM['n_ver'] = []                                       # normals: detections the delineator confirms (all are FPs)
for iid in normals:
    p, _ = P[iid]; real = [c for c in candidates(p) if not c['fallback']][:KMAX]
    if not real: NRM['n_ver'].append(0); continue
    g = read_dcm(dcms[iid]); H, W = g.shape
    pr = predict(roi, np.stack([crop_input(g, crop_window(H, W, *grid_to_full_bbox(c['bbox'], H, W))) for c in real]))
    NRM['n_ver'].append(int(np.sum(pr.max(axis=(1, 2)) >= THR))); del g
IMG = {k: [] for k in ('iid', 'n_det', 'n_used', 'n_ver', 'fallback', 'dF_det', 'iF_det', 'pgF_det',
                       'dF_top1', 'iF_top1', 'pgF_top1', 'dF_all', 'iF_all', 'pgF_all', 'dF_ver', 'iF_ver', 'pgF_ver')}
LES = {k: [] for k in ('iid', 'area', 'side', 'frac', 'd_oracle', 'i_oracle', 'pg_oracle',
                       'd_oracle_canvas', 'i_oracle_canvas', 'd_det', 'i_det', 'd_top1', 'i_top1',
                       'd_all', 'i_all', 'd_ver', 'i_ver', 'detected', 'detected_ver')}
FR_s, FR_t, OP_s, OP_t, OV_s, OV_t = [], [], [], [], [], []
SW = {t: [0, 0] for t in SWEEP}
n_les_unique = 0

for iid, masses in cases:
    gray = read_dcm(dcms[iid]); H, W = gray.shape
    p, shp = P[iid]; assert shp is None or shp == (H, W)
    full = np.zeros((H, W), np.uint8)
    for poly in masses: cv2.fillPoly(full, [poly], 1)
    gt = full.astype(bool)
    pmasks = []
    for poly in masses:
        m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [poly], 1); pmasks.append(m.astype(bool))

    det_full = cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR) >= THR
    yy, xx = np.unravel_index(int(np.argmax(p)), p.shape); det_pk = grid_to_full_pt(yy, xx, H, W)
    cands = candidates(p); used = cands[:KMAX]
    wins = [crop_window(H, W, *grid_to_full_bbox(c['bbox'], H, W)) for c in used]
    # published crop: polygon bbox -> crop_around geometry; GT = union of all masses inside the window
    lwin = [crop_window(H, W, poly[:, 1].min(), poly[:, 1].max(), poly[:, 0].min(), poly[:, 0].max()) for poly in masses]
    probs = predict(roi, np.stack([crop_input(gray, w) for w in wins + lwin]))

    e2e_all = np.zeros((H, W), bool); e2e_top1 = np.zeros((H, W), bool); e2e_ver = np.zeros((H, W), bool); pk = []
    ver = [float(probs[k].max()) >= THR for k in range(len(wins))]
    for k, w in enumerate(wins):
        v, loc = paste(e2e_all, probs[k], w); pk.append((v, loc))
        if k == 0: paste(e2e_top1, probs[k], w)
        if ver[k]: paste(e2e_ver, probs[k], w)
    all_pk = max(pk, key=lambda z: z[0])[1]; top1_pk = pk[0][1]
    ver_pk = max((pk[k] for k in range(len(wins)) if ver[k]), key=lambda z: z[0])[1] if any(ver) else None

    dF = {n: dice_iou(gt, c) for n, c in (('det', det_full), ('top1', e2e_top1), ('all', e2e_all), ('ver', e2e_ver))}
    for k, v in (('iid', iid), ('n_det', sum(not c['fallback'] for c in cands)), ('n_used', len(used)),
                 ('n_ver', int(sum(ver))), ('fallback', int(cands[0]['fallback'])),
                 ('dF_det', dF['det'][0]), ('iF_det', dF['det'][1]), ('pgF_det', int(gt[det_pk])),
                 ('dF_top1', dF['top1'][0]), ('iF_top1', dF['top1'][1]), ('pgF_top1', int(gt[top1_pk])),
                 ('dF_all', dF['all'][0]), ('iF_all', dF['all'][1]), ('pgF_all', int(gt[all_pk])),
                 ('dF_ver', dF['ver'][0]), ('iF_ver', dF['ver'][1]), ('pgF_ver', int(gt[ver_pk]) if ver_pk else 0)):
        IMG[k].append(v)

    for q, (poly, win) in enumerate(zip(masses, lwin)):
        gtc = crop_mask(gt, win)
        if not gtc.any(): continue                      # same skip rule as the notebook
        po = probs[len(wins) + q]
        yy, xx = np.unravel_index(int(np.argmax(po)), po.shape)
        m = pmasks[q]; r0, r1, c0, c1 = mask_bbox(m) if m.any() else (0, 0, 0, 0)
        row = dict(iid=iid, area=int(m.sum()), side=int(max(r1 - r0, c1 - c0) + 1), frac=float(m.mean()),
                   pg_oracle=int(gtc[yy, xx]),
                   detected=int(m.any() and any(m[grid_to_full_pt(*c['peak'], H, W)] for c in cands if not c['fallback'])),
                   detected_ver=int(m.any() and any(m[grid_to_full_pt(*c['peak'], H, W)]
                                                    for c, v in zip(used, ver) if v and not c['fallback'])))
        row['d_oracle'], row['i_oracle'] = dice_iou(gtc, po >= THR)
        row['d_oracle_canvas'], row['i_oracle_canvas'] = dice_iou(gtc, window_binary(po, win))
        for n, c in (('det', det_full), ('top1', e2e_top1), ('all', e2e_all), ('ver', e2e_ver)):
            row['d_' + n], row['i_' + n] = dice_iou(gtc, crop_mask(c, win))
        for k in LES: LES[k].append(row[k])

    les_masks = [m for m in pmasks if m.any()]; n_les_unique += len(les_masks)
    pts, vals = local_peaks(p)
    rec, _ = match_points([grid_to_full_pt(y, x, H, W) for y, x in pts], vals, les_masks)
    FR_s += [r[0] for r in rec]; FR_t += [r[1] for r in rec]
    real = [c for c in cands if not c['fallback']]
    rec, _ = match_points([grid_to_full_pt(*c['peak'], H, W) for c in real], [c['score'] for c in real], les_masks)
    OP_s += [r[0] for r in rec]; OP_t += [r[1] for r in rec]
    rv = [c for c, v in zip(used, ver) if v and not c['fallback']]
    rec, _ = match_points([grid_to_full_pt(*c['peak'], H, W) for c in rv], [c['score'] for c in rv], les_masks)
    OV_s += [r[0] for r in rec]; OV_t += [r[1] for r in rec]
    for t, (a, b) in sweep_counts(p, H, W, les_masks).items(): SW[t][0] += a; SW[t][1] += b
    del gray, full, gt, pmasks, det_full, e2e_all, e2e_top1, e2e_ver; gc.collect()

log('pass 2 (delineator) done')

# ------------------------------------------------------------------ save + summary
IMG = {k: np.asarray(v) for k, v in IMG.items()}; LES = {k: np.asarray(v) for k, v in LES.items()}
n_img = len(cases)
fr_at, fr_cpm, _, _ = froc(FR_s, FR_t, n_les_unique, n_img)
# sensitivity on mass images against false positives per NORMAL image
s_tp = np.sort(np.asarray(FR_s)[np.asarray(FR_t, bool)])[::-1]; s_fp = np.sort(np.asarray(NRM['peaks']))[::-1]
nrm_at = {}
for f in FROC_FPI:
    k = int(np.floor(f * max(len(normals), 1)))          # allow k false peaks on normals
    t = s_fp[k] if k < len(s_fp) else -np.inf            # threshold just above the (k+1)-th normal peak
    nrm_at[str(f)] = float(np.sum(s_tp > t) / max(n_les_unique, 1))
op_tp = int(np.sum(OP_t)); op_fp = int(len(OP_t) - op_tp)
ov_tp = int(np.sum(OV_t)); ov_fp = int(len(OV_t) - ov_tp)
np.savez_compressed(OUT + f'/inbreast_e2e{SUF}.npz', **{'img_' + k: v for k, v in IMG.items()},
                    **{'les_' + k: v for k, v in LES.items()}, froc_scores=np.asarray(FR_s), froc_tp=np.asarray(FR_t),
                    op_scores=np.asarray(OP_s), op_tp=np.asarray(OP_t), ov_scores=np.asarray(OV_s), ov_tp=np.asarray(OV_t),
                    normal_n_det=np.asarray(NRM['n_det']), normal_n_ver=np.asarray(NRM['n_ver']),
                    normal_peaks=np.asarray(NRM['peaks']), n_lesions_unique=n_les_unique, n_images=n_img,
                    n_normals=len(normals), sweep_t=np.array(SWEEP), sweep_tp=np.array([SW[t][0] for t in SWEEP]),
                    sweep_fp=np.array([SW[t][1] for t in SWEEP]), sweep_fp_normals=np.array([NRM['sweep_fp'][t] for t in SWEEP]))
_, _, _, sw_at, sw_cpm = sweep_curve({t: tuple(SW[t]) for t in SWEEP}, max(n_les_unique, 1), max(n_img, 1))
_, _, _, swn_at, swn_cpm = sweep_curve({t: (SW[t][0], NRM['sweep_fp'][t]) for t in SWEEP}, max(n_les_unique, 1), max(len(normals), 1))
S = dict(
    n_mass_images=n_img, n_lesions=int(len(LES['iid'])), n_lesions_unique=n_les_unique, n_normals=len(normals),
    reproduce=dict(oracle_dice=float(LES['d_oracle'].mean()), oracle_iou=float(LES['i_oracle'].mean()),
                   oracle_pg=float(LES['pg_oracle'].mean()), published='Dice 0.797 +/- 0.110, IoU 0.676, PG 100%, n=116'),
    delineator=os.path.basename(args.roi), detector=os.path.basename(args.pmap or args.det),
    per_image_fullres={n: dict(dice=float(IMG['dF_' + n].mean()), iou=float(IMG['iF_' + n].mean()),
                               pg=float(IMG['pgF_' + n].mean())) for n in ('det', 'top1', 'all', 'ver')},
    per_lesion_window={n: dict(dice=float(LES['d_' + n].mean()), dice_sd=float(LES['d_' + n].std()),
                               median=float(np.median(LES['d_' + n])), iou=float(LES['i_' + n].mean()),
                               dice_gt05=float(np.mean(LES['d_' + n] > 0.5)), dice_gt07=float(np.mean(LES['d_' + n] > 0.7)))
                       for n in ('oracle', 'oracle_canvas', 'det', 'top1', 'all', 'ver')},
    lesion_detected_by_a_detection=float(LES['detected'].mean()),
    lesion_detected_after_verification=float(LES['detected_ver'].mean()),
    operating_point_verified=dict(sensitivity=ov_tp / max(n_les_unique, 1), fp_per_mass_image=ov_fp / max(n_img, 1),
                                  fp_per_normal_image=float(np.mean(NRM['n_ver'])) if normals else None),
    detections_per_image=dict(mean=float(IMG['n_det'].mean()), fallback_images=int(IMG['fallback'].sum())),
    operating_point_thr05=dict(sensitivity=op_tp / max(n_les_unique, 1), fp_per_mass_image=op_fp / max(n_img, 1),
                               fp_per_normal_image=float(np.mean(NRM['n_det'])) if normals else None,
                               normals_with_any_detection=float(np.mean(np.asarray(NRM['n_det']) > 0)) if normals else None),
    froc_local_maxima=dict(sens_at_fp_per_mass_image={str(k): v for k, v in fr_at.items()}, cpm=fr_cpm,
                           sens_at_fp_per_normal_image=nrm_at),
    froc_threshold_sweep=dict(sens_at_fp_per_mass_image={str(k): v for k, v in sw_at.items()}, cpm=sw_cpm,
                              sens_at_fp_per_normal_image={str(k): v for k, v in swn_at.items()}, cpm_normals=swn_cpm),
)
json.dump(S, open(OUT + f'/inbreast_e2e{SUF}.json', 'w'), indent=2)
log('SUMMARY\n' + json.dumps(S, indent=2))
