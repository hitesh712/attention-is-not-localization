"""End-to-end detect-then-delineate evaluation on the CBIS-DDSM patient-disjoint test split.

The published ROI result (Dice 0.871) used crops built from the ground-truth lesion box. Here the
crop is built from the full-image detector's own prediction; everything else is unchanged, and both
are scored on the same lesions, in the same window, at the same resolution.

Steps
  A. detector (v3) on the cached 256x256 test inputs     -> reproduce published Dice 0.316 / PG 54.4%
  B. delineator on the cached oracle crops               -> reproduce published Dice 0.871 / PG 98.8%
  C. per image: detector components -> crops -> delineator -> full-resolution mask
     scored per image (full res + 256 grid) and per lesion (published oracle window), plus FROC.
"""
import os, sys, json, time, gc, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2e_common import *                       # sets DML_VISIBLE_DEVICES before importing TF
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--limit', type=int, default=0, help='only the first N test images in step C (dry run)')
ap.add_argument('--roi', default=W_ROI, help='delineator weights (default: the published ROI model)')
ap.add_argument('--det', default=W_DET, help='detector weights (default: the published v3 model)')
ap.add_argument('--pmap', default='', help='precomputed detector probability maps (N,256,256) .npy, e.g. nnU-Net')
ap.add_argument('--tag', default='', help='suffix for the output files')
args = ap.parse_args()
SUF = ('_dryrun' if args.limit else '') + (f'_{args.tag}' if args.tag else '')

DATA_ROOT = 'D:/cbis-ddsm'; JPEG_ROOT = DATA_ROOT + '/jpeg'; CSV_DIR = DATA_ROOT + '/csv'
DICOM_INFO = CSV_DIR + '/dicom_info.csv'; CACHE = DATA_ROOT + '/_seg_cache'
OUT = MODELS + '/e2e'; os.makedirs(OUT, exist_ok=True)
TEST_CSVS = ['mass_case_description_test_set.csv', 'calc_case_description_test_set.csv']
KMAX = 10                                      # max detections delineated per image
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


# ------------------------------------------------------------------ path resolution (verbatim from the notebooks)
def _to_local(p):
    p = str(p).replace('\\', '/'); i = p.find('jpeg/')
    return os.path.join(JPEG_ROOT, p[i + 5:]) if i >= 0 else p


def build_uid_maps(dinfo):
    di = pd.read_csv(dinfo); di['SeriesDescription'] = di['SeriesDescription'].fillna('')
    di['local'] = di['image_path'].apply(_to_local); di['uid'] = di['SeriesInstanceUID'].astype(str)
    full, mask, allf = {}, {}, {}
    for _, r in di.iterrows():
        allf.setdefault(r['uid'], []).append(r['local']); d = r['SeriesDescription'].lower()
        if 'full' in d: full.setdefault(r['uid'], r['local'])
        elif 'roi mask' in d: mask.setdefault(r['uid'], r['local'])
    return full, mask, allf


FULL_MAP, MASK_MAP, ANY_MAP = build_uid_maps(DICOM_INFO)


def series_uid(p):
    q = str(p).replace('\\', '/').strip().split('/'); return q[-2] if len(q) >= 2 else None


def resolve_mask(p):
    u = series_uid(p)
    if u in MASK_MAP: return MASK_MAP[u]
    for c in ANY_MAP.get(u, []):
        if is_mask_like(cv2.imread(c, cv2.IMREAD_GRAYSCALE)): return c
    return None


def load_tables(names):
    df = pd.concat([pd.read_csv(os.path.join(CSV_DIR, n)) for n in names
                    if os.path.exists(os.path.join(CSV_DIR, n))], ignore_index=True)
    df.columns = [c.strip().replace(' ', '_').lower() for c in df.columns]; return df


raw = load_tables(TEST_CSVS)
raw['full_path'] = raw['image_file_path'].apply(lambda p: FULL_MAP.get(series_uid(p)))
raw['mask_path'] = raw['roi_mask_file_path'].apply(resolve_mask)
raw['patient_id'] = raw['patient_id'].astype(str); raw['image_key'] = raw['image_file_path'].astype(str)

# image list == make_items() of cbis_attention_unet_v3.ipynb (grouped by image, masks unioned)
d = raw.dropna(subset=['full_path']); d = d[d['mask_path'].notna()]
items_img, pats_img = [], []
for key, rows in d.groupby('image_key'):
    items_img.append((rows.iloc[0]['full_path'], list(rows['mask_path']))); pats_img.append(rows.iloc[0]['patient_id'])
# lesion list == prep() of cbis_roi_seg.ipynb (one row per ROI)
les = raw.dropna(subset=['full_path', 'mask_path']).reset_index(drop=True)
for c in ('abnormality_type', 'pathology', 'subtlety', 'breast_density'):
    if c not in les.columns: les[c] = np.nan

Xte = np.load(CACHE + '/te_X_256.npy', mmap_mode='r'); Yte = np.load(CACHE + '/te_Y_256.npy', mmap_mode='r')
Xro = np.load(CACHE + '/roite_X_256.npy', mmap_mode='r'); Yro = np.load(CACHE + '/roite_Y_256.npy', mmap_mode='r')
vro = np.load(CACHE + '/roite_valid_256.npy')
assert len(items_img) == len(Xte), (len(items_img), len(Xte))
assert len(les) == len(Xro), (len(les), len(Xro))
log(f'test images {len(items_img)} | test lesions {len(les)} (valid {int(vro.sum())})')


def score256(y, p):
    """== per_image() of the training notebooks."""
    gt = np.asarray(y)[..., 0].astype(bool); dc, io = dice_iou(gt, p >= THR)
    yy, xx = np.unravel_index(int(np.argmax(p)), p.shape)
    return dc, io, int(gt[yy, xx])


def summ(a):
    a = np.asarray(a, float); return f'{a.mean():.3f} +/- {a.std():.3f} (median {np.median(a):.3f})'


# ------------------------------------------------------------------ A. detector (v3) -> reproduce
if args.pmap:                                   # another detector's maps (e.g. nnU-Net), same downstream code
    P_PATH = args.pmap; Psrc = np.load(P_PATH, mmap_mode='r'); assert Psrc.shape == (len(Xte), SZ, SZ), Psrc.shape
    A = np.array([score256(Yte[i], np.asarray(Psrc[i], np.float32)) for i in range(len(Xte))])
    log('detector probability maps loaded from', os.path.basename(P_PATH))
else:
    det = build_model(args.det)
    log('detector weights:', os.path.basename(args.det))
    P_PATH = CACHE + '/e2e_te_P_256.npy'
    Pmm = np.lib.format.open_memmap(P_PATH, mode='w+', dtype=np.float32, shape=(len(Xte), SZ, SZ))
    A = []
    for b in range(0, len(Xte), 8):
        pr = predict(det, Xte[b:b + 8]); Pmm[b:b + 8] = pr
        A += [score256(Yte[b + k], pr[k]) for k in range(len(pr))]
    Pmm.flush(); A = np.array(A); del det
log(f'A. detector v3 (256 grid)   Dice {summ(A[:, 0])} | IoU {A[:, 1].mean():.3f} | '
    f'PG {A[:, 2].mean() * 100:.1f}% | Dice>0.5 {np.mean(A[:, 0] > 0.5) * 100:.1f}%  (n={len(A)})   '
    f'[published: 0.316 +/- 0.329, IoU 0.239, PG 54.4%, Dice>0.5 36.3%]')
gc.collect()

# ------------------------------------------------------------------ B. delineator on oracle crops -> reproduce
roi = build_model(args.roi)
log('delineator weights:', os.path.basename(args.roi))
sel = np.where(vro)[0]; B = {}
for b in range(0, len(sel), 8):
    idx = sel[b:b + 8]; pr = predict(roi, Xro[idx])
    for k, j in enumerate(idx): B[j] = score256(Yro[j], pr[k])
Bv = np.array([B[j] for j in sel])
log(f'B. delineator, ORACLE crops Dice {summ(Bv[:, 0])} | IoU {Bv[:, 1].mean():.3f} | '
    f'PG {Bv[:, 2].mean() * 100:.1f}% | Dice>0.5 {np.mean(Bv[:, 0] > 0.5) * 100:.1f}% | '
    f'Dice>0.7 {np.mean(Bv[:, 0] > 0.7) * 100:.1f}%  (n={len(Bv)})   [published: 0.871 +/- 0.080, IoU 0.779, PG 98.8%]')

# ------------------------------------------------------------------ C. end-to-end
les_by_fp = {}
for j in sel: les_by_fp.setdefault(les.at[j, 'full_path'], []).append(j)
Pmm = np.load(P_PATH, mmap_mode='r')
n_img = len(items_img) if not args.limit else min(args.limit, len(items_img))

IMG = {k: [] for k in ('i', 'patient', 'H', 'W', 'n_det', 'n_used', 'n_ver', 'fallback', 'chk_x',
                       'd256_det', 'i256_det', 'pg256_det', 'd256_top1', 'd256_all', 'd256_ver',
                       'dF_det', 'iF_det', 'pgF_det', 'dF_top1', 'iF_top1', 'pgF_top1', 'dF_all', 'iF_all', 'pgF_all',
                       'dF_ver', 'iF_ver', 'pgF_ver', 'gt_frac', 'pred_frac_all')}
LES = {k: [] for k in ('j', 'i', 'patient', 'abn', 'pathology', 'subtlety', 'density', 'area', 'side', 'frac',
                       'd_oracle', 'i_oracle', 'pg_oracle', 'd_oracle_canvas', 'i_oracle_canvas',
                       'd_det', 'i_det', 'd_top1', 'i_top1', 'd_all', 'i_all', 'd_ver', 'i_ver',
                       'detected', 'detected_ver', 'pg_all_in_win', 'chk_x', 'chk_y')}
FR_s, FR_t, OP_s, OP_t = [], [], [], []      # FROC (local maxima) and operating point (components >= THR)
OV_s, OV_t = [], []                          # operating point after delineator verification
SW = {t: [0, 0] for t in SWEEP}              # threshold-swept component FROC (tp, fp)
n_les_unique = 0

for i in range(n_img):
    fp, mps = items_img[i]
    gray = cv2.imread(fp, cv2.IMREAD_GRAYSCALE); H, W = gray.shape
    chk_x = int(np.abs(full_input(gray).astype(np.int16) - np.asarray(Xte[i]).astype(np.int16)).max())

    gt = np.zeros((H, W), bool); seen = {}
    for mp in mps:                                              # same per-mask rules as load_mask(), native size
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) if mp else None
        if not is_mask_like(m): continue
        if m.shape != (H, W): m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        m = m >= MASK_THRESH; gt |= m
        if m.any() and mp not in seen: seen[mp] = m              # unique lesions for detection analysis
    les_masks = list(seen.values()); n_les_unique += len(les_masks)

    p = np.asarray(Pmm[i], np.float32)
    det_full = cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR) >= THR
    yy, xx = np.unravel_index(int(np.argmax(p)), p.shape); det_pk = grid_to_full_pt(yy, xx, H, W)

    # detections -> crops (identical geometry to the oracle) -> delineator, plus the oracle crops of this image
    cands = candidates(p); used = cands[:KMAX]
    wins = [crop_window(H, W, *grid_to_full_bbox(c['bbox'], H, W)) for c in used]
    js = les_by_fp.get(fp, [])
    lwin, lmask = [], []
    for j in js:
        m = cv2.imread(les.at[j, 'mask_path'], cv2.IMREAD_GRAYSCALE)
        if m.shape != (H, W): m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        m = m >= MASK_THRESH; lmask.append(m); lwin.append(crop_window(H, W, *mask_bbox(m)))
    Xc = np.stack([crop_input(gray, w) for w in wins + lwin])
    probs = predict(roi, Xc)

    e2e_all = np.zeros((H, W), bool); e2e_top1 = np.zeros((H, W), bool); e2e_ver = np.zeros((H, W), bool); pk = []
    ver = [float(probs[k].max()) >= THR for k in range(len(wins))]     # delineator confirms a lesion in the crop
    for k, w in enumerate(wins):
        v, loc = paste(e2e_all, probs[k], w); pk.append((v, loc))
        if k == 0: paste(e2e_top1, probs[k], w)
        if ver[k]: paste(e2e_ver, probs[k], w)
    all_pk = max(pk, key=lambda z: z[0])[1]; top1_pk = pk[0][1]
    ver_pk = max((pk[k] for k in range(len(wins)) if ver[k]), key=lambda z: z[0])[1] if any(ver) else None

    # ---- per image
    gt256 = np.asarray(Yte[i])[..., 0].astype(bool)
    to256 = lambda m: cv2.resize(m.astype(np.uint8) * 255, (SZ, SZ), interpolation=cv2.INTER_NEAREST) >= 127
    dF = {n: dice_iou(gt, c) for n, c in (('det', det_full), ('top1', e2e_top1), ('all', e2e_all), ('ver', e2e_ver))}
    for k, v in (('i', i), ('patient', pats_img[i]), ('H', H), ('W', W), ('n_det', sum(not c['fallback'] for c in cands)),
                 ('n_used', len(used)), ('n_ver', int(sum(ver))), ('fallback', int(cands[0]['fallback'])), ('chk_x', chk_x),
                 ('d256_det', A[i, 0]), ('i256_det', A[i, 1]), ('pg256_det', A[i, 2]),
                 ('d256_top1', dice_iou(gt256, to256(e2e_top1))[0]), ('d256_all', dice_iou(gt256, to256(e2e_all))[0]),
                 ('d256_ver', dice_iou(gt256, to256(e2e_ver))[0]),
                 ('dF_det', dF['det'][0]), ('iF_det', dF['det'][1]), ('pgF_det', int(gt[det_pk])),
                 ('dF_top1', dF['top1'][0]), ('iF_top1', dF['top1'][1]), ('pgF_top1', int(gt[top1_pk])),
                 ('dF_all', dF['all'][0]), ('iF_all', dF['all'][1]), ('pgF_all', int(gt[all_pk])),
                 ('dF_ver', dF['ver'][0]), ('iF_ver', dF['ver'][1]), ('pgF_ver', int(gt[ver_pk]) if ver_pk else 0),
                 ('gt_frac', gt.mean()), ('pred_frac_all', e2e_all.mean())):
        IMG[k].append(v)

    # ---- per lesion, scored in the published oracle window
    for q, j in enumerate(js):
        win, m = lwin[q], lmask[q]
        gtc = crop_mask(m, win)
        po = probs[len(wins) + q]
        r0, r1, c0, c1 = mask_bbox(m)
        row = dict(j=j, i=i, patient=les.at[j, 'patient_id'], abn=les.at[j, 'abnormality_type'],
                   pathology=les.at[j, 'pathology'], subtlety=les.at[j, 'subtlety'], density=les.at[j, 'breast_density'],
                   area=int(m.sum()), side=int(max(r1 - r0, c1 - c0) + 1), frac=float(m.mean()),
                   d_oracle=B[j][0], i_oracle=B[j][1], pg_oracle=B[j][2],
                   detected=int(any(m[grid_to_full_pt(*c['peak'], H, W)] for c in cands if not c['fallback'])),
                   detected_ver=int(any(m[grid_to_full_pt(*c['peak'], H, W)] for c, v in zip(used, ver) if v and not c['fallback'])),
                   pg_all_in_win=int(m[all_pk]),
                   chk_x=int(np.abs(Xc[len(wins) + q].astype(np.int16) - np.asarray(Xro[j]).astype(np.int16)).max()),
                   chk_y=int(np.count_nonzero(gtc != np.asarray(Yro[j])[..., 0].astype(bool))))
        row['d_oracle_canvas'], row['i_oracle_canvas'] = dice_iou(gtc, window_binary(po, win))
        for n, c in (('det', det_full), ('top1', e2e_top1), ('all', e2e_all), ('ver', e2e_ver)):
            row['d_' + n], row['i_' + n] = dice_iou(gtc, crop_mask(c, win))
        for k in LES: LES[k].append(row[k])

    # ---- detection analysis: FROC over local maxima, and the pipeline's operating point (components >= THR)
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

    del gray, gt, det_full, e2e_all, e2e_top1, e2e_ver, lmask, les_masks, seen; gc.collect()
    if (i + 1) % 25 == 0 or i + 1 == n_img:
        log(f'C. {i + 1}/{n_img}  running per-image Dice: det {np.mean(IMG["dF_det"]):.3f} | '
            f'e2e-all {np.mean(IMG["dF_all"]):.3f}  | per-lesion e2e-all {np.mean(LES["d_all"]) if LES["d_all"] else float("nan"):.3f}')

# ------------------------------------------------------------------ save + summary
IMG = {k: np.asarray(v) for k, v in IMG.items()}; LES = {k: np.asarray(v) for k, v in LES.items()}
fr_at, fr_cpm, _, _ = froc(FR_s, FR_t, n_les_unique, n_img)
op_tp = int(np.sum(OP_t)); op_fp = int(len(OP_t) - op_tp)
ov_tp = int(np.sum(OV_t)); ov_fp = int(len(OV_t) - ov_tp)
np.savez_compressed(OUT + f'/cbis_e2e{SUF}.npz',
                    **{'img_' + k: v for k, v in IMG.items()}, **{'les_' + k: v for k, v in LES.items()},
                    froc_scores=np.asarray(FR_s), froc_tp=np.asarray(FR_t), op_scores=np.asarray(OP_s), op_tp=np.asarray(OP_t),
                    ov_scores=np.asarray(OV_s), ov_tp=np.asarray(OV_t), n_lesions_unique=n_les_unique, n_images=n_img,
                    sweep_t=np.array(SWEEP), sweep_tp=np.array([SW[t][0] for t in SWEEP]), sweep_fp=np.array([SW[t][1] for t in SWEEP]))
_, _, _, sw_at, sw_cpm = sweep_curve({t: tuple(SW[t]) for t in SWEEP}, max(n_les_unique, 1), max(n_img, 1))

S = dict(
    n_images=int(n_img), n_lesions=int(len(LES['j'])), n_lesions_unique=int(n_les_unique),
    integrity=dict(full_input_maxdiff=int(IMG['chk_x'].max()), oracle_crop_maxdiff=int(LES['chk_x'].max()),
                   oracle_gt_mismatch_px=int(LES['chk_y'].max())),
    reproduce=dict(det_dice256=float(A[:, 0].mean()), det_pg256=float(A[:, 2].mean()),
                   oracle_dice=float(Bv[:, 0].mean()), oracle_pg=float(Bv[:, 2].mean())),
    delineator=os.path.basename(args.roi), detector=os.path.basename(args.pmap or args.det),
    detector_dice256=float(A[:, 0].mean()), detector_iou256=float(A[:, 1].mean()), detector_pg256=float(A[:, 2].mean()),
    per_image_fullres={n: dict(dice=float(IMG['dF_' + n].mean()), dice_sd=float(IMG['dF_' + n].std()),
                               iou=float(IMG['iF_' + n].mean()), pg=float(IMG['pgF_' + n].mean()),
                               dice_gt05=float(np.mean(IMG['dF_' + n] > 0.5))) for n in ('det', 'top1', 'all', 'ver')},
    per_image_256={n: float(IMG['d256_' + n].mean()) for n in ('det', 'top1', 'all', 'ver')},
    per_lesion_window={n: dict(dice=float(LES['d_' + n].mean()), dice_sd=float(LES['d_' + n].std()),
                               median=float(np.median(LES['d_' + n])), iou=float(LES['i_' + n].mean()),
                               dice_gt05=float(np.mean(LES['d_' + n] > 0.5)), dice_gt07=float(np.mean(LES['d_' + n] > 0.7)))
                       for n in ('oracle', 'oracle_canvas', 'det', 'top1', 'all', 'ver')},
    lesion_detected_by_a_detection=float(LES['detected'].mean()),
    lesion_detected_after_verification=float(LES['detected_ver'].mean()),
    operating_point_verified=dict(sensitivity=ov_tp / max(n_les_unique, 1), fp_per_image=ov_fp / max(n_img, 1), tp=ov_tp, fp=ov_fp),
    detections_per_image=dict(mean=float(IMG['n_det'].mean()), max=int(IMG['n_det'].max()),
                              fallback_images=int(IMG['fallback'].sum()), capped_images=int(np.sum(IMG['n_det'] > KMAX))),
    operating_point_thr05=dict(sensitivity=op_tp / max(n_les_unique, 1), fp_per_image=op_fp / max(n_img, 1),
                               tp=op_tp, fp=op_fp),
    froc_local_maxima=dict(sens_at_fpi={str(k): v for k, v in fr_at.items()}, cpm=fr_cpm),
    froc_threshold_sweep=dict(sens_at_fpi={str(k): v for k, v in sw_at.items()}, cpm=sw_cpm),
)
json.dump(S, open(OUT + f'/cbis_e2e{SUF}.json', 'w'), indent=2)
log('SUMMARY\n' + json.dumps(S, indent=2))
