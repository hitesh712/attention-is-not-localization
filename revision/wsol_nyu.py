"""Dedicated weakly-supervised localization baselines (Reviewer 1 #3, Reviewer 2 #3): the public NYU models GMIC
(Shen et al., Med. Image Anal. 2021; five models) and GLAM (Liu et al., MIDL 2021; jointly / separately trained),
trained with breast-level benign / malignant labels on NYU screening mammograms, applied WITHOUT retraining to the
399 masked CBIS-DDSM test mammograms and to INbreast (107 mass images + 67 BI-RADS 1 normals).

Preprocessing is the repositories' own pipeline, called directly: breast crop (crop_mammogram), all breasts facing
right, optimal-centre 2944x1920 window (get_optimal_centers / augmentations, no crop noise at test time) and per-image
standardisation. CBIS-DDSM film scans (~42-50 um/pixel) are resampled x0.5 to approach the ~70-100 um pixel size of
the NYU full-field digital training images; INbreast (70 um) is used at native resolution. The breast side is
inferred from the breast mask, and the film background of CBIS is set to 0 as in digital mammograms.

Saliency maps are mapped back to original-image coordinates (window -> crop -> un-flip -> original resolution) and
scored exactly like every other method in the study: per-image pointing (peak inside an annotated lesion) and
IoU / Dice of the >= half-max region. Lesion map = pixel-wise max of the benign and malignant maps.
Usage (Mammo-CLIP venv: torch + CUDA, cv2, pydicom):  python wsol_nyu.py --model gmic   |   --model glam
"""
import os, sys, json, time, argparse, glob, re, plistlib, types
ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True, choices=['gmic', 'glam'])
ap.add_argument('--limit', type=int, default=0, help='first N images per dataset (smoke test)')
ap.add_argument('--device', default='gpu', choices=['gpu', 'cpu'])
ap.add_argument('--viz', default='', help='folder: save overlays (input window + mapped-back map) for checking')
ap.add_argument('--datasets', default='cbis,inbreast', help="comma-separated: cbis, inbreast, vindr (VinDr-Mammo DICOMs; box annotations)")
args = ap.parse_args()
REPO = {'gmic': 'D:/wsol_baselines/GMIC', 'glam': 'D:/wsol_baselines/GLAM'}[args.model]
sys.path.insert(0, REPO); sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
for _m in ('imageio', 'h5py'): sys.modules.setdefault(_m, types.ModuleType(_m))     # used only by unused file readers
import numpy as np, cv2, torch, pandas as pd, pydicom, scipy.ndimage as ndi
from sklearn.metrics import roc_auc_score
if not hasattr(torch, 'has_cudnn'): torch.has_cudnn = torch.backends.cudnn.is_available()
import src.data_loading.augmentations as aug
import src.cropping.crop_mammogram as cropm
import src.optimal_centers.get_optimal_centers as goc
import cbis_data as D
import vindr_data as VD

OUT = 'G:/My Drive/VS_CC/models/revision'; WIN = (2944, 1920)
T0 = time.time(); log = lambda *a: print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)
dev = torch.device('cuda' if args.device == 'gpu' else 'cpu')


# ------------------------------------------------------------------ models
def load_models():
    out = {}
    if args.model == 'gmic':
        import src.modeling.gmic as G
        from src.constants import PERCENT_T_DICT
        base = dict(device_type=args.device, gpu_number=0, max_crop_noise=(100, 100), max_crop_size_noise=100, cam_size=(46, 30),
                    K=6, crop_shape=(256, 256), post_processing_dim=256, num_classes=2, use_v1_global=False)
        for i in range(1, 6):
            m = G.GMIC(dict(base, percent_t=PERCENT_T_DICT[str(i)]))
            mi, un = m.load_state_dict(torch.load(f'{REPO}/models/sample_model_{i}.p', map_location='cpu', weights_only=False), strict=False)
            log(f'GMIC model {i}: {len(mi)} missing, unexpected {un}'); assert len(mi) == 0, mi
            out[f'gmic_{i}'] = m.to(dev).eval()
    else:
        import src.modeling.glam as G
        import src.modeling.common_functions as CF

        def get_max_window(input_image, window_shape, pooling_logic='avg'):
            # the repository's version divides the argmax index with '/', which was integer division in PyTorch 1.1
            # but is true division in PyTorch >= 1.5 and silently misplaces the local patch; floor division restores it
            N, C, H, W = input_image.size()
            pool_map = torch.nn.functional.avg_pool2d(input_image, window_shape, stride=1)
            idx = torch.max(pool_map.view(N, C, -1), -1)[1]; x = torch.div(idx, pool_map.size(3), rounding_mode='floor')
            return torch.cat([x.unsqueeze(-1), (idx - x * pool_map.size(3)).unsqueeze(-1)], dim=-1)
        CF.get_max_window = get_max_window
        base = dict(device_type=args.device, gpu_number=0, max_crop_noise=(100, 100), max_crop_size_noise=100, cam_size=(184, 120),
                    top_k_per_class=1, crop_shape=(512, 512), percent_k=0.03, detection_pooling_percent_k=0.3)
        for name in ('model_joint', 'model_sep'):
            m = G.MILSingleImageModel(dict(base))
            sd = torch.load(f'{REPO}/models/{name}.ckpt', map_location='cpu', weights_only=False)
            sd = sd.get('model', sd.get('state_dict', sd)) if isinstance(sd, dict) and not any(torch.is_tensor(v) for v in sd.values()) else sd
            mi, un = m.load_state_dict(sd, strict=False)
            log(f'GLAM {name}: {len(mi)} missing, unexpected {un}'); assert len(mi) == 0, mi
            out[f'glam_{name.split("_")[1]}'] = m.to(dev).eval()
    return out


@torch.no_grad()
def run(models, win):
    """-> {model: (saliency (2, 2944, 1920) numpy, prediction (2,))}"""
    x = torch.from_numpy(win[None, None]).to(dev); res = {}
    for name, m in models.items():
        y = m(x); s = m.saliency_map
        if s.shape[-2:] != WIN: s = torch.nn.functional.interpolate(s, size=WIN, mode='bilinear', align_corners=False)
        res[name] = (s[0].float().cpu().numpy(), y[0].float().cpu().numpy())
    return res


# ------------------------------------------------------------------ NYU preprocessing + exact inverse mapping
def prepare(gray, view, scale, otsu):
    H0, W0 = gray.shape
    img = cv2.resize(gray, (int(round(W0 * scale)), int(round(H0 * scale))), interpolation=cv2.INTER_AREA) if scale != 1 else gray
    if otsu:                                          # film: Otsu breast mask, background -> 0
        g8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        mask = g8 > cv2.threshold(g8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    else:                                             # digital: background is already the minimum
        mask = img > img.min() + 1e-3 * (img.max() - img.min())
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    mask = ndi.binary_fill_holes(lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))
    if otsu: mask = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0
    flip = bool(np.nonzero(mask)[1].mean() > img.shape[1] / 2)      # breast on the right -> faces left
    if flip: img, mask = img[:, ::-1], mask[:, ::-1]
    img0 = np.where(mask, img, 0).astype(np.float32)
    (top, bottom, left, right), rightmost, bottommost, _ = cropm.crop_img_from_largest_connected(
        mask.astype(np.uint8), 'right', False, 0, 50, 1 / 3)
    crop = img0[top:bottom, left:right]
    full_view = 'L-' + view                                           # canonical orientation: no further flip
    centre = goc.extract_center(dict(full_view=full_view, view=view, horizontal_flip='NO',
                                     rightmost_points=rightmost, bottommost_points=bottommost), crop)
    rr, cc = np.mgrid[:crop.shape[0], :crop.shape[1]]
    auxi = np.stack([rr + 1, cc + 1], -1).astype(np.float32); del rr, cc
    win, auxw = aug.random_augmentation_best_center(image=crop, input_size=WIN, random_number_generator=np.random.RandomState(0),
                                                    best_center=centre, view=full_view, auxiliary_image=auxi)
    win = win.astype(np.float32).copy(); win -= win.mean(); win /= max(float(win.std()), 1e-5)
    r = auxw[..., 0].astype(np.int64) - 1; c = auxw[..., 1].astype(np.int64) - 1; valid = r >= 0
    info = dict(flip=flip, top=top, left=left, ds_shape=img.shape, shape=(H0, W0), r=r[valid] + top, c=c[valid] + left,
                valid=valid, crop_box=(int(top), int(bottom), int(left), int(right)), centre=tuple(map(int, centre)))
    return win, info


def to_original(sal_win, info):
    canvas = np.zeros(info['ds_shape'], np.float32)
    canvas[info['r'], info['c']] = sal_win[info['valid']]
    if info['flip']: canvas = canvas[:, ::-1]
    if canvas.shape != info['shape']: canvas = cv2.resize(np.ascontiguousarray(canvas), info['shape'][::-1], interpolation=cv2.INTER_LINEAR)
    return canvas


def score(m, gt):
    py, px = np.unravel_index(int(np.argmax(m)), m.shape)
    lo, hi = float(m.min()), float(m.max())
    pm = (m - lo) >= 0.5 * (hi - lo) if hi > lo else np.zeros(m.shape, bool)
    inter = np.count_nonzero(pm & gt); s = np.count_nonzero(pm) + np.count_nonzero(gt); u = s - inter
    return int(gt[py, px]), (inter / u if u else 0.0), (2 * inter / s if s else 0.0)


def overlay(gray, L, gt, win, Lw, path, h=900):
    """left: original image + mapped-back lesion map + GT contour (green) + peak (cyan); right: the model's input window."""
    def heat(img, m, mask=None):
        f = h / img.shape[0]; sz = (max(1, int(img.shape[1] * f)), h)
        g = cv2.resize(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), sz)
        mm = cv2.resize(m.astype(np.float32), sz); mm = (255 * (mm - mm.min()) / (mm.max() - mm.min() + 1e-9)).astype(np.uint8)
        o = cv2.addWeighted(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), 0.6, cv2.applyColorMap(mm, cv2.COLORMAP_JET), 0.4, 0)
        if mask is not None:
            cs, _ = cv2.findContours(cv2.resize(mask.astype(np.uint8), sz, interpolation=cv2.INTER_NEAREST), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(o, cs, -1, (0, 255, 0), 2)
        py, px = np.unravel_index(int(np.argmax(m)), m.shape); cv2.circle(o, (int(px * f), int(py * f)), 8, (255, 255, 0), 2)
        return o
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.hstack([heat(gray, L, gt), heat(win, Lw)]))


def evaluate(models, items, scale, otsu, tag):
    """items: (key, loader -> gray float32, view, gt-mask fn or None, extra dict)"""
    names = list(models); R = dict(key=[], pg=[], iou=[], dice=[], pg_benign=[], pg_malignant=[],
                                   pg_each={n: [] for n in names}, pred={n: [] for n in names}, area=[], in_window=[])
    R['failed'] = []; R['pg_any'] = []                     # pg_any: peak inside ANY annotated finding (datasets that supply 'anyfn')
    for i, (key, load, view, gtfn, extra) in enumerate(items):
        gray = load()
        try:
            win, info = prepare(gray, view, scale, otsu); out = run(models, win)
        except Exception as e:                              # scored as a miss (never silently excluded)
            log(f'{tag} {key}: preprocessing/inference failed ({type(e).__name__}: {e}) -> scored as a miss')
            R['failed'].append(key); R['key'].append(key)
            for n in names: R['pred'][n].append(np.full(2, np.nan, np.float32))
            if gtfn is not None:
                gt = gtfn(gray.shape)
                for k_ in ('pg', 'pg_benign', 'pg_malignant'): R[k_].append(0)
                R['iou'].append(0.0); R['dice'].append(0.0); R['area'].append(float(gt.mean())); R['in_window'].append(0.0)
                for n in names: R['pg_each'][n].append(0)
                if 'anyfn' in extra: R['pg_any'].append(0)
            continue
        ens = np.mean([out[n][0] for n in names], 0)                                       # (2, 2944, 1920)
        R['key'].append(key)
        for n in names: R['pred'][n].append(out[n][1])
        if gtfn is not None:
            gt = gtfn(gray.shape)
            L = to_original(ens.max(0), info); p, io, di = score(L, gt)
            R['pg'].append(p); R['iou'].append(io); R['dice'].append(di); R['area'].append(float(gt.mean()))
            R['in_window'].append(float(np.count_nonzero(gt & (to_original(np.ones(WIN, np.float32), info) > 0.5)) / max(gt.sum(), 1)))
            R['pg_benign'].append(score(to_original(ens[0], info), gt)[0]); R['pg_malignant'].append(score(to_original(ens[1], info), gt)[0])
            for n in names: R['pg_each'][n].append(score(to_original(out[n][0].max(0), info), gt)[0])
            if 'anyfn' in extra: R['pg_any'].append(score(L, extra['anyfn'](gray.shape))[0])
            if args.viz: overlay(gray, L, gt, win, ens.max(0), f'{args.viz}/{tag.replace("/", "_")}_{i:03d}.png')
        if (i + 1) % 25 == 0 or i + 1 == len(items):
            msg = f'PG {np.mean(R["pg"]) * 100:.1f}% IoU {np.mean(R["iou"]):.3f}' if R['pg'] else ''
            log(f'{tag} {i + 1}/{len(items)} {msg}')
    return R


# ------------------------------------------------------------------ datasets
def cbis_items():
    rows = D.test_rows(); first = rows.drop_duplicates('image_key').set_index('image_key')
    mal = rows.groupby('image_key')['pathology'].apply(lambda s: int((s.astype(str).str.upper() == 'MALIGNANT').any()))
    items = []
    for key, fp, mps, pid, lb in D.test_images():
        view = 'MLO' if 'MLO' in str(first.at[key, 'image_view']).upper() else 'CC'
        def gtfn(shape, mps=mps):
            g = np.zeros(shape, bool)
            for mp in mps:
                mk = D.read_mask(mp, shape)
                if mk is not None: g |= mk
            return g
        items.append((key, (lambda fp=fp: cv2.imread(fp, cv2.IMREAD_GRAYSCALE).astype(np.float32)), view, gtfn,
                      dict(patient=pid, malignant=int(mal[key]), mass=lb)))
    return items


INB = 'D:/INbreast Release 1.0'
def inbreast_items():
    dcms = {m.group(1): f for f in glob.glob(INB + '/AllDICOMs/*.dcm') for m in [re.match(r'(\d+)_', os.path.basename(f))] if m}
    def load(f):
        ds = pydicom.dcmread(f); a = ds.pixel_array.astype(np.float32)
        if str(getattr(ds, 'PhotometricInterpretation', '')).upper() == 'MONOCHROME1': a = a.max() - a
        return (a - a.min()) / (a.max() - a.min() + 1e-6)
    def view_of(f):
        m = re.search(r'_MG_([LR])_(CC|MLO|ML)_', os.path.basename(f)); return 'CC' if (m and m.group(2) == 'CC') else 'MLO'
    def masses(xp):
        with open(xp, 'rb') as fh: pl = plistlib.load(fh, fmt=plistlib.FMT_XML)
        out = []
        for r in pl['Images'][0]['ROIs']:
            if str(r.get('Name', '')).strip().lower() == 'mass':
                pts = [tuple(map(float, re.findall(r'[-\d.]+', s)[:2])) for s in r.get('Point_px', [])]; pts = [p for p in pts if len(p) == 2]
                if len(pts) >= 3: out.append(np.array(pts, np.int32))
        return out
    meta = pd.read_csv(INB + '/INbreast.csv', sep=None, engine='python')
    birads = {str(f): str(b).strip() for f, b in zip(meta['File Name'], meta['Bi-Rads'])}
    items = []
    for xp in sorted(glob.glob(INB + '/AllXML/*.xml')):
        iid = os.path.splitext(os.path.basename(xp))[0]
        if iid not in dcms: continue
        try: ps = masses(xp)
        except Exception: continue
        if not ps: continue
        def gtfn(shape, ps=ps):
            g = np.zeros(shape, np.uint8); cv2.fillPoly(g, ps, 1); return g.astype(bool)
        items.append((iid, (lambda f=dcms[iid]: load(f)), view_of(dcms[iid]), gtfn, dict(birads=birads.get(iid, ''), normal=0)))
    normals = [str(f) for f in meta.loc[meta['Bi-Rads'].astype(str).str.strip() == '1', 'File Name'] if str(f) in dcms]
    for iid in normals:
        items.append((iid, (lambda f=dcms[iid]: load(f)), view_of(dcms[iid]), None, dict(birads='1', normal=1)))
    return items


def vindr_items():
    """VinDr-Mammo (digital, ~0.085 mm pixels: native resolution, like INbreast). Target = boxes of masses and suspicious
    calcifications, i.e. the same targets as for the supervised detectors; 300 BI-RADS 1 images as normal controls."""
    A = VD.annotations(); items = []
    for r in VD.selection(VD.LESION_SETS):
        a = A[r['image_id']]; tb = [b['box'] for b in a['boxes'] if b['target']]; ab = [b['box'] for b in a['boxes']]
        if not tb: continue
        items.append((r['image_id'], (lambda s=r['study_id'], i=r['image_id']: VD.read_float(s, i)), 'CC' if a['view'] == 'CC' else 'MLO',
                      (lambda shape, bs=tb: VD.box_mask(shape, bs)),
                      dict(normal=0, set=r['set'], study=r['study_id'], anyfn=(lambda shape, bs=ab: VD.box_mask(shape, bs)))))
    for r in VD.selection((VD.NORMAL_SET,)):
        a = A[r['image_id']]
        items.append((r['image_id'], (lambda s=r['study_id'], i=r['image_id']: VD.read_float(s, i)), 'CC' if a['view'] == 'CC' else 'MLO',
                      None, dict(normal=1, set=r['set'], study=r['study_id'])))
    return items


def summarise(R, items):
    s = dict(n_images=len(R['key']), n_scored=len(R['pg']), n_failed=len(R['failed']), failed=R['failed'])
    if R['pg']:
        s.update(pointing=float(np.mean(R['pg'])), iou=float(np.mean(R['iou'])), dice=float(np.mean(R['dice'])),
                 pointing_benign_map=float(np.mean(R['pg_benign'])), pointing_malignant_map=float(np.mean(R['pg_malignant'])),
                 pointing_each={n: float(np.mean(v)) for n, v in R['pg_each'].items()},
                 lesion_fraction_inside_window=float(np.mean(R['in_window'])))
    return s


models = load_models()
res, arrays = {}, {}
DATASETS = {'cbis': (cbis_items, 0.5, True), 'inbreast': (inbreast_items, 1.0, False), 'vindr': (vindr_items, 1.0, False)}
chosen = [d for d in args.datasets.split(',') if d]
for ds in chosen:
    items = DATASETS[ds][0](); scale, otsu = DATASETS[ds][1:]
    if args.limit: items = items[:args.limit] + ([it for it in items if it[3] is None][:args.limit] if ds != 'cbis' else [])
    R = evaluate(models, items, scale, otsu, f'{args.model}/{ds}'); s = summarise(R, items)
    if R['pg_any']: s['pointing_any_finding'] = float(np.mean(R['pg_any'])); arrays[f'{ds}_pg_any'] = np.array(R['pg_any'])
    ex = [it[4] for it in items]; P = {n: np.array(v) for n, v in R['pred'].items()}; Pm = np.mean([P[n] for n in P], 0)
    ok = np.isfinite(Pm).all(1)                              # failed images (if any) are excluded from the AUC only
    if ds == 'cbis':
        y = np.array([e['malignant'] for e in ex])
        if 0 < y[ok].sum() < ok.sum(): s['auc_malignant'] = float(roc_auc_score(y[ok], Pm[ok, 1]))
    else:
        y = np.array([1 - e['normal'] for e in ex])
        if 0 < y[ok].sum() < ok.sum(): s['auc_mass_vs_normal' if ds == 'inbreast' else 'auc_lesion_vs_normal'] = float(roc_auc_score(y[ok], Pm[ok].max(1)))
    res[ds] = s; log(ds, json.dumps(s))
    arrays.update({f'{ds}_keys': np.array(R['key']), f'{ds}_scored': np.array([it[3] is not None for it in items]),
                   f'{ds}_pg': np.array(R['pg']), f'{ds}_iou': np.array(R['iou']),
                   f'{ds}_dice': np.array(R['dice']), f'{ds}_area': np.array(R['area']), f'{ds}_pred_mean': Pm,
                   f'{ds}_pg_benign': np.array(R['pg_benign']), f'{ds}_pg_malignant': np.array(R['pg_malignant']),
                   **{f'{ds}_pg_{n}': np.array(v) for n, v in R['pg_each'].items()}})
suf = ('' if chosen == ['cbis', 'inbreast'] else '_' + '_'.join(chosen)) + ('_smoke' if args.limit else '')
json.dump(res, open(f'{OUT}/wsol_{args.model}{suf}.json', 'w'), indent=2)
np.savez_compressed(f'{OUT}/wsol_{args.model}{suf}.npz', **arrays)
log('done')
