"""Collect every number quoted in the revised manuscript from the result files -> numbers.tex (+ numbers.md).

Each value becomes a LaTeX key read by \\V{key} in main_rev.tex; keys whose source file does not exist yet are
simply not written (the manuscript then prints them in red). Confidence intervals: patient-clustered bootstrap
(5,000 resamples of patients); paired tests: exact McNemar (pointing) and Wilcoxon signed-rank (Dice).
"""
import os, re, glob, json
import numpy as np
from scipy.stats import binomtest, wilcoxon

M = 'G:/My Drive/VS_CC/models'; REV = M + '/revision'; E2E = M + '/e2e'
TEX = 'D:/Research 9.9.26/current research/IEEE_Access/numbers.tex'
B = 5000; RNG = np.random.RandomState(2021)
K = {}                                                     # key -> LaTeX string
RAW = {}                                                   # key -> raw value (for numbers.md / sanity checks)


def put(key, txt, raw=None):
    K[key] = txt; RAW[key] = raw if raw is not None else txt


def load_json(p):
    return json.load(open(p)) if os.path.exists(p) else None


def load_npz(p):
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None


# ------------------------------------------------------------------ formatting
def f3(x): return f'{x:.3f}'
def f2(x): return f'{x:.2f}'
def pc(x, d=1): return f'{100 * x:.{d}f}\\%'
def pcn(x, d=1): return f'{100 * x:.{d}f}'                 # percent number without the sign (table cells)
def pmf(v, d=3): v = np.asarray(v, float); return f'{v.mean():.{d}f} $\\pm$ {v.std():.{d}f}'
def pmp(v, d=1): v = 100 * np.asarray(v, float); return f'{v.mean():.{d}f} $\\pm$ {v.std():.{d}f}'


def fp(p, up=False):
    """p-value: 3 decimals (2 significant digits below 0.01), or a x 10^-k for small values.
    up=True rounds upwards, for values quoted as bounds ("p <= ...")."""
    if p is None or not np.isfinite(p): return '--'
    rnd = (lambda x, d: np.ceil(x * 10 ** d - 1e-9) / 10 ** d) if up else (lambda x, d: round(x, d))
    if p >= 0.01: return f'{rnd(p, 3):.3f}'
    e = int(np.floor(np.log10(p))); m = rnd(p / 10 ** e, 1)
    if m >= 10: m, e = m / 10, e + 1
    if e >= -3: return f'{m * 10 ** e:.{1 - e}f}'
    return f'${m:.1f}\\times10^{{{e}}}$'


def ci_txt(t, pct=True, d=1):
    m, lo, hi = t
    return (f'{100 * m:.{d}f}\\% [{100 * lo:.{d}f}, {100 * hi:.{d}f}]' if pct else f'{m:.3f} [{lo:.3f}, {hi:.3f}]')


# ------------------------------------------------------------------ statistics
def cluster_ci(values, clusters):
    v = np.asarray(values, float); g = np.asarray(clusters).astype(str)
    uniq, inv = np.unique(g, return_inverse=True)
    S = np.bincount(inv, weights=v, minlength=len(uniq)); N = np.bincount(inv, minlength=len(uniq)).astype(float)
    idx = RNG.randint(len(uniq), size=(B, len(uniq)))
    bs = S[idx].sum(1) / N[idx].sum(1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def mcnemar(a, b):
    a = np.asarray(a, bool); b = np.asarray(b, bool); n10 = int(np.sum(a & ~b)); n01 = int(np.sum(~a & b))
    p = float(binomtest(min(n10, n01), n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0
    return n10, n01, p


def wil(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0


# ------------------------------------------------------------------ INbreast patient ids (2nd token of the DICOM name)
INB_PAT = {m.group(1): m.group(2) for f in glob.glob('D:/INbreast Release 1.0/AllDICOMs/*.dcm')
           for m in [re.match(r'(\d+)_([0-9a-f]+)_', os.path.basename(f))] if m}
inb_pat = lambda ids: np.array([INB_PAT.get(str(i), str(i)) for i in ids])

# ================================================================== 1. weakly-supervised EfficientNet-B0 (CBIS) + detector
W = load_npz(REV + '/ws_reeval.npz'); WJ = load_json(REV + '/ws_reeval.json')
EC = load_npz(E2E + '/cbis_e2e_pub.npz'); ECJ = load_json(E2E + '/cbis_e2e_pub.json')
det_hit = None
if EC is not None:
    det_hit = EC['img_pgF_det'].astype(bool); cpat = EC['img_patient']
    put('det.cbis.pg', ci_txt(cluster_ci(det_hit, cpat)), det_hit.mean())
    put('det.cbis.pg.rate', pc(det_hit.mean()), det_hit.mean())
    put('det.cbis.pg.k', f'{int(det_hit.sum())}/{len(det_hit)}')
    put('det.cbis.iouF', f3(EC['img_iF_det'].mean())); put('det.cbis.diceF', f3(EC['img_dF_det'].mean()))
if W is not None and WJ is not None:
    wpat = W['info_patient']
    if det_hit is not None: assert (wpat == cpat).all(), 'image order mismatch ws_reeval vs e2e'
    ra = WJ['row_auc']
    put('ws.auc.frozen', f3(ra['frozen'])); put('ws.auc.finetuned', f3(ra['finetuned'])); put('ws.auc.mil', f3(ra['mil']))
    put('ws.mil.tileauc', f3(ra['mil_rows_pg_tileauc_iou'][2])); put('ws.mil.rowpg', pc(ra['mil_rows_pg_tileauc_iou'][1]))
    put('chance.cbis', pc(W['info_gt_frac'].mean()), W['info_gt_frac'].mean())
    rates = {}
    for m in ['fz_gradcam', 'fz_cbam', 'ft_gradcam', 'ft_gradcampp', 'ft_layercam', 'ft_scorecam', 'ft_cbam', 'mil']:
        h = (W['mil_pg_tileF'] if m == 'mil' else W[f'{m}_pgF']).astype(bool); rates[m] = h.mean()
        put(f'ws.{m}.pg', ci_txt(cluster_ci(h, wpat)), h.mean()); put(f'ws.{m}.pg.rate', pc(h.mean()), h.mean())
        if m == 'mil':
            put('ws.mil.iou', f3(W['mil_iou_tile'].mean()))
        else:
            put(f'ws.{m}.pg224', pc(W[f'{m}_pg224'].mean()))
            for s in ('', '25', '75'):
                put(f'ws.{m}.iou{s}', f3(W[f'{m}_iou{s}'].mean())); put(f'ws.{m}.dice{s}', f3(W[f'{m}_dice{s}'].mean()))
        if det_hit is not None:
            a, b_, p = mcnemar(det_hit, h)
            put(f'ws.{m}.mcn', fp(p), p); put(f'ws.{m}.disc', f'{a}/{b_}')
    best = max(rates, key=rates.get)
    put('ws.max.pg', pc(rates[best]), rates[best]); put('ws.min.pg', pc(min(rates.values())))
    put('ws.max.name', {'ft_layercam': 'LayerCAM', 'mil': 'attention-MIL', 'ft_scorecam': 'Score-CAM'}.get(best, best))
    if det_hit is not None:
        ps = [RAW[f'ws.{m}.mcn'] for m in rates]; put('ws.mcn.max', fp(max(ps), up=True), max(ps))
        d = cluster_ci(det_hit.astype(float) - (W['ft_layercam_pgF'] if best != 'mil' else W['mil_pg_tileF']).astype(float), wpat)
        put('ws.diff.det_minus_best', f'{100 * d[0]:.1f} [{100 * d[1]:.1f}, {100 * d[2]:.1f}]')

# ================================================================== 2. Mini-MIAS
MJ = load_json(REV + '/mias_cv.json')
if MJ:
    for s in ('patient', 'image'):
        r = MJ[s]
        put(f'mias.{s}.auc', f3(r['pooled_auc'])); put(f'mias.{s}.foldauc', f'{r["fold_auc_mean"]:.2f} $\\pm$ {r["fold_auc_sd"]:.2f}')
        for k in ('gradcam_pg', 'cbam_pg'): put(f'mias.{s}.{k}', pc(r[k]))
        for k in ('gradcam_iou', 'cbam_iou'): put(f'mias.{s}.{k}', f3(r[k]))
        put(f'mias.{s}.nloc', str(r['n_loc']))

# ================================================================== 3. Mammo-CLIP heads (+ ImageNet control)
crop_gt = 'D:/cbis-ddsm/_mc_cache/gt_test_crop760.npy'
if os.path.exists(crop_gt):
    g = np.load(crop_gt); put('chance.crop', pc(g.reshape(len(g), -1).mean(1).mean()))
MAPS = (('cam', ('cam',)), ('cbam', ('gradcam', 'cbam_spatial')), ('mil', ('mil_attention',)))
for enc, suf in (('mc', ''), ('in', '_imagenet')):
    R = load_json(f'{REV}/mc_heads{suf}.json'); Z = load_npz(f'{REV}/mc_heads{suf}.npz')
    if not R: continue
    for r in ('r380', 'r760', 'r1520'):
        if f'{r}_pca_explained' in R: put(f'{enc}.{r}.pca', pc(R[f'{r}_pca_explained']))
        for kind, maps in MAPS:
            runs = [R[f'{r}_{kind}_s{s}'] for s in (0, 1, 2) if f'{r}_{kind}_s{s}' in R]
            if not runs: continue
            put(f'{enc}.{r}.{kind}.auc', pmf([x['auc_rows'] for x in runs]), np.mean([x['auc_rows'] for x in runs]))
            put(f'{enc}.{r}.{kind}.aucimg', pmf([x['auc_images'] for x in runs]))
            for mp in maps:
                pg = [x[mp]['pointing'] for x in runs]
                put(f'{enc}.{r}.{kind}.{mp}.pg', pmp(pg), float(np.mean(pg)))
                put(f'{enc}.{r}.{kind}.{mp}.iou', pmf([x[mp]['iou'] for x in runs]))
                put(f'{enc}.{r}.{kind}.{mp}.dice', pmf([x[mp]['dice'] for x in runs]))
                if Z is not None and det_hit is not None and f'{r}_{kind}_{mp}_s0' in Z.files and W is not None:
                    assert (Z['patients'] == wpat).all()
                    pv = [mcnemar(det_hit, Z[f'{r}_{kind}_{mp}_s{s}'][:, 0].astype(bool)) for s in (0, 1, 2) if f'{r}_{kind}_{mp}_s{s}' in Z.files]
                    put(f'{enc}.{r}.{kind}.{mp}.mcn', fp(max(x[2] for x in pv), up=True), max(x[2] for x in pv))   # bound over seeds
                    put(f'{enc}.{r}.{kind}.{mp}.disc', ', '.join(f'{x[0]}/{x[1]}' for x in pv))
    if Z is not None and 'r1520_mil_mil_attention_s0' in Z.files and det_hit is not None:
        h = np.mean([Z[f'r1520_mil_mil_attention_s{s}'][:, 0] for s in (0, 1, 2)], 0)
        put(f'{enc}.r1520.mil.pg.ci', ci_txt(cluster_ci(h, wpat)))

# ================================================================== 4. GMIC / GLAM
EI = load_npz(E2E + '/inbreast_e2e_pub.npz')
det_inb = None
if EI is not None:
    det_inb = dict(zip(EI['img_iid'].astype(str), EI['img_pgF_det'].astype(bool)))
    iids = EI['img_iid'].astype(str); h = EI['img_pgF_det'].astype(bool)
    put('det.inb.pg', ci_txt(cluster_ci(h, inb_pat(iids))), h.mean()); put('det.inb.iouF', f3(EI['img_iF_det'].mean()))
    put('det.inb.diceF', f3(EI['img_dF_det'].mean()))
for model in ('gmic', 'glam'):
    J = load_json(f'{REV}/wsol_{model}.json'); Z = load_npz(f'{REV}/wsol_{model}.npz')
    if not J or Z is None: continue
    for ds in ('cbis', 'inbreast'):
        if ds not in J: continue
        s = J[ds]; keys = Z[f'{ds}_keys'].astype(str); scored = Z[f'{ds}_scored'].astype(bool); hk = keys[scored]
        h = Z[f'{ds}_pg'].astype(bool)
        if ds == 'cbis' and W is not None:
            k2p = dict(zip(W['info_key'].astype(str), wpat)); pat = np.array([k2p[k] for k in hk])
            ref = dict(zip(W['info_key'].astype(str), det_hit)) if det_hit is not None else None
        else:
            pat = inb_pat(hk); ref = det_inb
        put(f'wsol.{model}.{ds}.pg', ci_txt(cluster_ci(h, pat)), h.mean()); put(f'wsol.{model}.{ds}.pg.rate', pc(h.mean()), h.mean())
        put(f'wsol.{model}.{ds}.nfail', str(int(s.get('n_failed', 0))))
        put(f'wsol.{model}.{ds}.iou', f3(s['iou'])); put(f'wsol.{model}.{ds}.dice', f3(s['dice']))
        put(f'wsol.{model}.{ds}.pgB', pc(s['pointing_benign_map'])); put(f'wsol.{model}.{ds}.pgM', pc(s['pointing_malignant_map']))
        put(f'wsol.{model}.{ds}.each', pmp(list(s['pointing_each'].values())))
        put(f'wsol.{model}.{ds}.inwin', pc(s['lesion_fraction_inside_window']))
        put(f'wsol.{model}.{ds}.n', str(int(scored.sum())))
        if 'auc_malignant' in s: put(f'wsol.{model}.{ds}.auc', f3(s['auc_malignant']))
        if 'auc_mass_vs_normal' in s: put(f'wsol.{model}.{ds}.auc', f3(s['auc_mass_vs_normal']))
        if ref is not None:
            a, b_, p = mcnemar(np.array([ref[k] for k in hk]), h); put(f'wsol.{model}.{ds}.mcn', fp(p), p); put(f'wsol.{model}.{ds}.disc', f'{a}/{b_}')
        if ds == 'inbreast':
            put('chance.inb', pc(Z['inbreast_area'].mean()))

# ================================================================== 5. strata: lesion size, abnormality type, classification correctness
if W is not None and det_hit is not None:
    side = W['info_max_side']; q = np.quantile(side, [1 / 3, 2 / 3]); lab = W['info_label'].astype(int)
    strata = {'small': side < q[0], 'medium': (side >= q[0]) & (side < q[1]), 'large': side >= q[1], 'mass': lab == 1, 'calc': lab == 0}
    put('strata.cut1', f'{q[0]:.0f}'); put('strata.cut2', f'{q[1]:.0f}')
    for s, msk in strata.items(): put(f'strata.n.{s}', str(int(msk.sum())))
    meth = {'det': (det_hit, None), 'ftgc': (W['ft_gradcam_pgF'], W['info_p_ft']), 'ftlc': (W['ft_layercam_pgF'], W['info_p_ft']),
            'mil': (W['mil_pg_tileF'], W['info_p_mil'])}
    Zm = load_npz(REV + '/mc_heads.npz')
    if Zm is not None and 'r1520_mil_mil_attention_s0' in Zm.files:
        meth['mcmil'] = (np.mean([Zm[f'r1520_mil_mil_attention_s{s}'][:, 0] for s in (0, 1, 2)], 0),
                         np.mean([Zm[f'r1520_mil_prob_s{s}'] for s in (0, 1, 2)], 0))
    for model in ('gmic', 'glam'):
        Zw = load_npz(f'{REV}/wsol_{model}.npz')
        if Zw is not None:
            k2h = dict(zip(Zw['cbis_keys'].astype(str)[Zw['cbis_scored'].astype(bool)], Zw['cbis_pg']))
            meth[model] = (np.array([k2h[k] for k in W['info_key'].astype(str)]), None)
    for name, (h, prob) in meth.items():
        h = np.asarray(h, float)
        for s, msk in strata.items(): put(f'strata.{name}.{s}', pcn(h[msk].mean()))
        if prob is not None:
            corr = (np.asarray(prob) >= 0.5).astype(int) == lab
            put(f'strata.{name}.correct', pcn(h[corr].mean())); put(f'strata.{name}.incorrect', pcn(h[~corr].mean()))
            put(f'strata.{name}.ncorrect', str(int(corr.sum()))); put(f'strata.{name}.nincorrect', str(int((~corr).sum())))
    if EC is not None:                                        # lesion-level detection by size tercile and type
        ls = EC['les_side']; lq = np.quantile(ls, [1 / 3, 2 / 3]); dt = EC['les_detected'].astype(bool); ab = EC['les_abn'].astype(str)
        for nm, msk in (('small', ls < lq[0]), ('medium', (ls >= lq[0]) & (ls < lq[1])), ('large', ls >= lq[1]),
                        ('mass', np.char.find(np.char.lower(ab), 'mass') >= 0), ('calc', np.char.find(np.char.lower(ab), 'calc') >= 0)):
            put(f'det.cbis.sens.{nm}', pc(dt[msk].mean())); put(f'det.cbis.n.{nm}', str(int(msk.sum())))

# ================================================================== 6. full-image detectors
def det_block(tag, label):
    jc = load_json(f'{E2E}/cbis_e2e_{tag}.json'); ji = load_json(f'{E2E}/inbreast_e2e_{tag}.json'); out = {}
    if jc:
        out.update(cdice=jc['per_image_256']['det'], cpg=jc['detector_pg256'], cpgF=jc['per_image_fullres']['det']['pg'],
                   csens=jc['operating_point_thr05']['sensitivity'], cfpi=jc['operating_point_thr05']['fp_per_image'],
                   ccpm=jc['froc_local_maxima']['cpm'], cs1=jc['froc_local_maxima']['sens_at_fpi']['1'],
                   cs05=jc['froc_local_maxima']['sens_at_fpi']['0.5'], ccpmS=jc['froc_threshold_sweep']['cpm'],
                   cfb=jc['detections_per_image']['fallback_images'])
    if ji:
        out.update(idice=ji['per_image_fullres']['det']['dice'], ipg=ji['per_image_fullres']['det']['pg'],
                   isens=ji['operating_point_thr05']['sensitivity'], ifpm=ji['operating_point_thr05']['fp_per_mass_image'],
                   ifpn=ji['operating_point_thr05']['fp_per_normal_image'], iany=ji['operating_point_thr05']['normals_with_any_detection'],
                   icpm=ji['froc_local_maxima']['cpm'], is1=ji['froc_local_maxima']['sens_at_fp_per_mass_image']['1'])
    for k, v in out.items():
        fmt = f3 if k in ('cdice', 'idice', 'ccpm', 'icpm', 'ccpmS') else (lambda x: str(int(x))) if k == 'cfb' else (lambda x: f'{x:.2f}') if k in ('cfpi', 'ifpm', 'ifpn') else pc
        put(f'det.{label}.{k}', fmt(v), v)
    for ds, d in (('cbis', 'c'), ('inbreast', 'i')):              # where the local-maximum FROC ends
        Z = load_npz(f'{E2E}/{ds}_e2e_{tag}.npz')
        if Z is not None and 'froc_tp' in Z.files:
            t = Z['froc_tp'].astype(bool)
            put(f'det.{label}.{d}endfpi', f'{(~t).sum() / int(Z["n_images"]):.2f}'); put(f'det.{label}.{d}endsens', pc(t.sum() / int(Z['n_lesions_unique'])))
    return out


DET = {}
for tag, label in (('pub', 'v3'), ('s1', 'v3s1'), ('s2', 'v3s2'), ('nnunet_pub', 'nnunet'), ('segformer_pub', 'segformer')):
    DET[label] = det_block(tag, label)
seeds = [DET[l] for l in ('v3', 'v3s1', 'v3s2') if DET.get(l)]
if len(seeds) >= 2:
    for k in seeds[0]:
        if all(k in s for s in seeds):
            v = [s[k] for s in seeds]
            put(f'det.v3seeds.{k}', pmp(v) if k not in ('cdice', 'idice', 'ccpm', 'icpm', 'ccpmS', 'cfpi', 'ifpm', 'ifpn', 'cfb') else pmf(v, 3 if k not in ('cfpi', 'ifpm', 'ifpn', 'cfb') else 2))
            if k == 'cfb': put('det.v3seeds.cfbi', pmf(v, 0))          # images without a detection: whole numbers
    put('det.v3seeds.n', str(len(seeds)))
for s in (1, 2):                                            # the seed scripts' own 256-grid numbers (as in detector_v3.py)
    j = load_json(f'{REV}/detector_v3_s{s}.json')
    if j: put(f'det.v3s{s}.own', f'{j["test_dice"]:.3f} / {100 * j["test_pg"]:.1f}\\%')
SF = load_json(REV + '/segformer_mit_b2_s0.json')
if SF: put('segformer.epoch', str(SF['best_epoch'])); put('segformer.params', f'{SF["params_m"]:.1f}')
if ECJ: put('det.cbis.fallback', str(ECJ['detections_per_image']['fallback_images']))

# ================================================================== 7. conditional vs end-to-end (per lesion, same window)
def e2e_block(tag, ds):
    Z = load_npz(f'{E2E}/{ds}_e2e_{tag}.npz')
    if Z is None: return None
    L = {k[4:]: Z[k] for k in Z.files if k.startswith('les_')}
    cl = L['patient'] if ds == 'cbis' else inb_pat(L['iid'])
    d = 'cbis' if ds == 'cbis' else 'inb'
    for n in ('oracle', 'oracle_canvas', 'det', 'top1', 'all', 'ver'):
        if 'd_' + n not in L: continue
        x = L['d_' + n]; ci = cluster_ci(x, cl)
        put(f'e2e.{tag}.{d}.{n}', ci_txt(ci, pct=False), ci[0]); put(f'e2e.{tag}.{d}.{n}.dice', f3(ci[0]), ci[0])
        put(f'e2e.{tag}.{d}.{n}.gt05', pc((x > 0.5).mean()))
    if 'd_all' in L:
        put(f'e2e.{tag}.{d}.p_all_det', fp(wil(L['d_all'], L['d_det'])))
        dd = cluster_ci(L['d_all'] - L['d_det'], cl); put(f'e2e.{tag}.{d}.diff_all_det', f'{dd[0]:+.3f} [{dd[1]:+.3f}, {dd[2]:+.3f}]')
        do = cluster_ci(L['d_oracle'] - L['d_all'], cl); put(f'e2e.{tag}.{d}.diff_oracle_all', f'{do[0]:.3f} [{do[1]:.3f}, {do[2]:.3f}]')
        dt = L['detected'].astype(bool)
        put(f'e2e.{tag}.{d}.detected', pc(dt.mean()))
        put(f'e2e.{tag}.{d}.det_all', f3(L['d_all'][dt].mean())); put(f'e2e.{tag}.{d}.det_oracle', f3(L['d_oracle'][dt].mean()))
        put(f'e2e.{tag}.{d}.det_det', f3(L['d_det'][dt].mean())); put(f'e2e.{tag}.{d}.miss_all', f3(L['d_all'][~dt].mean()))
        put(f'e2e.{tag}.{d}.det_gt05', pc((L['d_all'][dt] > 0.5).mean()))
        if ds == 'cbis':
            ab = np.char.lower(L['abn'].astype(str))
            for nm in ('mass', 'calc'):
                msk = np.char.find(ab, nm) >= 0
                put(f'e2e.{tag}.cbis.{nm}.all', f3(L['d_all'][msk].mean())); put(f'e2e.{tag}.cbis.{nm}.oracle', f3(L['d_oracle'][msk].mean()))
                put(f'e2e.{tag}.cbis.{nm}.det_all', f3(L['d_all'][msk & dt].mean())); put(f'e2e.{tag}.cbis.{nm}.detected', pc(dt[msk].mean()))
    return L


EL = {}
for tag in ('pub', 'jit', 's1', 's2', 'nnunet_pub', 'nnunet_jit', 'segformer_pub', 'segformer_jit'):
    for ds in ('cbis', 'inbreast'):
        EL[(tag, ds)] = e2e_block(tag, ds)
for ds, d in (('cbis', 'cbis'), ('inbreast', 'inb')):          # pipeline-aware delineator, 3 seeds (detector + delineator retrained)
    v = [EL[(t, ds)]['d_all'].mean() for t in ('jit', 's1', 's2') if EL.get((t, ds)) is not None]
    o = [EL[(t, ds)]['d_oracle'].mean() for t in ('jit', 's1', 's2') if EL.get((t, ds)) is not None]
    if len(v) >= 2: put(f'e2e.seeds.{d}.all', pmf(v)); put(f'e2e.seeds.{d}.oracle', pmf(o)); put(f'e2e.seeds.{d}.n', str(len(v)))
for s in (2021, 1, 2):
    j = load_json(f'{REV}/jitter_roi_s{s}.json') or load_json(f'{REV}/roi_jitter_s{s}.json')
    if j and 'oracle_dice' in j: put(f'jit.s{s}.oracle', f3(j['oracle_dice']))

# ================================================================== 8. crop-size sensitivity (and the geometric prior)
CS = load_json(REV + '/crop_sensitivity.json')
if CS:
    for ds, d in (('cbis', 'cbis'), ('inbreast', 'inb')):
        for m, r in CS.get(ds, {}).items():
            mm = m.replace('.', '')
            for k, v in r.items():
                if k != 'n': put(f'crop.{d}.{mm}.{k}', f3(v), v)
        if '1.8' in CS.get(ds, {}): put(f'prior.{d}', f3(CS[ds]['1.8']['prior']), CS[ds]['1.8']['prior'])

# ================================================================== 9. VinDr-Mammo (second external cohort; DICOM, box annotations)
# bootstrap clusters = studies (one study = the four views of one examination; VinDr-Mammo has no patient identifier)
VTAGS = (('pub', 'v3'), ('minmax', 'v3mm'), ('s1', 'v3s1'), ('s2', 'v3s2'), ('nnunet_pub', 'nnunet'), ('segformer_pub', 'segformer'), ('jit', 'jit'))
VZ, VS = {}, {}
for tag, lab in VTAGS:
    J = load_json(f'{E2E}/vindrdcm_e2e_{tag}.json'); Z = load_npz(f'{E2E}/vindrdcm_e2e_{tag}.npz')
    if not J or Z is None: continue
    VZ[lab] = Z; st = Z['img_study'].astype(str); pg = Z['img_pg'].astype(bool); hm = Z['img_has_mass'] == 1
    tst = np.char.endswith(Z['img_set'].astype(str), '_test'); lst = Z['les_study'].astype(str); ld = Z['les_detected'].astype(float)
    op = J['operating_point_thr05']; fr = J['froc_local_maxima']
    if lab == 'v3':
        put('vd.n.img', f'{len(pg):,}'); put('vd.n.massimg', f'{int(hm.sum()):,}'); put('vd.n.calcimg', f'{int((~hm).sum()):,}')
        put('vd.n.les', f'{J["n_lesions"]:,}'); put('vd.n.mass', f'{J["n_masses"]:,}'); put('vd.n.calc', f'{J["n_calcifications"]:,}')
        put('vd.n.norm', str(J['n_normals'])); put('vd.n.testimg', str(int(tst.sum()))); put('vd.n.study', f'{len(np.unique(st)):,}')
        _, inv = np.unique(Z['les_image'].astype(str), return_inverse=True)
        put('chance.vd', pc(np.bincount(inv, weights=Z['les_frac']).clip(max=1).mean()))       # box area fraction per image
    put(f'vd.{lab}.pg', ci_txt(cluster_ci(pg, st)), pg.mean()); put(f'vd.{lab}.pg.rate', pc(pg.mean()), pg.mean())
    put(f'vd.{lab}.pgmass', pc(pg[hm].mean())); put(f'vd.{lab}.pgcalc', pc(pg[~hm].mean())); put(f'vd.{lab}.pgtest', pc(pg[tst].mean()))
    put(f'vd.{lab}.pgany', pc(Z['img_pg_any'].mean()))
    put(f'vd.{lab}.sens', pc(op['sensitivity'])); put(f'vd.{lab}.sens.ci', ci_txt(cluster_ci(ld, lst)))
    put(f'vd.{lab}.sensmass', pc(op['sensitivity_mass'])); put(f'vd.{lab}.senscalc', pc(op['sensitivity_calc']))
    put(f'vd.{lab}.senstest', pc(ld[np.char.endswith(Z['les_set'].astype(str), '_test')].mean()))
    put(f'vd.{lab}.fpi', f2(op['fp_per_image'])); put(f'vd.{lab}.fpn', f2(op['fp_per_normal_image'])); put(f'vd.{lab}.any', pc(op['normals_with_any_detection']))
    put(f'vd.{lab}.cpm', f3(fr['cpm'])); put(f'vd.{lab}.s05', pc(fr['sens_at_fp_per_image']['0.5'])); put(f'vd.{lab}.s1', pc(fr['sens_at_fp_per_image']['1']))
    put(f'vd.{lab}.endsens', pc(fr['end_sensitivity'])); put(f'vd.{lab}.endfpi', f2(fr['end_fp_per_image']))
    put(f'vd.{lab}.nodet', str(J['detections_per_image']['fallback_images']))
    VS[lab] = dict(pg=pg.mean(), sens=op['sensitivity'], sensmass=op['sensitivity_mass'], senscalc=op['sensitivity_calc'], fpi=op['fp_per_image'],
                   fpn=op['fp_per_normal_image'], cpm=fr['cpm'])
    if J.get('box_level'):                                   # pipeline at box level (per lesion, in the lesion's evaluation window)
        dt = Z['les_detected'] == 1
        for n in ('oracle', 'prior', 'det', 'all'):
            b = Z['les_biou_' + n].astype(float); okb = np.isfinite(b)
            put(f'vd.{lab}.biou.{n}', f3(b[okb].mean())); put(f'vd.{lab}.biou.{n}.ci', ci_txt(cluster_ci(b[okb], lst[okb]), pct=False))
            put(f'vd.{lab}.biou.{n}.gt05', pc((b[okb] > 0.5).mean())); put(f'vd.{lab}.cont.{n}', f3(np.nanmean(Z['les_cont_' + n])))
            put(f'vd.{lab}.biou.{n}.det', f3(b[okb & dt].mean())); put(f'vd.{lab}.biou.{n}.miss', f3(b[okb & ~dt].mean()))
        a_, d_ = Z['les_biou_all'].astype(float), Z['les_biou_det'].astype(float); okb = np.isfinite(a_) & np.isfinite(d_)
        dd = cluster_ci(a_[okb] - d_[okb], lst[okb]); put(f'vd.{lab}.biou.diff_all_det', f'{dd[0]:+.3f} [{dd[1]:+.3f}, {dd[2]:+.3f}]')
        put(f'vd.{lab}.biou.p_all_det', fp(wil(a_[okb], d_[okb])))
        po = Z['les_pg_oracle']; put(f'vd.{lab}.pgoracle', pc((po[po >= 0] == 1).mean()))
vs = [VS[l] for l in ('v3', 'v3s1', 'v3s2') if l in VS]
if len(vs) >= 2:
    for k in vs[0]: put(f'vd.v3seeds.{k}', pmf([s[k] for s in vs], 3 if k == 'cpm' else 2) if k in ('fpi', 'fpn', 'cpm') else pmp([s[k] for s in vs]))
for model in ('gmic', 'glam'):                               # dedicated weakly-supervised localizers on the same lesion images
    J = load_json(f'{REV}/wsol_{model}_vindr.json'); Z = load_npz(f'{REV}/wsol_{model}_vindr.npz')
    if not J or Z is None or 'vindr' not in J: continue
    s = J['vindr']; hk = Z['vindr_keys'].astype(str)[Z['vindr_scored'].astype(bool)]; h = Z['vindr_pg'].astype(bool)
    put(f'vd.{model}.pg.rate', pc(h.mean()), h.mean()); put(f'vd.{model}.n', f'{len(h):,}'); put(f'vd.{model}.nfail', str(s['n_failed']))
    put(f'vd.{model}.misspct', f'{100 * (1 - h.mean()):.0f}')
    put(f'vd.{model}.auc', f3(s['auc_lesion_vs_normal'])); put(f'vd.{model}.pgB', pc(s['pointing_benign_map'])); put(f'vd.{model}.pgM', pc(s['pointing_malignant_map']))
    if 'vindr_pg_any' in Z.files: put(f'vd.{model}.pgany', pc(Z['vindr_pg_any'].mean()))
    if 'v3' in VZ:
        v = VZ['v3']; vi = v['img_image'].astype(str); row = {k: j for j, k in enumerate(vi)}; o = np.array([row[k] for k in hk])
        put(f'vd.{model}.pg', ci_txt(cluster_ci(h, v['img_study'].astype(str)[o])), h.mean())
        hm = v['img_has_mass'][o] == 1; put(f'vd.{model}.pgmass', pc(h[hm].mean())); put(f'vd.{model}.pgcalc', pc(h[~hm].mean()))
        a, b_, p = mcnemar(v['img_pg'].astype(bool)[o], h); put(f'vd.{model}.mcn', fp(p), p); put(f'vd.{model}.disc', f'{a}/{b_}')
        dv = v['img_pg'].astype(bool)[o]; tst = np.char.endswith(v['img_set'].astype(str)[o], '_test')
        put(f'vd.{model}.pgtest', pc(h[tst].mean()))
        for nm, msk in (('mass', hm), ('calc', ~hm)):         # the same paired test within mass images / calcification-only images
            a, b_, p = mcnemar(dv[msk], h[msk]); put(f'vd.{model}.mcn.{nm}', fp(p), p); put(f'vd.{model}.disc.{nm}', f'{a}/{b_}')
        pv = []                                               # against every supervised detector (three seeds, nnU-Net, SegFormer)
        for lab2 in ('v3', 'v3s1', 'v3s2', 'nnunet', 'segformer'):
            if lab2 in VZ:
                r2 = {k: j for j, k in enumerate(VZ[lab2]['img_image'].astype(str))}
                pv.append(mcnemar(VZ[lab2]['img_pg'].astype(bool)[[r2[k] for k in hk]], h)[2])
        if pv: put(f'vd.{model}.mcn.maxdet', fp(max(pv), up=True), max(pv)); put(f'vd.{model}.mcn.ndet', str(len(pv)))
PS = load_json('D:/vindr_dicom/cache/prep_summary.json')
if PS and not PS.get('n_missing'):                           # scanner census of the evaluated images
    for man in ('SIEMENS', 'Planmed', 'IMS'): put(f'vd.census.{man.lower()}', f'{sum(c["n"] for c in PS["census"] if c["manufacturer"] == man):,}')
    put('vd.census.mono1', f'{sum(c["n"] for c in PS["census"] if c["photometric"] == "MONOCHROME1"):,}'); put('vd.census.n', f'{PS["n"]:,}')
    put('vd.band.voi', pc(PS['breast_intensity_band']['voi_lut'], 0)); put('vd.band.minmax', pc(PS['breast_intensity_band']['minmax'], 0))

# ================================================================== bounds quoted in the abstract
# largest McNemar p of any weakly-supervised map against the supervised detector ON THE CBIS-DDSM TEST SET (B0 maps, both
# encoders at every resolution / head / seed, GMIC and GLAM). The external cohorts are quoted separately: on VinDr-Mammo
# GLAM points at lesions MORE often than the detector, so no single bound holds across datasets.
ps = [v for k, v in RAW.items() if re.match(r'^(ws\.\w+|(mc|in)\.r\d+\.\w+\.\w+|wsol\.\w+\.cbis)\.mcn$', k) and isinstance(v, float)]
if ps: put('abs.mcn.max', fp(max(ps), up=True), max(ps)); put('abs.mcn.nmaps', str(len(ps)))

# ================================================================== code repository (one line in code_url.txt next to the manuscript)
cu = os.path.join(os.path.dirname(TEX), 'code_url.txt')
if os.path.exists(cu) and open(cu, encoding='utf-8').read().strip():
    put('code.url', '\\url{' + open(cu, encoding='utf-8').read().strip() + '}')

# ================================================================== write
with open(TEX, 'w', encoding='utf-8') as fh:
    fh.write('% generated by make_numbers.py - do not edit\n\\makeatletter\n')
    for k in sorted(K): fh.write(f'\\@namedef{{num@{k}}}{{{K[k]}}}\n')
    fh.write('\\makeatother\n')
with open(TEX.replace('.tex', '.md'), 'w', encoding='utf-8') as fh:
    for k in sorted(K): fh.write(f'{k}\t{K[k]}\n')
print(f'{len(K)} keys -> {TEX}')
