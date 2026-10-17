"""Statistics for the revision (Reviewers 1 & 2): paired tests and patient-clustered confidence intervals.

  * exact McNemar tests: supervised detector vs every weakly-supervised method, paired per image (n = 399)
  * patient-clustered bootstrap 95% CIs (patients resampled, so multi-lesion images and multiple images per
    patient are handled correctly); Wilson CIs kept for comparison with the original submission
  * attention localization stratified by lesion size (terciles) and by correct vs incorrect classification
  * end-to-end vs detector vs oracle: clustered CIs and paired differences (CBIS, INbreast)
Runs with whatever result files exist.
"""
import os, glob, re, json
import numpy as np
from scipy.stats import binomtest, wilcoxon

M = 'G:/My Drive/VS_CC/models'; REV = M + '/revision'; E2E = M + '/e2e'
B, RNG = 5000, np.random.RandomState(2021)


def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def cluster_ci(values, clusters, stat=np.mean):
    v = np.asarray(values, float); g = np.asarray(clusters)
    uniq, inv = np.unique(g, return_inverse=True); idx = [np.where(inv == k)[0] for k in range(len(uniq))]
    bs = []
    for _ in range(B):
        pick = RNG.randint(len(uniq), size=len(uniq))
        bs.append(stat(v[np.concatenate([idx[k] for k in pick])]))
    return float(stat(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def mcnemar(a, b):
    a = np.asarray(a, bool); b = np.asarray(b, bool); n10 = int(np.sum(a & ~b)); n01 = int(np.sum(~a & b))
    return dict(a_only=n10, b_only=n01, p=float(binomtest(min(n10, n01), n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0)


def fmt(t): return f'{t[0] * 100:.1f}% [{t[1] * 100:.1f}, {t[2] * 100:.1f}]'


OUT = {}; md = ['# Revision statistics', '']

# ------------------------------------------------------------------ 1. paired pointing: supervised vs weakly-supervised (CBIS, 399 images)
ws_p = REV + '/ws_reeval.npz'
e2e_p = E2E + ('/cbis_e2e_pub.npz' if os.path.exists(E2E + '/cbis_e2e_pub.npz') else '/cbis_e2e.npz')
if os.path.exists(ws_p) and os.path.exists(e2e_p):
    W = np.load(ws_p, allow_pickle=True); E = np.load(e2e_p, allow_pickle=True)
    pat = W['info_patient']; assert (pat == E['img_patient']).all(), 'image order mismatch'
    det = E['img_pgF_det'].astype(bool)
    methods = {'Supervised detector (v3)': det,
               'Frozen CBAM: Grad-CAM': W['fz_gradcam_pgF'], 'Frozen CBAM: spatial map': W['fz_cbam_pgF'],
               'Fine-tuned: Grad-CAM': W['ft_gradcam_pgF'], 'Fine-tuned: Grad-CAM++': W['ft_gradcampp_pgF'],
               'Fine-tuned: LayerCAM': W['ft_layercam_pgF'], 'Fine-tuned: Score-CAM': W['ft_scorecam_pgF'],
               'Fine-tuned: CBAM spatial map': W['ft_cbam_pgF'], 'Attention-MIL (top tile)': W['mil_pg_tileF']}
    if os.path.exists(M + '/perimage_v1_hit.npy'):
        methods['Supervised v1 (256 grid)'] = np.load(M + '/perimage_v1_hit.npy'); methods['Supervised v2 (256 grid)'] = np.load(M + '/perimage_v2_hit.npy')
    md += ['## Pointing game, CBIS test (n = 399 images; paired)', '',
           '| Method | Pointing (clustered 95% CI) | Wilson 95% CI | vs detector: detector-only / method-only hits | McNemar p |', '|---|---|---|---|---|']
    OUT['pointing'] = {}
    for name, h in methods.items():
        h = np.asarray(h).astype(bool); ci = cluster_ci(h, pat); w = wilson(int(h.sum()), len(h))
        mc = mcnemar(det, h) if name != 'Supervised detector (v3)' else None
        OUT['pointing'][name] = dict(rate=ci[0], ci=ci[1:], wilson=w, mcnemar=mc)
        md.append(f'| {name} | {fmt(ci)} | [{w[0] * 100:.1f}, {w[1] * 100:.1f}] | '
                  + (f'{mc["a_only"]} / {mc["b_only"]} | {mc["p"]:.1e} |' if mc else '— | — |'))
    # paired difference CI: detector minus best weakly-supervised
    best = max((n for n in methods if n.startswith(('Frozen', 'Fine', 'Attention'))), key=lambda n: np.mean(methods[n]))
    diff = cluster_ci(det.astype(float) - np.asarray(methods[best]).astype(float), pat)
    OUT['pointing']['difference_detector_minus_best_ws'] = dict(best_ws=best, diff=diff)
    md += ['', f'Detector minus best weakly-supervised method ({best}): {diff[0] * 100:.1f} percentage points '
               f'[{diff[1] * 100:.1f}, {diff[2] * 100:.1f}] (patient-clustered bootstrap).', '']

    # ---- stratification: lesion size and classification correctness (Reviewer 1 resolution; Reviewer 2 #6)
    side = W['info_max_side']; q = np.quantile(side, [1 / 3, 2 / 3]); lab = W['info_label']
    strata = {'small': side < q[0], 'medium': (side >= q[0]) & (side < q[1]), 'large': side >= q[1]}
    md += ['## Attention localization stratified by lesion size and by classification outcome (CBIS)', '',
           f'Size terciles by the longest side of the lesion mask at full resolution: small < {q[0]:.0f} px <= medium < {q[1]:.0f} px <= large.', '',
           '| Method | small | medium | large | correctly classified | misclassified |', '|---|---|---|---|---|---|']
    OUT['strata'] = {'size_cuts_px': q.tolist()}
    for name, key, prob in (('Fine-tuned: Grad-CAM', 'ft_gradcam_pgF', 'info_p_ft'), ('Fine-tuned: Grad-CAM++', 'ft_gradcampp_pgF', 'info_p_ft'),
                            ('Fine-tuned: Score-CAM', 'ft_scorecam_pgF', 'info_p_ft'), ('Attention-MIL (top tile)', 'mil_pg_tileF', 'info_p_mil'),
                            ('Supervised detector (v3)', None, None)):
        h = (det if key is None else W[key]).astype(bool)
        cells = {s: (float(h[m].mean()), int(m.sum())) for s, m in strata.items()}
        if prob:
            corr = (W[prob] >= 0.5).astype(int) == lab
            cells['correct'] = (float(h[corr].mean()), int(corr.sum())); cells['incorrect'] = (float(h[~corr].mean()), int((~corr).sum()))
        OUT['strata'][name] = cells
        md.append(f'| {name} | ' + ' | '.join(f'{cells[s][0] * 100:.1f}% (n={cells[s][1]})' if s in cells else '—'
                                              for s in ('small', 'medium', 'large', 'correct', 'incorrect')) + ' |')
    md.append('')

# ------------------------------------------------------------------ 2. Mammo-CLIP heads (stronger, higher-resolution classifier)
if os.path.exists(REV + '/mc_heads.json') and os.path.exists(REV + '/mc_heads.npz'):
    R = json.load(open(REV + '/mc_heads.json')); Z = np.load(REV + '/mc_heads.npz', allow_pickle=True)
    mpat = Z['patients']; md += ['## Frozen Mammo-CLIP encoder + attention heads (CBIS, mean ± SD over 3 seeds)', '',
                                 '| Input | Head | Map | Test AUC (rows) | Pointing | IoU | McNemar vs detector (seed 0) |', '|---|---|---|---|---|---|---|']
    OUT['mammoclip'] = {}
    for r in ('r380', 'r760', 'r1520'):
        for kind, maps in (('cam', ('cam',)), ('cbam', ('gradcam', 'cbam_spatial')), ('mil', ('mil_attention',))):
            runs = [R[f'{r}_{kind}_s{s}'] for s in (0, 1, 2) if f'{r}_{kind}_s{s}' in R]
            if not runs: continue
            auc = np.array([x['auc_rows'] for x in runs])
            for mp in maps:
                pg = np.array([x[mp]['pointing'] for x in runs]); iou = np.array([x[mp]['iou'] for x in runs])
                mc = None
                if os.path.exists(ws_p) and f'{r}_{kind}_{mp}_s0' in Z.files:
                    assert (mpat == pat).all(); mc = mcnemar(det, Z[f'{r}_{kind}_{mp}_s0'][:, 0].astype(bool))
                OUT['mammoclip'][f'{r}_{kind}_{mp}'] = dict(auc=(auc.mean(), auc.std()), pointing=(pg.mean(), pg.std()), iou=(iou.mean(), iou.std()), mcnemar=mc)
                md.append(f'| {r} | {kind} | {mp} | {auc.mean():.3f} ± {auc.std():.3f} | {pg.mean() * 100:.1f}% ± {pg.std() * 100:.1f} | '
                          f'{iou.mean():.3f} ± {iou.std():.3f} | ' + (f'{mc["a_only"]} / {mc["b_only"]}, p={mc["p"]:.1e}' if mc else '—') + ' |')
    md.append('')

# ------------------------------------------------------------------ 3. end-to-end (CBIS, INbreast): clustered CIs and paired differences
def e2e_block(tag, path, cluster_of):
    if not os.path.exists(path): return
    Z = np.load(path, allow_pickle=True); L = {k[4:]: Z[k] for k in Z.files if k.startswith('les_')}
    cl = cluster_of(L)
    md.extend([f'## End-to-end vs oracle, {tag} (per-lesion Dice in the published oracle window; patient-clustered 95% CI)', '',
               '| Pipeline | Dice | Dice > 0.5 |', '|---|---|---|'])
    OUT['e2e_' + tag] = {}
    for n, lbl in (('oracle', 'Oracle crop (ground-truth box)'), ('det', 'Detector alone'), ('all', 'End-to-end'), ('ver', 'End-to-end, verified')):
        if 'd_' + n not in L: continue
        d = L['d_' + n]; ci = cluster_ci(d, cl); g5 = cluster_ci(d > 0.5, cl)
        OUT['e2e_' + tag][n] = dict(dice=ci, gt05=g5)
        md.append(f'| {lbl} | {ci[0]:.3f} [{ci[1]:.3f}, {ci[2]:.3f}] | {fmt(g5)} |')
    dd = cluster_ci(L['d_all'] - L['d_det'], cl); do = cluster_ci(L['d_oracle'] - L['d_all'], cl)
    OUT['e2e_' + tag]['diff_e2e_minus_det'] = dd; OUT['e2e_' + tag]['diff_oracle_minus_e2e'] = do
    md.extend(['', f'End-to-end minus detector: {dd[0]:+.3f} [{dd[1]:+.3f}, {dd[2]:+.3f}] (Wilcoxon p = {wilcoxon(L["d_all"], L["d_det"]).pvalue:.1e}); '
               f'oracle minus end-to-end: {do[0]:+.3f} [{do[1]:+.3f}, {do[2]:+.3f}].', ''])


e2e_block('CBIS', e2e_p, lambda L: L['patient'])
inb_dcm = {m.group(1): m.group(2) for f in glob.glob('D:/INbreast Release 1.0/AllDICOMs/*.dcm')
           for m in [re.match(r'(\d+)_([0-9a-f]+)_', os.path.basename(f))] if m}
inb_p = E2E + ('/inbreast_e2e_pub.npz' if os.path.exists(E2E + '/inbreast_e2e_pub.npz') else '/inbreast_e2e.npz')
e2e_block('INbreast', inb_p, lambda L: np.array([inb_dcm.get(str(i), str(i)) for i in L['iid']]))

json.dump(OUT, open(REV + '/stats_revision.json', 'w'), indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
open(REV + '/stats_revision.md', 'w', encoding='utf-8').write('\n'.join(md))
print('\n'.join(md))
