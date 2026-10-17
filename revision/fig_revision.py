"""Figures of the revised manuscript (re-run whenever more results exist; missing inputs are skipped).
  FIG_resolution.pdf : pointing vs input resolution (Mammo-CLIP / ImageNet heads) + reference methods
  FIG_froc_rev.pdf   : local-maximum FROC of the full-image detectors, CBIS-DDSM and INbreast
  FIG_e2e.pdf        : conditional vs end-to-end per-lesion Dice
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

M = 'G:/My Drive/VS_CC/models'; REV = M + '/revision'; E2E = M + '/e2e'
FIG = 'D:/Research 9.9.26/current research/IEEE_Access/figures'
plt.rcParams.update({'font.size': 7.5, 'font.family': 'serif', 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                     'ytick.major.width': 0.6, 'legend.frameon': False, 'pdf.fonttype': 42})
lj = lambda p: json.load(open(p)) if os.path.exists(p) else None
ln = lambda p: np.load(p, allow_pickle=True) if os.path.exists(p) else None


def save(fig, name):
    fig.savefig(f'{FIG}/{name}.pdf', bbox_inches='tight'); fig.savefig(f'{FIG}/{name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig); print('saved', name)


# ------------------------------------------------------------------ 1. resolution
def fig_resolution():
    RES = ('r380', 'r760', 'r1520'); X = np.arange(3)
    series = (('mil', 'mil_attention', 'MIL attention', 'C3'), ('cam', 'cam', 'CAM', 'C0'),
              ('cbam', 'gradcam', 'CBAM Grad-CAM', 'C2'), ('cbam', 'cbam_spatial', 'CBAM spatial map', 'C7'))
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    for suf, ls, tag in (('', '-', 'Mammo-CLIP'), ('_imagenet', '--', 'ImageNet')):
        R = lj(f'{REV}/mc_heads{suf}.json')
        if not R: continue
        for kind, mp, lab, c in series:
            m, s = [], []
            for r in RES:
                v = [R[f'{r}_{kind}_s{k}'][mp]['pointing'] * 100 for k in (0, 1, 2) if f'{r}_{kind}_s{k}' in R]
                m.append(np.mean(v) if v else np.nan); s.append(np.std(v) if v else np.nan)
            ax.errorbar(X + (0.04 if suf else 0), m, yerr=s, color=c, ls=ls, marker='o' if not suf else 's', ms=3, lw=1.0,
                        capsize=2, label=f'{lab} ({tag})')
    refs = []
    j = lj(f'{E2E}/cbis_e2e_pub.json')
    if j: refs.append((100 * j['per_image_fullres']['det']['pg'], 'Supervised detector (256$\\times$256)', 'k', '-'))
    for mdl, c in (('gmic', 'C1'), ('glam', 'C4')):
        w = lj(f'{REV}/wsol_{mdl}.json')
        if w and 'cbis' in w: refs.append((100 * w['cbis']['pointing'], f'{mdl.upper()} (NYU, full resolution)', c, ':'))
    w = lj(f'{REV}/ws_reeval.json')
    if w: refs.append((100 * max(v['pgF'] for k, v in w.items() if k.startswith('ft_') or k.startswith('fz_')),
                       'Best EfficientNet-B0 map (224$\\times$224)', 'C5', '-.'))
    g = 'D:/cbis-ddsm/_mc_cache/gt_test_crop760.npy'
    if os.path.exists(g):
        gg = np.load(g); refs.append((100 * gg.reshape(len(gg), -1).mean(1).mean(), 'Chance', '0.6', '-'))
    refs.sort(key=lambda r: r[0]); ty, last = [], -1e9           # stagger labels of nearby lines (>= 5 points apart)
    for y, *_ in refs: last = max(y, last + 5.0); ty.append(last)
    for (y, lab, c, ls), t in zip(refs, ty):
        ax.axhline(y, color=c, ls=ls, lw=0.8)
        ax.annotate(lab, xy=(2.1, y), xytext=(2.17, t), va='center', fontsize=5.8, color=c, annotation_clip=False,
                    arrowprops=dict(arrowstyle='-', lw=0.4, color=c, shrinkA=0, shrinkB=0) if abs(t - y) > 0.5 else None)
    ax.set_xticks(X); ax.set_xticklabels(['380$\\times$228', '760$\\times$456', '1520$\\times$912'])
    ax.set_xlim(-0.2, 2.1); ax.set_ylim(0, 84); ax.set_yticks(range(0, 61, 10))
    ax.set_ylabel('Pointing game (%)'); ax.set_xlabel('Input resolution (breast crop)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=5.4, loc='upper left', ncol=2, handlelength=2.0, columnspacing=0.8)
    save(fig, 'FIG_resolution')


# ------------------------------------------------------------------ 2. FROC (local maxima)
FPI = (0.125, 0.25, 0.5, 1, 2, 4, 8)


def curve(scores, tp, n_les, n_den):
    o = np.argsort(-np.asarray(scores, float), kind='stable'); t = np.asarray(tp, bool)[o]
    return np.cumsum(~t) / n_den, np.cumsum(t) / n_les, np.asarray(scores, float)[o]


def fig_froc():
    DET = (('pub', 'Attention U-Net v3', 'k', '-', 1.2), ('s1', 'v3, seed 1', '0.45', '-', 0.7), ('s2', 'v3, seed 2', '0.65', '-', 0.7),
           ('nnunet_pub', 'nnU-Net (2D)', 'C0', '-', 1.1), ('segformer_pub', 'SegFormer (MiT-B2)', 'C3', '-', 1.1))
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.5))
    for ax, ds in zip(axs, ('cbis', 'inbreast')):
        for tag, lab, c, ls, lw in DET:
            Z = ln(f'{E2E}/{ds}_e2e_{tag}.npz'); J = lj(f'{E2E}/{ds}_e2e_{tag}.json')
            if Z is None or J is None: continue
            nl = int(Z['n_lesions_unique']); ni = int(Z['n_images'])
            fpi, sen, sc = curve(Z['froc_scores'], Z['froc_tp'], nl, ni)
            ax.step(np.r_[1e-3, fpi], np.r_[0, sen], where='post', color=c, ls=ls, lw=lw, label=lab)
            op = J['operating_point_thr05']
            ax.plot(op['fp_per_image'] if ds == 'cbis' else op['fp_per_mass_image'], op['sensitivity'], 'o', color=c, ms=3.5, mfc='white', mew=1.0)
            if ds == 'inbreast' and 'normal_peaks' in Z.files:          # same candidates, FP counted on the normal images
                tps = np.asarray(Z['froc_scores'], float)[np.asarray(Z['froc_tp'], bool)]
                nps = np.sort(np.asarray(Z['normal_peaks'], float))[::-1]; nn = int(J['n_normals'])
                thr = np.unique(np.r_[tps, nps])[::-1]
                ax.step(np.r_[1e-3, [(nps >= t).sum() / nn for t in thr]], np.r_[0, [(tps >= t).sum() / nl for t in thr]],
                        where='post', color=c, ls=':', lw=lw * 0.8)
        # the axis starts at 1/32 so that the most conservative operating point (nnU-Net on INbreast) stays visible
        ax.set_xscale('log'); ax.set_xlim(0.028, 10); ax.set_ylim(0, 1); ax.set_xticks((1 / 32, 1 / 16) + FPI)
        ax.set_xticklabels(['1/32', '1/16', '1/8', '1/4', '1/2', '1', '2', '4', '8']); ax.minorticks_off()
        ax.set_xlabel('False positives per image' if ds == 'cbis' else 'FP per mass image (dotted: per normal image)')
        ax.set_ylabel('Lesion sensitivity'); ax.set_title('CBIS-DDSM' if ds == 'cbis' else 'INbreast (external)', fontsize=8)
        ax.grid(alpha=0.25, lw=0.4); ax.spines[['top', 'right']].set_visible(False)
    axs[0].legend(fontsize=6, loc='lower right')
    save(fig, 'FIG_froc_rev')


# ------------------------------------------------------------------ 3. conditional vs end-to-end
def fig_e2e():
    P = {(t, ds): ln(f'{E2E}/{ds}_e2e_{t}.npz') for t in ('pub', 'jit') for ds in ('cbis', 'inbreast')}
    if P[('pub', 'cbis')] is None: return
    CS = lj(f'{REV}/crop_sensitivity.json')
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.45), gridspec_kw=dict(width_ratios=[1.25, 1.25, 1]))
    for ax, ds, title in ((axs[0], 'cbis', 'CBIS-DDSM (422 lesions)'), (axs[1], 'inbreast', 'INbreast (116 masses)')):
        A, J = P[('pub', ds)], P[('jit', ds)]
        cols = [('Cond.\norig.', A['les_d_oracle'], 'C2'), ('Cond.\nPA', J['les_d_oracle'] if J is not None else None, 'C2'),
                ('Detector\nalone', A['les_d_det'], 'C0'), ('E2E\norig.', A['les_d_all'], 'C3'),
                ('E2E\nPA', J['les_d_all'] if J is not None else None, 'C3')]
        cols = [c for c in cols if c[1] is not None]
        vp = ax.violinplot([c[1] for c in cols], showmeans=False, showextrema=False, widths=0.8)
        for b, c in zip(vp['bodies'], cols): b.set_facecolor(c[2]); b.set_alpha(0.35); b.set_edgecolor('none')
        for i, c in enumerate(cols, 1):
            ax.plot([i - 0.25, i + 0.25], [c[1].mean()] * 2, color='k', lw=1.2)
            ax.text(i, 1.04, f'{c[1].mean():.2f}', ha='center', fontsize=6)
        if CS and ds in CS and '1.8' in CS[ds]:
            ax.axhline(CS[ds]['1.8']['prior'], color='C1', ls='--', lw=0.8); ax.text(0.55, CS[ds]['1.8']['prior'] + 0.02, 'centred disk (no image)', color='C1', fontsize=5.8,
                                                                                     bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none', alpha=0.8), zorder=5)
        ax.set_xticks(range(1, len(cols) + 1)); ax.set_xticklabels([c[0] for c in cols], fontsize=6.2)
        ax.set_ylim(0, 1.1); ax.set_ylabel('Per-lesion Dice'); ax.set_title(title, fontsize=8); ax.spines[['top', 'right']].set_visible(False)
    A = P[('pub', 'cbis')]; d = A['les_detected'].astype(bool)
    axs[2].scatter(A['les_d_oracle'][d], A['les_d_all'][d], s=5, c='C3', alpha=0.6, lw=0, label=f'detected ({d.sum()})')
    axs[2].scatter(A['les_d_oracle'][~d], A['les_d_all'][~d], s=5, c='0.5', alpha=0.6, lw=0, label=f'missed ({(~d).sum()})')
    axs[2].plot([0, 1], [0, 1], color='k', lw=0.5, ls=':')
    axs[2].set_xlabel('Conditional Dice (annotation crop)'); axs[2].set_ylabel('End-to-end Dice'); axs[2].set_xlim(0, 1); axs[2].set_ylim(0, 1)
    axs[2].set_title('CBIS-DDSM, per lesion', fontsize=8); axs[2].legend(fontsize=6, loc='upper left', markerscale=2)
    axs[2].spines[['top', 'right']].set_visible(False)
    fig.tight_layout(w_pad=1.0)
    save(fig, 'FIG_e2e')


if __name__ == '__main__':
    fig_resolution(); fig_froc(); fig_e2e()
