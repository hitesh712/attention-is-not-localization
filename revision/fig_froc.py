"""FROC curves of the full-image detector (CBIS test; INbreast external, against false positives per mass
image and per BI-RADS 1 normal image). Lesion-level pointing criterion; see e2e_common.match_points."""
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

E2E = 'G:/My Drive/VS_CC/models/e2e'; OUT = 'G:/My Drive/VS_CC/models/revision'
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.linewidth': 0.6, 'lines.linewidth': 1.3})


def curve(scores, tp, n_les, n_img):
    s = np.asarray(scores, float); t = np.asarray(tp, bool); o = np.argsort(-s, kind='stable'); t = t[o]
    return np.concatenate([[0], np.cumsum(~t) / n_img]), np.concatenate([[0], np.cumsum(t) / n_les])


def curve_vs_normals(scores, tp, n_les, normal_peaks, n_norm):
    s_tp = np.sort(np.asarray(scores, float)[np.asarray(tp, bool)])[::-1]; s_fp = np.sort(np.asarray(normal_peaks, float))[::-1]
    th = np.unique(np.concatenate([s_tp, s_fp]))[::-1]
    return (np.array([np.sum(s_fp >= t) / n_norm for t in th]), np.array([np.sum(s_tp >= t) / n_les for t in th]))


C = np.load(E2E + '/cbis_e2e.npz', allow_pickle=True); I = np.load(E2E + '/inbreast_e2e.npz', allow_pickle=True)
fig, ax = plt.subplots(figsize=(3.45, 2.6), dpi=300)
x, y = curve(C['froc_scores'], C['froc_tp'], int(C['n_lesions_unique']), int(C['n_images']))
ax.step(x, y, where='post', color='#0072B2', label=f'CBIS-DDSM test (FP per image)')
x, y = curve(I['froc_scores'], I['froc_tp'], int(I['n_lesions_unique']), int(I['n_images']))
ax.step(x, y, where='post', color='#D55E00', label='INbreast masses (FP per mass image)')
x, y = curve_vs_normals(I['froc_scores'], I['froc_tp'], int(I['n_lesions_unique']), I['normal_peaks'], int(I['n_normals']))
ax.step(x, y, where='post', color='#D55E00', ls='--', label='INbreast masses (FP per normal image)')
for name, Z, fp in (('CBIS', C, 'fp_per_image'), ('INbreast', I, None)):
    tp = np.asarray(Z['op_tp'], bool); sens = tp.sum() / int(Z['n_lesions_unique']); fpi = (~tp).sum() / int(Z['n_images'])
    ax.plot(fpi, sens, 'o', ms=3.5, color='#0072B2' if name == 'CBIS' else '#D55E00', zorder=5)
ax.set_xscale('log'); ax.set_xlim(1 / 16, 16); ax.set_ylim(0, 1)
ax.set_xticks([0.125, 0.25, 0.5, 1, 2, 4, 8]); ax.set_xticklabels(['1/8', '1/4', '1/2', '1', '2', '4', '8'])
ax.set_xlabel('False positives per image'); ax.set_ylabel('Lesion-level sensitivity')
ax.grid(alpha=0.3, lw=0.4); ax.legend(frameon=False, fontsize=6.5, loc='lower right')
ax.text(0.07, 0.95, 'dots: operating threshold p = 0.5', fontsize=6, transform=ax.transAxes, va='top')
fig.tight_layout(); fig.savefig(OUT + '/FIG_froc.pdf'); fig.savefig(OUT + '/FIG_froc.png', dpi=300)
print('saved FIG_froc')
