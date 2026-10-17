"""Figure 1 of the revised manuscript: study design (same visual style as the submitted flowchart)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = 'D:/Research 9.9.26/current research/IEEE_Access/figures/FIG1_study_design_rev'
GREY = ('#ededed', '#8c8c8c'); WEAK = ('#fce8d7', '#dd8434'); WEAK2 = ('#f3c694', '#d97a2f')
SUP = ('#dce8f4', '#2f78bd'); SUP2 = ('#a6c8e7', '#2a6db0'); EXT = ('#e3f2ea', '#13a173'); STAT = ('#ffffff', '#666666')

fig = plt.figure(figsize=(10, 12.4)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 10); ax.set_ylim(0, 12.4); ax.axis('off')


def box(x0, y0, x1, y1, col, title, lines=(), tsize=12.5, lsize=10.3, bold=True, lw=2.2, gap=0.30):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle='round,pad=0.0,rounding_size=0.08',
                                fc=col[0], ec=col[1], lw=lw))
    n = len(lines); cx = (x0 + x1) / 2; cy = (y0 + y1) / 2
    top = cy + (n * gap) / 2 + (0.05 if n else 0)
    ax.text(cx, top if n else cy, title, ha='center', va='center', fontsize=tsize, fontweight='bold' if bold else 'normal', color='#1a1a1a')
    for i, t in enumerate(lines):
        ax.text(cx, top - (i + 1) * gap - 0.04, t, ha='center', va='center', fontsize=lsize, color='#1a1a1a')


def arrow(x0, y0, x1, y1, col, lw=2.2, both=False, ls='-'):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='<|-|>' if both else '-|>', mutation_scale=16,
                                 color=col, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0))


# ---- data
box(0.25, 11.25, 2.65, 12.15, GREY, 'CBIS-DDSM', ['2,408 train / 399 test images', 'pixel masks, 422 test lesions'], tsize=11, lsize=8.8, gap=0.27)
box(2.77, 11.25, 4.45, 12.15, GREY, 'Mini-MIAS', ['322 films, 161 women', '111 lesion-annotated'], tsize=11, lsize=8.8, gap=0.27)
box(4.57, 11.25, 6.93, 12.15, GREY, 'INbreast (external)', ['107 mass images, 116 masses', '67 normal images'], tsize=11, lsize=8.8, gap=0.27)
box(7.05, 11.25, 9.75, 12.15, GREY, 'VinDr-Mammo (external)', ['1,363 lesion images (boxes)', '300 normal images'], tsize=11, lsize=8.8, gap=0.27)
for x in (1.45, 3.61, 5.75, 8.4): arrow(x, 11.25, 5.0 + (x - 5.0) * 0.45, 10.62, GREY[1])
box(0.55, 9.75, 9.45, 10.62, GREY, 'Leakage control',
    ['CBIS: patient-disjoint split (31 shared patients removed)  ·  Mini-MIAS: patient-grouped 5-fold CV',
     'augmentation on training partitions only  ·  no external image used for training or selection'], lsize=9.4, gap=0.27)
arrow(3.6, 9.75, 2.45, 9.17, GREY[1]); arrow(6.4, 9.75, 7.55, 9.17, GREY[1])

# ---- weakly supervised branch
box(0.25, 8.45, 4.65, 9.17, WEAK, 'WEAKLY SUPERVISED', ['image-level labels only'], tsize=13.5, lsize=11)
box(0.25, 6.95, 4.65, 8.05, WEAK, 'EfficientNet-B0 + SE / CBAM (224×224)',
    ['Grad-CAM · Grad-CAM++ · LayerCAM · Score-CAM', 'CBAM spatial map · attention-MIL (4×4 tiles)'], tsize=11, lsize=9.6, bold=False)
box(0.25, 5.35, 4.65, 6.55, WEAK, 'Stronger, higher-resolution classifier',
    ['frozen Mammo-CLIP B5 (control: ImageNet B5)', '380×228 · 760×456 · 1520×912', 'CAM · CBAM · MIL heads, 3 seeds each'], tsize=11, lsize=9.6, bold=False, gap=0.27)
box(0.25, 4.15, 4.65, 4.95, WEAK, 'Dedicated weakly-supervised models',
    ['GMIC · GLAM (public NYU weights, no retraining)'], tsize=11, lsize=9.6, bold=False)
box(0.25, 2.75, 4.65, 3.75, WEAK2, 'Localization scored vs ground truth',
    ['pointing game · IoU / Dice · chance level', 'strata: size · type · classification outcome'], tsize=11.5, lsize=9.6, gap=0.29)
for y0, y1 in ((8.45, 8.05), (6.95, 6.55), (5.35, 4.95), (4.15, 3.75)): arrow(2.45, y0, 2.45, y1, WEAK[1])

# ---- supervised branch
box(5.35, 8.45, 9.75, 9.17, SUP, 'SUPERVISED', ['pixel-level lesion masks'], tsize=13.5, lsize=11)
box(5.35, 6.95, 9.75, 8.05, SUP, 'Full-image detectors (256×256)',
    ['Attention U-Net v1-v3 (v3: 3 seeds)', 'nnU-Net (2D) · SegFormer (MiT-B2)'], tsize=11, lsize=9.6, bold=False)
box(5.35, 5.75, 9.75, 6.55, SUP, 'Lesion-level detection',
    ['sensitivity · FP per image · FROC / CPM'], tsize=11, lsize=9.6, bold=False)
box(5.35, 4.15, 9.75, 5.35, SUP, 'Delineator on 256×256 crops (margin 1.8)',
    ['original (ground-truth-centred training crops)', 'pipeline-aware (detector-error crops, 3 seeds)'], tsize=11, lsize=9.6, bold=False)
box(5.35, 2.75, 7.45, 3.75, SUP2, 'Conditional', ['crop from annotation', '+ centred-disk prior'], tsize=11.5, lsize=9.4, gap=0.27)
box(7.65, 2.75, 9.75, 3.75, SUP2, 'End-to-end', ['crop from detections', 'pasted back, per lesion'], tsize=11.5, lsize=9.4, gap=0.27)
for y0, y1 in ((8.45, 8.05), (6.95, 6.55), (5.75, 5.35)): arrow(7.55, y0, 7.55, y1, SUP[1])
arrow(7.0, 4.15, 6.4, 3.75, SUP[1]); arrow(8.1, 4.15, 8.7, 3.75, SUP[1])

# ---- external validation and statistics
box(0.25, 1.35, 9.75, 2.30, EXT, 'EXTERNAL VALIDATION ON INbreast AND VinDr-Mammo (no retraining)',
    ['detection and FP per normal image  ·  conditional and end-to-end delineation (box level on VinDr-Mammo)  ·  GMIC / GLAM'], tsize=12, lsize=9.2)
arrow(6.4, 2.75, 6.4, 2.30, EXT[1]); arrow(8.7, 2.75, 8.7, 2.30, EXT[1]); arrow(2.45, 2.75, 2.45, 2.30, EXT[1], ls='--')
box(1.4, 0.15, 8.6, 0.95, STAT, 'Paired statistics on identical images',
    ['exact McNemar (per image) · Wilcoxon (per lesion) · patient-clustered bootstrap 95% CIs'], tsize=10.5, lsize=9.2, lw=1.4, gap=0.28)
arrow(5.0, 1.35, 5.0, 0.95, STAT[1], lw=1.6)
fig.savefig(OUT + '.png', dpi=300); fig.savefig(OUT + '.pdf'); print('saved', OUT)
