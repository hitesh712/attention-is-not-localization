"""Shared components for the end-to-end detect-then-delineate evaluation.

Model definition, preprocessing and crop geometry are copied verbatim from the training
notebooks (cbis_attention_unet_v3.ipynb, cbis_roi_seg.ipynb). The ONLY difference between
the conditional (oracle-crop) and end-to-end protocols is where the crop comes from:
the ground-truth lesion box, or the full-image detector's own prediction.
"""
import os
os.environ.setdefault('DML_VISIBLE_DEVICES', '0')     # hide the Intel iGPU; must precede the TF import
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np, cv2
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from scipy.ndimage import maximum_filter

SZ, MARGIN, THR, MASK_THRESH = 256, 1.8, 0.5, 127
MODELS = 'G:/My Drive/VS_CC/models'
W_DET = MODELS + '/cbis_attention_unet_v3.weights.h5'   # full-image detector (v3)
W_ROI = MODELS + '/cbis_roi_seg.weights.h5'             # ROI delineator
FROC_FPI = (0.125, 0.25, 0.5, 1, 2, 4, 8)
_clahe = cv2.createCLAHE(2.0, (8, 8))

for _g in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(_g, True)
    except Exception:
        pass


# ------------------------------------------------------------------ model (as trained)
def _conv_block(x, f):
    x = layers.Conv2D(f, 3, padding='same')(x); x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
    x = layers.Conv2D(f, 3, padding='same')(x); x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
    return x


def _attn_gate(skip, gating, inter):
    tx = layers.Conv2D(inter, 1)(skip); pg = layers.Conv2D(inter, 1)(gating)
    r = layers.Activation('relu')(layers.Add()([tx, pg]))
    return layers.Multiply()([skip, layers.Conv2D(1, 1, activation='sigmoid')(r)])


_SKIPS = ['block_1_expand_relu', 'block_3_expand_relu', 'block_6_expand_relu',
          'block_13_expand_relu', 'block_16_project']


def build_model(weights_path, sz=SZ):
    inp = layers.Input((sz, sz, 3)); x = layers.Rescaling(1. / 127.5, offset=-1)(inp)
    base = MobileNetV2(input_tensor=x, include_top=False, weights=None)   # all weights come from the file
    s128, s64, s32, s16, s8 = [base.get_layer(n).output for n in _SKIPS]

    def up(x, skip, f):
        x = layers.UpSampling2D()(x)
        return _conv_block(layers.Concatenate()([x, _attn_gate(skip, x, f)]), f)

    d = up(s8, s16, 256); d = up(d, s32, 128); d = up(d, s64, 64); d = up(d, s128, 32)
    d = _conv_block(layers.UpSampling2D()(d), 16)
    m = Model(inp, layers.Conv2D(1, 1, activation='sigmoid')(d))
    m.load_weights(weights_path)
    return m


def predict(model, X, batch=8):
    out = [model(np.asarray(X[b:b + batch], np.float32), training=False).numpy()[..., 0]
           for b in range(0, len(X), batch)]
    return np.concatenate(out) if out else np.zeros((0, SZ, SZ), np.float32)


# ------------------------------------------------------------------ preprocessing (as trained)
def full_input(gray):
    """== load_img() in cbis_attention_unet_v3.ipynb: whole mammogram -> 256x256 -> CLAHE -> 3 channels."""
    return np.stack([_clahe.apply(cv2.resize(gray, (SZ, SZ)))] * 3, -1)


def crop_window(H, W, r0, r1, c0, c1, margin=MARGIN):
    """== crop_around() geometry in cbis_roi_seg.ipynb: centred on the box, side = margin x longest side (+16 px)."""
    r0, r1, c0, c1 = int(r0), int(r1), int(c0), int(c1)
    cy, cx = (r0 + r1) // 2, (c0 + c1) // 2
    half = int(max(r1 - r0, c1 - c0) * margin / 2) + 8
    return max(0, cy - half), min(H, cy + half), max(0, cx - half), min(W, cx + half)


def crop_input(gray, win):
    ri0, ri1, ci0, ci1 = win
    return np.stack([_clahe.apply(cv2.resize(gray[ri0:ri1, ci0:ci1], (SZ, SZ)))] * 3, -1)


def crop_mask(mask, win):
    """Binary full-resolution mask -> its 256x256 view of `win` (nearest neighbour, as for the GT crops)."""
    ri0, ri1, ci0, ci1 = win
    cm = mask[ri0:ri1, ci0:ci1].astype(np.uint8)
    return cv2.resize(cm * 255, (SZ, SZ), interpolation=cv2.INTER_NEAREST) >= 127


def mask_bbox(mask):
    ys, xs = np.where(mask)
    return (ys.min(), ys.max(), xs.min(), xs.max()) if len(ys) else None


def is_mask_like(g):
    return g is not None and np.mean((g <= 10) | (g >= 245)) > 0.90 and len(np.unique(g)) <= 64


# ------------------------------------------------------------------ detection from the full-image output
def candidates(p, thr=THR):
    """Detections = connected components of the detector's own binary output (p >= thr), each scored by its
    peak probability. If no pixel reaches thr, the half-maximum region around the global peak is used as a
    single low-confidence detection, so every image yields at least one crop. Sorted by score, uncapped."""
    n, lab = cv2.connectedComponents((p >= thr).astype(np.uint8), connectivity=8)
    out = []
    for k in range(1, n):
        ys, xs = np.where(lab == k)
        j = int(np.argmax(p[ys, xs]))
        out.append(dict(score=float(p[ys[j], xs[j]]), peak=(int(ys[j]), int(xs[j])), area=len(ys), fallback=False,
                        bbox=(int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))))
    if not out:
        y, x = np.unravel_index(int(np.argmax(p)), p.shape)
        _, lab = cv2.connectedComponents((p >= 0.5 * p[y, x]).astype(np.uint8), connectivity=8)
        ys, xs = np.where(lab == lab[y, x])
        out.append(dict(score=float(p[y, x]), peak=(int(y), int(x)), area=len(ys), fallback=True,
                        bbox=(int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))))
    out.sort(key=lambda d: -d['score'])
    return out


def grid_to_full_bbox(b, H, W):
    y0, y1, x0, x1 = b
    return (int(np.floor(y0 * H / SZ)), min(H - 1, int(np.ceil((y1 + 1) * H / SZ)) - 1),
            int(np.floor(x0 * W / SZ)), min(W - 1, int(np.ceil((x1 + 1) * W / SZ)) - 1))


def grid_to_full_pt(y, x, H, W):
    return min(H - 1, int((y + 0.5) * H / SZ)), min(W - 1, int((x + 0.5) * W / SZ))


def paste(canvas, prob, win, thr=THR):
    """Resize a 256x256 crop probability back to its window and OR its binary into `canvas`.
    Returns the crop's peak (value, (row, col)) in full-resolution coordinates."""
    ri0, ri1, ci0, ci1 = win
    h, w = ri1 - ri0, ci1 - ci0
    canvas[ri0:ri1, ci0:ci1] |= cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR) >= thr
    yy, xx = np.unravel_index(int(np.argmax(prob)), prob.shape)
    return float(prob[yy, xx]), (ri0 + min(h - 1, int((yy + 0.5) * h / SZ)),
                                 ci0 + min(w - 1, int((xx + 0.5) * w / SZ)))


def window_binary(prob, win, thr=THR):
    """A single crop's prediction pasted into an otherwise empty canvas, viewed through the same window."""
    ri0, ri1, ci0, ci1 = win
    h, w = ri1 - ri0, ci1 - ci0
    return crop_mask(cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR) >= thr, (0, h, 0, w))


# ------------------------------------------------------------------ metrics
def dice_iou(gt, pm):
    gt = np.asarray(gt, bool); pm = np.asarray(pm, bool)
    inter = np.count_nonzero(gt & pm); s = np.count_nonzero(gt) + np.count_nonzero(pm); u = s - inter
    return (2 * inter / s if s else 1.0), (inter / u if u else 1.0)


def local_peaks(p, size=9, floor=0.05, kmax=20):
    """Local maxima of the probability map (9x9 non-maximum suppression), for FROC analysis."""
    yx = np.argwhere((p >= maximum_filter(p, size=size, mode='constant', cval=0)) & (p >= floor))
    v = p[yx[:, 0], yx[:, 1]]
    o = np.argsort(-v, kind='stable')[:kmax]
    return yx[o], v[o]


def match_points(points, scores, lesion_masks):
    """Greedy pointing-criterion matching by score: a point inside an unmatched lesion is a TP, inside an
    already-matched lesion it is a duplicate (ignored), anywhere else it is a FP."""
    hit = [False] * len(lesion_masks); rec = []
    for (r, c), s in sorted(zip(points, scores), key=lambda z: -z[1]):
        inside = [k for k, m in enumerate(lesion_masks) if m[r, c]]
        if not inside:
            rec.append((float(s), False))
        else:
            free = [k for k in inside if not hit[k]]
            if free:
                hit[free[0]] = True; rec.append((float(s), True))
    return rec, hit


SWEEP = tuple(np.round(np.arange(0.05, 0.96, 0.05), 2))


def sweep_counts(p, H, W, lesion_masks):
    """FROC by sweeping the detector's own binarisation threshold t: detections = connected components of
    p >= t (peak = scored point), matched with the pointing criterion. Returns {t: (tp, fp)}; t = 0.5 is the
    pipeline's operating point, so it lies exactly on this curve."""
    out = {}
    for t in SWEEP:
        n, lab = cv2.connectedComponents((p >= t).astype(np.uint8), connectivity=8)
        pts, sc = [], []
        for k in range(1, n):
            ys, xs = np.where(lab == k); j = int(np.argmax(p[ys, xs]))
            pts.append(grid_to_full_pt(ys[j], xs[j], H, W)); sc.append(float(p[ys[j], xs[j]]))
        rec, _ = match_points(pts, sc, lesion_masks)
        out[t] = (sum(r[1] for r in rec), sum(not r[1] for r in rec))
    return out


def sweep_curve(tp_fp, n_lesions, n_images):
    """(fp per image, sensitivity) for every swept threshold, plus CPM by step interpolation."""
    ts = sorted(tp_fp); fpi = np.array([tp_fp[t][1] / n_images for t in ts]); sen = np.array([tp_fp[t][0] / n_lesions for t in ts])
    at = {f: float(sen[fpi <= f].max()) if np.any(fpi <= f) else 0.0 for f in FROC_FPI}
    return ts, fpi, sen, at, float(np.mean(list(at.values())))


def froc(scores, is_tp, n_lesions, n_images):
    s = np.asarray(scores, float); t = np.asarray(is_tp, bool)
    t = t[np.argsort(-s, kind='stable')]
    sens = np.cumsum(t) / max(n_lesions, 1); fpi = np.cumsum(~t) / max(n_images, 1)
    at = {f: float(sens[fpi <= f].max()) if np.any(fpi <= f) else 0.0 for f in FROC_FPI}
    return at, float(np.mean(list(at.values()))), fpi, sens
