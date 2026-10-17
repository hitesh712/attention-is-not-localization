"""Re-score the weakly-supervised CBIS models image-by-image (paired with the supervised detector).

Rebuilds the three saved weakly-supervised models exactly as trained (frozen EfficientNetB0+CBAM;
fine-tuned EfficientNetB0+CBAM; gated-attention MIL), reproduces the published numbers, and records
per-image outcomes on the 399 masked CBIS test mammograms:
  * pointing hit against the full-resolution union mask (paired with the supervised detector)
  * classifier probability (correct vs incorrect stratification, Reviewer 2 #6)
  * lesion size (size stratification, Reviewer 1)
Adds LayerCAM and Score-CAM for the fine-tuned model.
"""
import os, sys, json, time, gc
os.environ.setdefault('DML_VISIBLE_DEVICES', '0'); os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2, tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from sklearn.metrics import roc_auc_score
import cbis_data as D

for g in tf.config.list_physical_devices('GPU'):
    try: tf.config.experimental.set_memory_growth(g, True)
    except Exception: pass
MODELS = 'G:/My Drive/VS_CC/models'; OUT = MODELS + '/revision'; os.makedirs(OUT, exist_ok=True)
S = 224; GRID, TILE = 4, 224; CANVAS = GRID * TILE
_clahe = cv2.createCLAHE(2.0, (8, 8)); T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


# ------------------------------------------------------------------ architectures (verbatim)
def cbam_block(x, ratio=16, name='cbam'):
    f = x.shape[-1]
    s1 = layers.Dense(max(1, f // ratio), activation='relu', kernel_initializer='he_normal')
    s2 = layers.Dense(f, kernel_initializer='he_normal')
    avg = layers.Reshape((1, 1, f))(layers.GlobalAveragePooling2D()(x))
    mx = layers.Reshape((1, 1, f))(layers.GlobalMaxPooling2D()(x))
    x = layers.Multiply()([x, layers.Activation('sigmoid')(layers.Add()([s2(s1(avg)), s2(s1(mx))]))])
    ap = layers.Lambda(lambda t: tf.reduce_mean(t, -1, keepdims=True))(x)
    mp = layers.Lambda(lambda t: tf.reduce_max(t, -1, keepdims=True))(x)
    sp = layers.Conv2D(1, 7, padding='same', activation='sigmoid', name=name + '_spatial')(layers.Concatenate()([ap, mp]))
    return layers.Multiply()([x, sp])


# NOTE weights='imagenet' is required even though every weight is then overwritten: with 'imagenet', Keras inserts an
# extra input Rescaling(1/sqrt(std)) layer (TF issue #49930) that weights=None omits, which changes the outputs.
def frozen_model():        # cbis_attention_localization.ipynb, ATTENTION='cbam', TAP=block6a_activation
    base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(S, S, 3))
    x = layers.Activation('linear', name='loc_features')(cbam_block(base.get_layer('block6a_activation').output))
    m = Model(base.input, layers.Dense(1, activation='sigmoid')(layers.Dropout(0.5)(layers.GlobalAveragePooling2D(name='gap')(x))))
    m.load_weights(MODELS + '/efficientnet_cbam_cbis.weights.h5'); return m


def finetuned_model():     # cbis_localization_finetuned.ipynb (CBAM on top_activation; CAM at block5c_add)
    base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(S, S, 3))
    x = layers.Activation('linear', name='loc_features')(cbam_block(base.output))
    m = Model(base.input, layers.Dense(1, activation='sigmoid')(layers.Dropout(0.5)(layers.GlobalAveragePooling2D(name='gap')(x))))
    m.load_weights(MODELS + '/efficientnet_cbam_cbis_finetuned.weights.h5'); return m


def mil_model(feat_dim=1280, k=GRID * GRID, hid=128, att=64):   # cbis_attention_mil.ipynb
    inp = layers.Input((k, feat_dim))
    h = layers.TimeDistributed(layers.Dense(hid, activation='relu'))(inp)
    h = layers.TimeDistributed(layers.Dropout(0.3))(h)
    V = layers.TimeDistributed(layers.Dense(att, activation='tanh'))(h)
    U = layers.TimeDistributed(layers.Dense(att, activation='sigmoid'))(h)
    s = layers.Flatten()(layers.TimeDistributed(layers.Dense(1))(layers.Multiply()([V, U])))
    a = layers.Softmax(name='attention')(s)
    m = Model(inp, layers.Dense(1, activation='sigmoid')(layers.Dot(axes=1)([a, h])))
    m.load_weights(MODELS + '/cbis_mil.weights.h5')
    return m, Model(inp, m.get_layer('attention').output)


def inp224(gray):
    return np.stack([_clahe.apply(cv2.resize(gray, (S, S)))] * 3, -1)


def tiles(gray):
    img = _clahe.apply(cv2.resize(gray, (CANVAS, CANVAS)))
    return np.stack([np.stack([img[i * TILE:(i + 1) * TILE, j * TILE:(j + 1) * TILE]] * 3, -1)
                     for i in range(GRID) for j in range(GRID)])


# ------------------------------------------------------------------ CAMs (Grad-CAM/++ verbatim; LayerCAM, Score-CAM added)
def norm(c):
    c = np.maximum(np.asarray(c, np.float32), 0); return c / (c.max() + 1e-8)


def grad_cams(cam_model, x):
    with tf.GradientTape() as t:
        conv, pred = cam_model(x); y = pred[:, 0]
    g = t.gradient(y, conv)
    conv0, g0 = conv[0].numpy(), g[0].numpy()
    gc_ = norm(np.sum(conv0 * g.numpy().mean(axis=(0, 1, 2)), -1))
    g2, g3 = g0 ** 2, g0 ** 3
    den = 2 * g2 + conv0.sum(axis=(0, 1)) * g3
    alpha = g2 / np.where(den != 0, den, 1.0)
    gcpp = norm(np.sum(conv0 * np.sum(alpha * np.maximum(g0, 0), axis=(0, 1)), -1))
    lcam = norm(np.sum(np.maximum(g0, 0) * conv0, -1))
    return gc_, gcpp, lcam, conv0, float(pred[0, 0])


def score_cam(model, x, conv0, batch=16):
    A = np.stack([cv2.resize(conv0[..., k], (S, S)) for k in range(conv0.shape[-1])])
    lo = A.min(axis=(1, 2), keepdims=True); hi = A.max(axis=(1, 2), keepdims=True)
    M = ((A - lo) / np.where(hi - lo > 0, hi - lo, 1.0)).astype(np.float32)
    p = np.concatenate([model(x.astype(np.float32) * M[b:b + batch, ..., None], training=False).numpy()[:, 0]
                        for b in range(0, len(M), batch)])
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    w = np.exp(z - z.max()); w /= w.sum()
    return norm(np.sum(conv0 * w, -1))


def score_map(heat, gt224, gtF):
    """Published metrics at 224 (peak; IoU/Dice of the >= half-max region) + pointing at full resolution, plus
    the IoU/Dice threshold sensitivity (>= 25% and >= 75% of the maximum; Reviewer 2 #8)."""
    h = cv2.resize(heat.astype(np.float32), (S, S))
    py, px = np.unravel_index(int(np.argmax(h)), h.shape)
    H, W = gtF.shape
    out = dict(pg224=int(gt224[py, px]), pgF=int(gtF[min(H - 1, int((py + 0.5) * H / S)), min(W - 1, int((px + 0.5) * W / S))]))
    for tag, q in (('', 0.5), ('25', 0.25), ('75', 0.75)):
        pm = h >= q * h.max(); inter = np.count_nonzero(pm & gt224); u = np.count_nonzero(pm | gt224)
        s = np.count_nonzero(pm) + np.count_nonzero(gt224)
        out['iou' + tag] = inter / u if u else 0.0; out['dice' + tag] = 2 * inter / s if s else 0.0
    return out


# ------------------------------------------------------------------ run
imgs = D.test_images(); rows = D.test_rows()
log(f'test images {len(imgs)} | test rows {len(rows)}')
fz, ft = frozen_model(), finetuned_model()
fz_cam = Model(fz.inputs, [fz.get_layer('loc_features').output, fz.output])
ft_cam = Model(ft.inputs, [ft.get_layer('block5c_add').output, ft.output])
fz_sp = Model(fz.inputs, fz.get_layer('cbam_spatial').output)
ft_sp = Model(ft.inputs, ft.get_layer([l.name for l in ft.layers if l.name.endswith('_spatial')][0]).output)
_b = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(TILE, TILE, 3))
feat = Model(_b.input, layers.GlobalAveragePooling2D()(_b.output))
mil, mil_att = mil_model()
log('models loaded')

METHODS = ['fz_gradcam', 'fz_cbam', 'ft_gradcam', 'ft_gradcampp', 'ft_layercam', 'ft_scorecam', 'ft_cbam']
R = {m: {k: [] for k in ('pg224', 'iou', 'dice', 'pgF', 'iou25', 'dice25', 'iou75', 'dice75')} for m in METHODS}
R['mil'] = {k: [] for k in ('pg_tile', 'iou_tile', 'tile_auc', 'pg_tileF')}
INFO = {k: [] for k in ('key', 'patient', 'label', 'p_fz', 'p_ft', 'p_mil', 'gt_frac', 'max_side', 'n_lesions')}
key2idx = {}; MILROW = {}

for i, (key, fp, mps, pid, lab) in enumerate(imgs):
    gray = cv2.imread(fp, cv2.IMREAD_GRAYSCALE); H, W = gray.shape
    gtF = np.zeros((H, W), bool); gt224 = np.zeros((S, S), bool); gg = np.zeros((GRID, GRID), int); rowg = {}; nles = 0
    for mp in mps:                                         # each mask read once; same rules as the notebooks
        raw = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if not D.is_mask_like(raw): continue
        gt224 |= cv2.resize(raw, (S, S), interpolation=cv2.INTER_NEAREST) >= D.MASK_THRESH          # load_mask()
        g_ = ((cv2.resize(raw, (CANVAS, CANVAS), interpolation=cv2.INTER_NEAREST) >= D.MASK_THRESH)
              .reshape(GRID, TILE, GRID, TILE).sum(axis=(1, 3)) > 0).astype(int)                     # mask_tile_grid()
        rowg[mp] = g_; gg |= g_
        mF = (raw if raw.shape == (H, W) else cv2.resize(raw, (W, H), interpolation=cv2.INTER_NEAREST)) >= D.MASK_THRESH
        if mF.any(): gtF |= mF; nles += 1
    x = inp224(gray)[None]
    gc0, _, _, _, p_fz = grad_cams(fz_cam, x)
    maps = {'fz_gradcam': gc0, 'fz_cbam': fz_sp(x, training=False).numpy()[0, :, :, 0]}
    gc1, gcpp1, lc1, conv1, p_ft = grad_cams(ft_cam, x)
    maps.update(ft_gradcam=gc1, ft_gradcampp=gcpp1, ft_layercam=lc1, ft_scorecam=score_cam(ft, x, conv1),
                ft_cbam=ft_sp(x, training=False).numpy()[0, :, :, 0])
    for m in METHODS:
        for k, v in score_map(maps[m], gt224, gtF).items(): R[m][k].append(v)

    f = feat(tiles(gray).astype(np.float32), training=False).numpy()[None]
    a = mil_att(f, training=False).numpy()[0].reshape(GRID, GRID); p_mil = float(mil(f, training=False).numpy()[0, 0])
    top = np.unravel_index(int(np.argmax(a)), a.shape)
    for mp, g_ in rowg.items():                            # per-ROI scoring, as published (422 rows)
        if g_.sum() == 0: continue
        pr = (a >= 0.5 * a.max()).astype(int); it = (pr & g_).sum(); un = (pr | g_).sum()
        MILROW[mp] = (int(g_[top] == 1), roc_auc_score(g_.ravel(), a.ravel()) if g_.sum() < g_.size else np.nan,
                      it / un if un else 0.0)
    pm = (a >= 0.5 * a.max()).astype(int); inter = (pm & gg).sum(); u = (pm | gg).sum()
    r0, r1 = int(top[0] * H / GRID), int((top[0] + 1) * H / GRID); c0, c1 = int(top[1] * W / GRID), int((top[1] + 1) * W / GRID)
    R['mil']['pg_tile'].append(int(gg[top] == 1)); R['mil']['iou_tile'].append(inter / u if u else 0.0)
    R['mil']['tile_auc'].append(roc_auc_score(gg.ravel(), a.ravel()) if 0 < gg.sum() < gg.size else np.nan)
    R['mil']['pg_tileF'].append(int(gtF[r0:r1, c0:c1].any()))

    ys, xs = np.where(gtF)
    side = max(ys.max() - ys.min(), xs.max() - xs.min()) + 1 if len(ys) else 0
    for k, v in (('key', key), ('patient', pid), ('label', lab), ('p_fz', p_fz), ('p_ft', p_ft), ('p_mil', p_mil),
                 ('gt_frac', gtF.mean()), ('max_side', side), ('n_lesions', nles)):
        INFO[k].append(v)
    key2idx[key] = i
    del gray, gtF; gc.collect()
    if (i + 1) % 50 == 0 or i + 1 == len(imgs):
        log(f'{i + 1}/{len(imgs)}  ft Grad-CAM PG224 {np.mean(R["ft_gradcam"]["pg224"]) * 100:.1f}% | '
            f'ft Score-CAM PGfull {np.mean(R["ft_scorecam"]["pgF"]) * 100:.1f}% | MIL tile-PG {np.mean(R["mil"]["pg_tile"]) * 100:.1f}%')

# ------------------------------------------------------------------ reproduce the published row-level AUCs (422 rows)
rk = rows['image_key'].map(key2idx).values; yr = rows['label'].values
auc = {n: roc_auc_score(yr, np.asarray(INFO[c])[rk]) for n, c in (('frozen', 'p_fz'), ('finetuned', 'p_ft'), ('mil', 'p_mil'))}
mr = np.array([MILROW[mp] for mp in rows['mask_path'] if mp in MILROW], float)
auc['mil_rows_pg_tileauc_iou'] = [len(mr), float(mr[:, 0].mean()), float(np.nanmean(mr[:, 1])), float(mr[:, 2].mean())]
np.savez_compressed(OUT + '/ws_reeval.npz', **{f'{m}_{k}': np.asarray(v) for m in R for k, v in R[m].items()},
                    **{'info_' + k: np.asarray(v) for k, v in INFO.items()})
summary = dict(n_images=len(imgs), row_auc=auc,
               published='frozen AUC 0.556 GC PG 0.5% IoU .006 Dice .012 | FT AUC 0.615 GC 2.3% IoU .007, GC++ 1.0% IoU .006 Dice .010 | MIL AUC 0.544',
               **{m: {k: float(np.nanmean(v)) for k, v in R[m].items()} for m in R})
json.dump(summary, open(OUT + '/ws_reeval.json', 'w'), indent=2)
log('SUMMARY\n' + json.dumps(summary, indent=2))
