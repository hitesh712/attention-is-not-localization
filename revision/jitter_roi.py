"""Pipeline-aware ROI delineator: train on crops that look like the detector's crops.

The published delineator only ever saw ground-truth-centred crops, so it degrades on detector-generated crops
(INbreast end-to-end Dice 0.555 vs detector-alone 0.583). Here:
  0. the full-image detector (v3) is run on its own held-out VALIDATION patients (never trained on, never test)
     to measure how its crops err: centre offset and scale relative to the matched lesion, and the share of
     detections that hit no lesion;
  1. a crop pool is built from the training lesions: the GT-centred crop + 3 crops whose offset/scale are
     resampled from those measured errors, plus lesion-free crops (empty target) from the same breasts;
  2. the delineator is retrained with the published recipe (MobileNetV2 Attention U-Net, two-phase, BCE +
     Focal-Tversky, flips/rotations), sampling one variant per lesion per epoch plus negatives at the measured
     false-positive share; early stopping on validation patients.
Usage:  python jitter_roi.py [--seed 2021]
"""
import os, sys, json, time, gc, argparse
os.environ.setdefault('DML_VISIBLE_DEVICES', '0'); os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, 'G:/My Drive/VS_CC/e2e')
import numpy as np, cv2, tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import GroupShuffleSplit
import cbis_data as D
from e2e_common import (build_model as build_frozen, predict, candidates, crop_window, crop_input, crop_mask,
                        mask_bbox, W_DET, SZ, THR)

ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, default=2021); args = ap.parse_args()
CACHE = D.CACHE; MODELS = 'G:/My Drive/VS_CC/models'; OUT = MODELS + '/revision'; os.makedirs(OUT, exist_ok=True)
POOL = CACHE + '/jit'; N_JIT = 3; BATCH = 4; RS = 2021
T0 = time.time(); log = lambda *a: print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


# ------------------------------------------------------------------ 0. detector error distribution on validation patients
def detector_errors():
    path = OUT + '/detector_val_errors.json'
    if os.path.exists(path): return json.load(open(path))
    # the v3 notebook's own train/validation split (same items, same GroupShuffleSplit)
    tr = D.train_rows(); d = tr[tr['mask_path'].notna()]
    items, pats = [], []
    for key, rows in d.groupby('image_key'):
        items.append((rows.iloc[0]['full_path'], list(rows['mask_path']))); pats.append(rows.iloc[0]['patient_id'])
    Xtr = np.load(CACHE + '/tr_X_256.npy', mmap_mode='r'); Ytr = np.load(CACHE + '/tr_Y_256.npy', mmap_mode='r')
    assert len(items) == len(Xtr)
    _, va = next(GroupShuffleSplit(1, test_size=0.15, random_state=RS).split(np.arange(len(items)), groups=np.array(pats)))
    det = build_frozen(W_DET); off, scl = [], []; n_det = n_fp = 0
    for b in range(0, len(va), 8):
        idx = np.sort(va[b:b + 8]); P = predict(det, Xtr[idx])
        for k, i in enumerate(idx):
            gt = np.asarray(Ytr[i])[..., 0].astype(np.uint8)
            n, lab = cv2.connectedComponents(gt, connectivity=8)
            gtc = [lab == q for q in range(1, n)]
            for c in candidates(P[k]):
                if c['fallback']: continue
                n_det += 1; y0, y1, x0, x1 = c['bbox']; box = np.zeros_like(gt, bool); box[y0:y1 + 1, x0:x1 + 1] = True
                ov = [np.count_nonzero(box & g) for g in gtc]
                if not ov or max(ov) == 0: n_fp += 1; continue
                g = gtc[int(np.argmax(ov))]; r0, r1, c0, c1 = mask_bbox(g)
                L = max(r1 - r0, c1 - c0) + 1; Lp = max(y1 - y0, x1 - x0) + 1
                off.append(((y0 + y1) / 2 - (r0 + r1) / 2) / L); off.append(((x0 + x1) / 2 - (c0 + c1) / 2) / L)
                scl.append(Lp / L)
    del det; gc.collect(); tf.keras.backend.clear_session()
    o = np.array(off).reshape(-1, 2); s = np.array(scl)
    E = dict(n_val_images=int(len(va)), n_detections=n_det, fp_share=n_fp / max(n_det, 1), offsets=o.tolist(), scales=s.tolist(),
             summary=dict(abs_offset_median=float(np.median(np.abs(o))), abs_offset_p90=float(np.quantile(np.abs(o), 0.9)),
                          scale_median=float(np.median(s)), scale_p10=float(np.quantile(s, 0.1)), scale_p90=float(np.quantile(s, 0.9))))
    json.dump(E, open(path, 'w'))
    return E


# ------------------------------------------------------------------ 1. crop pool
def build_pool(E):
    if os.path.exists(POOL + '_meta.json'): return json.load(open(POOL + '_meta.json'))
    rng = np.random.RandomState(RS); off = np.array(E['offsets']); scl = np.array(E['scales'])
    tr = D.train_rows(); tr = tr[tr['mask_path'].notna()]
    groups = list(tr.groupby('image_key'))
    n_max = len(tr) * (1 + N_JIT) + len(groups)
    X = np.lib.format.open_memmap(POOL + '_X.npy', mode='w+', dtype=np.uint8, shape=(n_max, SZ, SZ, 3))
    Y = np.lib.format.open_memmap(POOL + '_Y.npy', mode='w+', dtype=np.uint8, shape=(n_max, SZ, SZ, 1))
    meta = []; n = 0; sides = []
    for gi, (key, rows) in enumerate(groups):
        gray = cv2.imread(rows.iloc[0]['full_path'], cv2.IMREAD_GRAYSCALE); H, W = gray.shape
        masks = [m for m in (D.read_mask(mp, (H, W)) for mp in rows['mask_path']) if m is not None and m.any()]
        if not masks: continue
        union = np.logical_or.reduce(masks); pid = rows.iloc[0]['patient_id']
        for li, m in enumerate(masks):
            r0, r1, c0, c1 = mask_bbox(m); L = max(r1 - r0, c1 - c0) + 1; cy, cx = (r0 + r1) / 2, (c0 + c1) / 2; sides.append(L)
            for v in range(1 + N_JIT):
                if v == 0: win = crop_window(H, W, r0, r1, c0, c1)                      # the published GT-centred crop
                else:
                    k = rng.randint(len(scl)); dy, dx = off[k] + rng.normal(0, 0.03, 2)     # joint (offset, scale) draw
                    Lp = L * scl[k] * np.exp(rng.normal(0, 0.05)); y, x = cy + dy * L, cx + dx * L
                    win = crop_window(H, W, y - Lp / 2, y + Lp / 2, x - Lp / 2, x + Lp / 2)
                if win[1] - win[0] < 4 or win[3] - win[2] < 4: continue
                X[n] = crop_input(gray, win); Y[n] = crop_mask(union, win)[..., None]
                meta.append(dict(i=n, key=key, patient=pid, lesion=f'{key}#{li}', kind='centred' if v == 0 else 'jitter')); n += 1
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)       # one lesion-free crop per image
        ys, xs = np.where(th > 0)
        for _ in range(30):
            L = sides[rng.randint(len(sides))] * scl[rng.randint(len(scl))]; j = rng.randint(len(ys))
            win = crop_window(H, W, ys[j] - L / 2, ys[j] + L / 2, xs[j] - L / 2, xs[j] + L / 2)
            if win[1] - win[0] < 4 or win[3] - win[2] < 4 or union[win[0]:win[1], win[2]:win[3]].any(): continue
            X[n] = crop_input(gray, win); Y[n] = 0
            meta.append(dict(i=n, key=key, patient=pid, lesion=None, kind='negative')); n += 1; break
        if (gi + 1) % 300 == 0: log(f'pool: {gi + 1}/{len(groups)} images, {n} crops')
    X.flush(); Y.flush(); del X, Y
    json.dump(meta, open(POOL + '_meta.json', 'w'))
    return meta


# ------------------------------------------------------------------ 2. model + training (published recipe)
def conv_block(x, f):
    for _ in range(2): x = layers.Activation('relu')(layers.BatchNormalization()(layers.Conv2D(f, 3, padding='same')(x)))
    return x


def attn_gate(skip, gating, inter):
    r = layers.Activation('relu')(layers.Add()([layers.Conv2D(inter, 1)(skip), layers.Conv2D(inter, 1)(gating)]))
    return layers.Multiply()([skip, layers.Conv2D(1, 1, activation='sigmoid')(r)])


def build_trainable():
    inp = layers.Input((SZ, SZ, 3)); x = layers.Rescaling(1. / 127.5, offset=-1)(inp)
    base = MobileNetV2(input_tensor=x, include_top=False, weights='imagenet'); base.trainable = False
    s128, s64, s32, s16, s8 = [base.get_layer(n).output for n in ['block_1_expand_relu', 'block_3_expand_relu',
                                                                  'block_6_expand_relu', 'block_13_expand_relu', 'block_16_project']]
    def up(x, skip, f):
        x = layers.UpSampling2D()(x); return conv_block(layers.Concatenate()([x, attn_gate(skip, x, f)]), f)
    d = up(s8, s16, 256); d = up(d, s32, 128); d = up(d, s64, 64); d = up(d, s128, 32)
    return Model(inp, layers.Conv2D(1, 1, activation='sigmoid')(conv_block(layers.UpSampling2D()(d), 16))), base


def _flat(a, b): return tf.reshape(tf.cast(a, tf.float32), [-1]), tf.reshape(b, [-1])
def tversky(yt, yp, a=0.7, b=0.3, s=1.):
    t, p = _flat(yt, yp); tp = tf.reduce_sum(t * p)
    return (tp + s) / (tp + a * tf.reduce_sum(t * (1 - p)) + b * tf.reduce_sum((1 - t) * p) + s)
def loss_fn(yt, yp):
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.cast(yt, tf.float32), yp)) + tf.pow(1 - tversky(yt, yp), 0.75)
def dice_coef(yt, yp, s=1.):
    t, p = _flat(yt, yp); i = tf.reduce_sum(t * p); return (2 * i + s) / (tf.reduce_sum(t) + tf.reduce_sum(p) + s)


class PoolSeq(tf.keras.utils.Sequence):
    """Each epoch: one random variant per lesion + negatives at the detector's false-positive share."""
    def __init__(s, X, Y, by_lesion, negatives, neg_share, augment, seed, fixed=False):
        s.X, s.Y, s.L, s.N, s.q, s.aug, s.rng, s.fixed = X, Y, by_lesion, negatives, neg_share, augment, np.random.RandomState(seed), fixed
        s.on_epoch_end()
    def on_epoch_end(s):
        if s.fixed: s.idx = np.array(sorted(i for v in s.L.values() for i in v) + list(s.N)); return
        pos = [v[s.rng.randint(len(v))] for v in s.L.values()]
        neg = list(s.rng.choice(s.N, min(len(s.N), int(round(s.q / max(1 - s.q, 1e-6) * len(pos)))), replace=False)) if len(s.N) else []
        s.idx = np.array(pos + neg); s.rng.shuffle(s.idx)
    def __len__(s): return int(np.ceil(len(s.idx) / BATCH))
    def __getitem__(s, b):
        sel = np.sort(s.idx[b * BATCH:(b + 1) * BATCH]); X = s.X[sel].astype('float32'); Y = s.Y[sel].astype('float32')
        if s.aug:
            for k in range(len(X)):
                if s.rng.rand() < 0.5: X[k] = X[k, :, ::-1]; Y[k] = Y[k, :, ::-1]
                if s.rng.rand() < 0.5: X[k] = X[k, ::-1]; Y[k] = Y[k, ::-1]
                r = s.rng.randint(4)
                if r: X[k] = np.rot90(X[k], r); Y[k] = np.rot90(Y[k], r)
        return X, Y


def main():
    E = detector_errors()
    log('detector errors on validation patients: ' + json.dumps(E['summary']) + f' | FP share {E["fp_share"]:.2f} '
        f'({E["n_detections"]} detections, {E["n_val_images"]} images)')
    meta = build_pool(E)
    X = np.load(POOL + '_X.npy', mmap_mode='r'); Y = np.load(POOL + '_Y.npy', mmap_mode='r')
    pats = np.array([m['patient'] for m in meta])
    tr_p, va_p = next(GroupShuffleSplit(1, test_size=0.15, random_state=RS).split(pats, groups=pats))
    va_pat = set(pats[va_p])
    def split(in_val):
        L, N = {}, []
        for m in meta:
            if (m['patient'] in va_pat) != in_val: continue
            if m['kind'] == 'negative': N.append(m['i'])
            else: L.setdefault(m['lesion'], []).append(m['i'])
        return L, np.array(N, int)
    Ltr, Ntr = split(False); Lva, Nva = split(True)
    log(f'pool {len(meta)} crops | train lesions {len(Ltr)} (+{len(Ntr)} negatives) | val lesions {len(Lva)} (+{len(Nva)} negatives)')
    q = float(np.clip(E['fp_share'], 0.1, 0.5))
    tf.keras.utils.set_random_seed(args.seed)
    gen_tr = PoolSeq(X, Y, Ltr, Ntr, q, True, args.seed); gen_va = PoolSeq(X, Y, Lva, Nva, q, False, RS, fixed=True)
    model, base = build_trainable()
    model.compile(Adam(1e-3), loss=loss_fn, metrics=[dice_coef])
    model.fit(gen_tr, validation_data=gen_va, epochs=12, verbose=2,
              callbacks=[EarlyStopping(monitor='val_dice_coef', mode='max', patience=6, restore_best_weights=True)])
    base.trainable = True
    for l in base.layers:
        if isinstance(l, layers.BatchNormalization): l.trainable = False
    model.compile(Adam(1e-4), loss=loss_fn, metrics=[dice_coef])
    model.fit(gen_tr, validation_data=gen_va, epochs=40, verbose=2,
              callbacks=[EarlyStopping(monitor='val_dice_coef', mode='max', patience=12, restore_best_weights=True),
                         ReduceLROnPlateau(monitor='val_dice_coef', mode='max', factor=0.5, patience=5, min_lr=1e-6)])
    w = MODELS + f'/cbis_roi_seg_jitter_s{args.seed}.weights.h5'; model.save_weights(w)
    # oracle (published) protocol on the CBIS test crops, for comparison with 0.871
    Xo = np.load(CACHE + '/roite_X_256.npy', mmap_mode='r'); Yo = np.load(CACHE + '/roite_Y_256.npy', mmap_mode='r')
    v = np.where(np.load(CACHE + '/roite_valid_256.npy'))[0]; d = []
    for b in range(0, len(v), 8):
        P = model.predict(np.asarray(Xo[v[b:b + 8]], np.float32), verbose=0)[..., 0]
        for k, j in enumerate(v[b:b + 8]):
            gt = np.asarray(Yo[j])[..., 0].astype(bool); pm = P[k] >= THR; i_ = np.count_nonzero(gt & pm)
            d.append(2 * i_ / (gt.sum() + pm.sum()) if gt.sum() + pm.sum() else 1.0)
    res = dict(seed=args.seed, weights=w, oracle_dice=float(np.mean(d)), oracle_dice_sd=float(np.std(d)), n=len(d))
    json.dump(res, open(OUT + f'/jitter_roi_s{args.seed}.json', 'w'), indent=2); log(json.dumps(res))


if __name__ == '__main__':
    main()
