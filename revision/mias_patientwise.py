"""Mini-MIAS re-analysis with PATIENT-grouped cross-validation (Reviewer 1: image-wise split is not patient-disjoint).

mini-MIAS images are left/right MLO pairs of the same woman (mdb[2k-1], mdb[2k]), so patient = (n + 1) // 2.
Same model and protocol as mias_attention_localization.ipynb (EfficientNetB0 frozen to block6a + CBAM, rotations
on training images only, class weights, early stopping on validation AUC), but evaluated with 5-fold
StratifiedGroupKFold by patient so that every annotated abnormal image is tested exactly once. An identical
image-wise 5-fold run is included to show whether the split choice changes the result.
"""
import os, sys, json, time, gc
os.environ.setdefault('DML_VISIBLE_DEVICES', '0'); os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np, pandas as pd, cv2, tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import EfficientNetB0
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

for g in tf.config.list_physical_devices('GPU'):
    try: tf.config.experimental.set_memory_growth(g, True)
    except Exception: pass
IMAGE_DIR = 'G:/My Drive/Colab/all-mias/'; INFO = IMAGE_DIR + 'Info1.txt'
OUT = 'G:/My Drive/VS_CC/models/revision'; os.makedirs(OUT, exist_ok=True)
S, ORIG, SEED = 224, 1024, 2021; T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)
_clahe = cv2.createCLAHE(2.0, (8, 8))


def label_table():                                             # verbatim build_label_table()
    rows = {}
    for line in open(INFO):
        p = line.split()
        if len(p) < 3 or not p[0].startswith('mdb'): continue
        r = rows.setdefault(p[0], {'image_id': p[0], 'label': 0, 'lesions': []})
        if p[2] != 'NORM':
            r['label'] = 1
            if len(p) >= 7:
                try: r['lesions'].append((int(p[4]), int(p[5]), int(p[6])))
                except ValueError: pass
    df = pd.DataFrame(rows.values()).sort_values('image_id').reset_index(drop=True)
    df['patient'] = (df['image_id'].str[3:].astype(int) + 1) // 2
    return df[df['image_id'].apply(lambda i: os.path.exists(IMAGE_DIR + i + '.pgm'))].reset_index(drop=True)


def load(img_id):
    return np.stack([_clahe.apply(cv2.resize(cv2.imread(IMAGE_DIR + img_id + '.pgm', cv2.IMREAD_GRAYSCALE), (S, S)))] * 3, -1)


def rotate(img, a):                                            # verbatim (train-only augmentation)
    if a == 0: return img
    M = cv2.getRotationMatrix2D((S // 2, S // 2), a, 1.0)
    return cv2.warpAffine(img, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def cbam_block(x, ratio=16, name='cbam'):
    f = x.shape[-1]
    s1 = layers.Dense(max(1, f // ratio), activation='relu', kernel_initializer='he_normal')
    s2 = layers.Dense(f, kernel_initializer='he_normal')
    avg = layers.Reshape((1, 1, f))(layers.GlobalAveragePooling2D()(x)); mx = layers.Reshape((1, 1, f))(layers.GlobalMaxPooling2D()(x))
    x = layers.Multiply()([x, layers.Activation('sigmoid')(layers.Add()([s2(s1(avg)), s2(s1(mx))]))])
    ap = layers.Lambda(lambda t: tf.reduce_mean(t, -1, keepdims=True))(x); mp = layers.Lambda(lambda t: tf.reduce_max(t, -1, keepdims=True))(x)
    sp = layers.Conv2D(1, 7, padding='same', activation='sigmoid', name=name + '_spatial')(layers.Concatenate()([ap, mp]))
    return layers.Multiply()([x, sp])


def build():
    base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(S, S, 3)); base.trainable = False
    x = layers.Activation('linear', name='loc_features')(cbam_block(base.get_layer('block6a_activation').output))
    m = Model(base.input, layers.Dense(1, activation='sigmoid')(layers.Dropout(0.5)(layers.GlobalAveragePooling2D(name='gap')(x))))
    m.compile(Adam(1e-4), 'binary_crossentropy', [tf.keras.metrics.AUC(name='auc')])
    return m


def gradcam(m, x):
    gm = Model(m.inputs, [m.get_layer('loc_features').output, m.output])
    with tf.GradientTape() as t:
        conv, pred = gm(x); y = pred[:, 0]
    g = t.gradient(y, conv)
    h = tf.nn.relu(tf.reduce_sum(conv[0] * tf.reduce_mean(g, axis=(0, 1, 2)), -1))
    return (h / (tf.reduce_max(h) + 1e-8)).numpy()


def localize(heat, lesions):                                   # verbatim metrics (MIAS origin bottom-left)
    heat = cv2.resize(heat.astype(np.float32), (S, S)); gt = np.zeros((S, S), np.uint8); centres = []
    for x, y, r in lesions:
        cx, cy, rr = x * S / ORIG, (ORIG - y) * S / ORIG, r * S / ORIG; centres.append((cx, cy))
        cv2.circle(gt, (int(round(cx)), int(round(cy))), max(1, int(round(rr))), 1, -1)
    py, px = np.unravel_index(int(np.argmax(heat)), heat.shape)
    pm = (heat >= 0.5 * heat.max()).astype(np.uint8); inter = np.logical_and(pm, gt).sum(); u = np.logical_or(pm, gt).sum()
    return int(gt[py, px] == 1), min(np.hypot(px - cx, py - cy) for cx, cy in centres) / (S * np.sqrt(2)), (inter / u if u else 0.0)


def run(df, X, scheme):
    y = df['label'].values; grp = df['patient'].values
    folds = (StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, grp) if scheme == 'patient'
             else StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
    rec = []
    for k, (trv, te) in enumerate(folds):
        tr_i, va_i = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(trv, groups=grp[trv] if scheme == 'patient' else trv))
        tr, va = trv[tr_i], trv[va_i]
        assert scheme != 'patient' or (set(grp[tr]).isdisjoint(grp[te]) and set(grp[va]).isdisjoint(grp[te]))
        xt = np.concatenate([np.stack([rotate(X[i], a) for i in tr]) for a in (0, 90, 180, 270)]); yt = np.tile(y[tr], 4)
        cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=yt)
        tf.keras.backend.clear_session(); tf.random.set_seed(SEED + k); m = build()
        m.fit(xt, yt, validation_data=(X[va], y[va]), epochs=30, batch_size=16, class_weight={0: cw[0], 1: cw[1]}, verbose=0,
              callbacks=[EarlyStopping(monitor='val_auc', mode='max', patience=6, restore_best_weights=True)])
        p = m.predict(X[te], batch_size=16, verbose=0).ravel()
        sp = Model(m.inputs, m.get_layer('cbam_spatial').output)
        for j, i in enumerate(te):
            r = dict(image_id=df.at[i, 'image_id'], patient=int(grp[i]), fold=k, label=int(y[i]), prob=float(p[j]))
            les = df.at[i, 'lesions']
            if y[i] == 1 and len(les):
                xi = X[i][None]
                r['gc_pg'], r['gc_dist'], r['gc_iou'] = localize(gradcam(m, xi), les)
                r['cb_pg'], r['cb_dist'], r['cb_iou'] = localize(sp.predict(xi, verbose=0)[0, :, :, 0], les)
            rec.append(r)
        log(f'{scheme}-wise fold {k}: test AUC {roc_auc_score(y[te], p):.3f} (n={len(te)}, {int(y[te].sum())} abnormal)')
        del xt, m, sp; gc.collect()
    return pd.DataFrame(rec)


def main():
    df = label_table(); X = np.stack([load(i) for i in df['image_id']])
    log(f'{len(df)} images | {df["patient"].nunique()} patients | {int(df["label"].sum())} abnormal | '
        f'{int((df["lesions"].str.len() > 0).sum())} with lesion coordinates')
    S_ = {}
    for scheme in ('patient', 'image'):
        R = run(df, X, scheme); R.to_csv(f'{OUT}/mias_{scheme}wise_cv.csv', index=False)
        fa = [roc_auc_score(g['label'], g['prob']) for _, g in R.groupby('fold')]
        L = R.dropna(subset=['gc_pg'])
        S_[scheme] = dict(pooled_auc=roc_auc_score(R['label'], R['prob']), fold_auc_mean=float(np.mean(fa)), fold_auc_sd=float(np.std(fa)),
                          n_loc=len(L), gradcam_pg=float(L['gc_pg'].mean()), gradcam_iou=float(L['gc_iou'].mean()),
                          gradcam_dist=float(L['gc_dist'].mean()), cbam_pg=float(L['cb_pg'].mean()), cbam_iou=float(L['cb_iou'].mean()))
        log(f'{scheme}-wise 5-fold: ' + json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in S_[scheme].items()}))
    json.dump(S_, open(f'{OUT}/mias_cv.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
