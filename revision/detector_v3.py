"""Retrain the full-image detector with the published v3 recipe under a different seed (Reviewer 2 #10:
run-to-run variability). Identical data, cache, train/validation split, architecture, loss and schedule as
cbis_attention_unet_v3.ipynb; only the initialisation / shuffling / augmentation seed differs.
Usage:  python detector_v3.py --seed 1
"""
import os, sys, json, time, argparse
os.environ.setdefault('DML_VISIBLE_DEVICES', '0'); os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import GroupShuffleSplit
import cbis_data as D

ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, required=True); args = ap.parse_args()
for g in tf.config.list_physical_devices('GPU'):
    try: tf.config.experimental.set_memory_growth(g, True)
    except Exception: pass
SZ, BATCH, RS = 256, 4, 2021; TV_A, TV_B, TV_G = 0.7, 0.3, 0.75
MODELS = 'G:/My Drive/VS_CC/models'; OUT = MODELS + '/revision'
T0 = time.time(); log = lambda *a: print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)
tf.keras.utils.set_random_seed(args.seed)

tr = D.train_rows(); d = tr[tr['mask_path'].notna()]
pats = np.array([rows.iloc[0]['patient_id'] for _, rows in d.groupby('image_key')])
Xtr = np.load(D.CACHE + '/tr_X_256.npy', mmap_mode='r'); Ytr = np.load(D.CACHE + '/tr_Y_256.npy', mmap_mode='r')
Xte = np.load(D.CACHE + '/te_X_256.npy', mmap_mode='r'); Yte = np.load(D.CACHE + '/te_Y_256.npy', mmap_mode='r')
assert len(pats) == len(Xtr)
trn, va = next(GroupShuffleSplit(1, test_size=0.15, random_state=RS).split(np.arange(len(Xtr)), groups=pats))


class SegSeq(tf.keras.utils.Sequence):                       # verbatim
    def __init__(self, X, Y, idx, batch, shuffle=True, augment=False):
        self.X = X; self.Y = Y; self.idx = np.asarray(idx); self.batch = batch
        self.shuffle = shuffle; self.augment = augment; self.on_epoch_end()
    def __len__(self): return int(np.ceil(len(self.idx) / self.batch))
    def on_epoch_end(self):
        self.order = np.arange(len(self.idx))
        if self.shuffle: np.random.shuffle(self.order)
    def __getitem__(self, b):
        sel = np.sort(self.idx[self.order[b * self.batch:(b + 1) * self.batch]])
        X = self.X[sel].astype('float32'); Y = self.Y[sel].astype('float32')
        if self.augment:
            for k in range(len(X)):
                if np.random.rand() < 0.5: X[k] = X[k, :, ::-1]; Y[k] = Y[k, :, ::-1]
                if np.random.rand() < 0.5: X[k] = X[k, ::-1]; Y[k] = Y[k, ::-1]
                r = np.random.randint(4)
                if r: X[k] = np.rot90(X[k], r); Y[k] = np.rot90(Y[k], r)
        return X, Y


def conv_block(x, f):
    for _ in range(2): x = layers.Activation('relu')(layers.BatchNormalization()(layers.Conv2D(f, 3, padding='same')(x)))
    return x


def attn_gate(skip, gating, inter):
    r = layers.Activation('relu')(layers.Add()([layers.Conv2D(inter, 1)(skip), layers.Conv2D(inter, 1)(gating)]))
    return layers.Multiply()([skip, layers.Conv2D(1, 1, activation='sigmoid')(r)])


def build():
    inp = layers.Input((SZ, SZ, 3)); x = layers.Rescaling(1. / 127.5, offset=-1)(inp)
    base = MobileNetV2(input_tensor=x, include_top=False, weights='imagenet'); base.trainable = False
    s128, s64, s32, s16, s8 = [base.get_layer(n).output for n in ['block_1_expand_relu', 'block_3_expand_relu',
                                                                  'block_6_expand_relu', 'block_13_expand_relu', 'block_16_project']]
    def up(x, skip, f):
        x = layers.UpSampling2D()(x); return conv_block(layers.Concatenate()([x, attn_gate(skip, x, f)]), f)
    d = up(s8, s16, 256); d = up(d, s32, 128); d = up(d, s64, 64); d = up(d, s128, 32)
    return Model(inp, layers.Conv2D(1, 1, activation='sigmoid')(conv_block(layers.UpSampling2D()(d), 16))), base


def _flat(a, b): return tf.reshape(tf.cast(a, tf.float32), [-1]), tf.reshape(b, [-1])
def tversky(yt, yp, s=1.):
    t, p = _flat(yt, yp); tp = tf.reduce_sum(t * p)
    return (tp + s) / (tp + TV_A * tf.reduce_sum(t * (1 - p)) + TV_B * tf.reduce_sum((1 - t) * p) + s)
def bce_ftversky(yt, yp):
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.cast(yt, tf.float32), yp)) + tf.pow(1 - tversky(yt, yp), TV_G)
def dice_coef(yt, yp, s=1.):
    t, p = _flat(yt, yp); i = tf.reduce_sum(t * p); return (2 * i + s) / (tf.reduce_sum(t) + tf.reduce_sum(p) + s)


model, base = build()
gtr = SegSeq(Xtr, Ytr, trn, BATCH, True, True); gva = SegSeq(Xtr, Ytr, va, BATCH, False, False)
model.compile(Adam(1e-3), loss=bce_ftversky, metrics=[dice_coef])
model.fit(gtr, validation_data=gva, epochs=12, verbose=2,
          callbacks=[EarlyStopping(monitor='val_dice_coef', mode='max', patience=6, restore_best_weights=True)])
base.trainable = True
for l in base.layers:
    if isinstance(l, layers.BatchNormalization): l.trainable = False
model.compile(Adam(1e-4), loss=bce_ftversky, metrics=[dice_coef])
model.fit(gtr, validation_data=gva, epochs=40, verbose=2,
          callbacks=[EarlyStopping(monitor='val_dice_coef', mode='max', patience=12, restore_best_weights=True),
                     ReduceLROnPlateau(monitor='val_dice_coef', mode='max', factor=0.5, patience=5, min_lr=1e-6)])
w = MODELS + f'/cbis_attention_unet_v3_s{args.seed}.weights.h5'; model.save_weights(w)
d_, pg = [], []
for b in range(0, len(Xte), 8):
    P = model.predict(np.asarray(Xte[b:b + 8], np.float32), verbose=0)[..., 0]
    for k in range(len(P)):
        gt = np.asarray(Yte[b + k])[..., 0].astype(bool); pm = P[k] >= 0.5; i_ = np.count_nonzero(gt & pm); s = gt.sum() + pm.sum()
        d_.append(2 * i_ / s if s else 1.0); yy, xx = np.unravel_index(int(np.argmax(P[k])), P[k].shape); pg.append(int(gt[yy, xx]))
res = dict(seed=args.seed, weights=w, test_dice=float(np.mean(d_)), test_dice_sd=float(np.std(d_)), test_pg=float(np.mean(pg)), n=len(d_))
json.dump(res, open(OUT + f'/detector_v3_s{args.seed}.json', 'w'), indent=2); log(json.dumps(res))
