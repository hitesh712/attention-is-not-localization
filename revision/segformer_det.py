"""Transformer-based full-image detector (Reviewer 2 #3): SegFormer (MiT-B2 encoder, ImageNet initialisation) trained
on exactly the data of the Attention U-Net v3 detector - the cached 256x256 CLAHE mammograms and union lesion masks,
the same patient-grouped train/validation partition (seed 2021), the same BCE + focal-Tversky loss and the same
flip / 90-degree-rotation augmentation. Its probability maps then drive the unchanged end-to-end pipeline
(e2e_cbis.py / e2e_inbreast.py --pmap), so the detector is the only component that differs.

Run with the nnU-Net venv (torch + CUDA + segmentation-models-pytorch), after nnunet_prep.py (INbreast inputs, split):
  D:/venvs/nnunet/Scripts/python.exe segformer_det.py --seed 0
"""
import os, sys, json, time, argparse, glob
os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
import numpy as np, torch, torch.nn.functional as F
import segmentation_models_pytorch as smp
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--encoder', default='mit_b2')
ap.add_argument('--smoke', action='store_true', help='tiny end-to-end check: 32 train / 16 val / 16 test images, 1 epoch')
args = ap.parse_args()
CACHE = 'D:/cbis-ddsm/_seg_cache'; NN = 'D:/nnunet/nnUNet_raw/Dataset501_CBISFull'
OUT = f'D:/segformer/{args.encoder}_s{args.seed}' + ('_smoke' if args.smoke else ''); os.makedirs(OUT, exist_ok=True)
RLOG = 'G:/My Drive/VS_CC/models/revision' if not args.smoke else OUT
BATCH, EPOCHS, LR, WD = 8, (1 if args.smoke else 60), 1e-4, 1e-2
TV_A, TV_B, TV_G = 0.7, 0.3, 0.75                                  # focal-Tversky, as the v3 detector
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda(); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
T0 = time.time(); log = lambda *a: print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)
torch.manual_seed(args.seed); np.random.seed(args.seed); rng = np.random.RandomState(args.seed)

Xtr = np.load(CACHE + '/tr_X_256.npy', mmap_mode='r'); Ytr = np.load(CACHE + '/tr_Y_256.npy', mmap_mode='r')
Xte = np.load(CACHE + '/te_X_256.npy', mmap_mode='r'); Yte = np.load(CACHE + '/te_Y_256.npy', mmap_mode='r')
# the detector's own patient-grouped partition, as exported by nnunet_prep.py (GroupShuffleSplit 0.15, seed 2021)
sp = NN + '/splits_final.backup.json'
sp = sp if os.path.exists(sp) else 'D:/nnunet/nnUNet_preprocessed/Dataset501_CBISFull/splits_final.json'
split = json.load(open(sp))[0]
trn = np.array(sorted(int(n.split('_')[-1]) for n in split['train'])); va = np.array(sorted(int(n.split('_')[-1]) for n in split['val']))
assert len(set(trn) & set(va)) == 0 and len(trn) + len(va) == len(Xtr)
if args.smoke: trn, va, Xte, Yte = trn[:32], va[:16], Xte[:16], Yte[:16]
log(f'train {len(trn)} | val {len(va)} | test {len(Xte)} | encoder {args.encoder} | seed {args.seed}')


def to_t(x):                                                          # uint8 NHWC -> normalised NCHW on the GPU
    t = torch.from_numpy(np.ascontiguousarray(x)).cuda().permute(0, 3, 1, 2).float() / 255.0
    return (t - MEAN) / STD


def batch(idx, augment):
    X = np.asarray(Xtr[idx]); Y = np.asarray(Ytr[idx])
    if augment:
        X = X.copy(); Y = Y.copy()
        for k in range(len(X)):
            if rng.rand() < 0.5: X[k] = X[k, :, ::-1]; Y[k] = Y[k, :, ::-1]
            if rng.rand() < 0.5: X[k] = X[k, ::-1]; Y[k] = Y[k, ::-1]
            r = rng.randint(4)
            if r: X[k] = np.rot90(X[k], r); Y[k] = np.rot90(Y[k], r)
    return to_t(X), torch.from_numpy(np.ascontiguousarray(Y)).cuda().permute(0, 3, 1, 2).float()


def loss_fn(logits, y):
    p = torch.sigmoid(logits).flatten(); t = y.flatten(); tp = (t * p).sum()
    tv = (tp + 1) / (tp + TV_A * (t * (1 - p)).sum() + TV_B * ((1 - t) * p).sum() + 1)
    return F.binary_cross_entropy_with_logits(logits, y) + (1 - tv).clamp_min(0) ** TV_G


@torch.no_grad()
def predict(model, X, bs=16):
    model.eval(); out = []
    for b in range(0, len(X), bs):
        with torch.autocast('cuda', dtype=torch.float16):
            out.append(torch.sigmoid(model(to_t(np.asarray(X[b:b + bs]))).float())[:, 0].cpu().numpy())
    return np.concatenate(out)


def dice_pg(P, Y):
    d, pg = [], []
    for k in range(len(P)):
        gt = np.asarray(Y[k])[..., 0].astype(bool); pm = P[k] >= 0.5; i_ = np.count_nonzero(gt & pm); s = gt.sum() + pm.sum()
        d.append(2 * i_ / s if s else 1.0); yy, xx = np.unravel_index(int(np.argmax(P[k])), P[k].shape); pg.append(int(gt[yy, xx]))
    return np.array(d), np.array(pg)


model = smp.Segformer(encoder_name=args.encoder, encoder_weights='imagenet', classes=1).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5, min_lr=1e-6)
scaler = torch.amp.GradScaler('cuda')
best, best_ep, bad, hist = -1.0, -1, 0, []
for ep in range(EPOCHS):
    model.train(); order = rng.permutation(trn); tl = []
    for b in range(0, len(order), BATCH):
        x, y = batch(np.sort(order[b:b + BATCH]), True)
        with torch.autocast('cuda', dtype=torch.float16): logits = model(x)
        loss = loss_fn(logits.float(), y)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tl.append(loss.item())
    vd = float(dice_pg(predict(model, Xtr[va]), Ytr[va])[0].mean()); sched.step(vd)
    hist.append(dict(epoch=ep + 1, loss=float(np.mean(tl)), val_dice=vd, lr=opt.param_groups[0]['lr']))
    log(f'epoch {ep + 1:2d} loss {np.mean(tl):.4f} val Dice {vd:.4f} lr {opt.param_groups[0]["lr"]:.1e}')
    if vd > best: best, best_ep, bad = vd, ep + 1, 0; torch.save(model.state_dict(), OUT + '/best.pt')
    else:
        bad += 1
        if bad >= 12: break
model.load_state_dict(torch.load(OUT + '/best.pt', map_location='cuda', weights_only=True))

# ---- CBIS test maps (published image order) + the detector-only metrics of detector_v3.py
P = predict(model, Xte).astype(np.float32); np.save(OUT + '/P_cbis_test.npy', P)
d, pg = dice_pg(P, Yte)
# ---- INbreast maps from the nnU-Net export (identical full_input() preprocessing: resize 256 + CLAHE)
inb = {}
for f in sorted(glob.glob(NN + '/imagesTs_inbreast/inb_*_0000.png'))[:(3 if args.smoke else None)]:
    g = np.array(Image.open(f).convert('L'))
    inb[os.path.basename(f)[4:-9]] = predict(model, np.repeat(g[None, :, :, None], 3, -1))[0].astype(np.float32)
np.savez_compressed(OUT + '/P_inbreast.npz', **inb)
res = dict(model=f'SegFormer-{args.encoder}', seed=args.seed, best_epoch=best_ep, val_dice=best, test_dice=float(d.mean()),
           test_dice_sd=float(d.std()), test_dice_median=float(np.median(d)), test_pg=float(pg.mean()), n=len(d),
           n_inbreast_maps=len(inb), params_m=sum(p.numel() for p in model.parameters()) / 1e6, history=hist,
           maps=dict(cbis=OUT + '/P_cbis_test.npy', inbreast=OUT + '/P_inbreast.npz'))
json.dump(res, open(f'{RLOG}/segformer_{args.encoder}_s{args.seed}.json', 'w'), indent=2)
log(json.dumps({k: v for k, v in res.items() if k != 'history'}))
