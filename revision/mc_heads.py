"""Attention heads on frozen Mammo-CLIP features: does a stronger, higher-resolution classifier localize?

For each input resolution (r380 ~ 224-equivalent, r760, r1520 = native Mammo-CLIP) and 3 seeds, train
  cam   : dropout -> GAP -> linear              (classic CAM is exact for this head)
  cbam  : CBAM(channel+spatial) -> GAP -> linear (CBAM spatial map + Grad-CAM on the features)
  mil   : gated-attention MIL over feature-map cells (attention weights = localization map)
on CBIS mass-vs-calcification (patient-disjoint; validation = 15% of training patients), then score on the
399 masked test mammograms: row-level and image-level AUC; pointing (map peak inside the lesion) and IoU/Dice
of the >= half-max region, both in breast-crop space (760x456).

Memory: the 2048-channel maps are projected onto their top-512 principal components (exact covariance streamed
on the GPU from the training set only); a linear projection keeps the CAM exact.
"""
import os, sys, json, time, argparse
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
import cbis_data as D

ap = argparse.ArgumentParser()
ap.add_argument('--encoder', default='mammoclip', choices=['mammoclip', 'imagenet'])   # imagenet = control (mc_features.py --encoder imagenet)
ENC = ap.parse_args().encoder; SUF = '' if ENC == 'mammoclip' else '_imagenet'
C = 'D:/cbis-ddsm/_mc_cache' + SUF; OUT = 'G:/My Drive/VS_CC/models/revision'; os.makedirs(OUT, exist_ok=True)
CFG = json.load(open(C + '/config.json')); SEEDS = (0, 1, 2); K = 512; T0 = time.time()
BS = {'r1520': 16, 'r760': 32, 'r380': 64}
log = lambda *a: print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)
dev = 'cuda'


def to_dev(X, sel):
    return torch.from_numpy(np.asarray(X[sel])).to(dev).float()          # fp16 on the host, fp32 on the GPU


def reduce(r):
    """Top-K principal components of the training cells (exact streamed covariance) -> reduced memmaps."""
    paths = {s: f'{C}/feat_{r}_{s}_p{K}.npy' for s in ('train', 'test')}
    if all(os.path.exists(p) for p in paths.values()):
        return {s: np.load(p, mmap_mode='r') for s, p in paths.items()}, json.load(open(f'{C}/pca_{r}.json'))['explained']
    Xtr = np.load(f'{C}/feat_{r}_train.npy', mmap_mode='r'); c = Xtr.shape[1]
    S = torch.zeros(c, dtype=torch.float64, device=dev); Q = torch.zeros(c, c, dtype=torch.float64, device=dev); n = 0
    for b in range(0, len(Xtr), BS[r]):
        f = to_dev(Xtr, np.arange(b, min(b + BS[r], len(Xtr)))).permute(0, 2, 3, 1).reshape(-1, c)
        S += f.sum(0).double(); Q += (f.T @ f).double(); n += f.shape[0]
    mu = S / n; cov = Q / n - torch.outer(mu, mu)
    ev, V = torch.linalg.eigh(cov); ev, V = ev.flip(0), V.flip(1)
    P = V[:, :K].float(); muf = mu.float(); explained = float(ev[:K].sum() / ev.sum())
    out = {}
    for s in ('train', 'test'):
        X = np.load(f'{C}/feat_{r}_{s}.npy', mmap_mode='r'); h, w = X.shape[2:]
        Y = np.lib.format.open_memmap(paths[s], mode='w+', dtype=np.float16, shape=(len(X), K, h, w))
        for b in range(0, len(X), BS[r]):
            f = to_dev(X, np.arange(b, min(b + BS[r], len(X))))
            z = torch.einsum('bchw,ck->bkhw', f - muf[None, :, None, None], P)
            Y[b:b + len(z)] = z.cpu().numpy().astype(np.float16)
        Y.flush(); out[s] = np.load(paths[s], mmap_mode='r')
    json.dump(dict(k=K, explained=explained, n_cells=n), open(f'{C}/pca_{r}.json', 'w'))
    return out, explained


class CAMHead(nn.Module):
    def __init__(s, c=K):
        super().__init__(); s.drop = nn.Dropout(0.3); s.fc = nn.Linear(c, 1)
    def forward(s, f):
        return s.fc(s.drop(f.mean((2, 3))))[:, 0]
    def maps(s, f):
        return torch.einsum('bchw,c->bhw', f, s.fc.weight[0])


class CBAMHead(nn.Module):
    def __init__(s, c=K, r=16):
        super().__init__()
        s.mlp = nn.Sequential(nn.Linear(c, c // r), nn.ReLU(), nn.Linear(c // r, c))
        s.sp = nn.Conv2d(2, 1, 7, padding=3); s.drop = nn.Dropout(0.3); s.fc = nn.Linear(c, 1)
    def attend(s, f):
        f = f * torch.sigmoid(s.mlp(f.mean((2, 3))) + s.mlp(f.amax((2, 3))))[:, :, None, None]
        sp = torch.sigmoid(s.sp(torch.cat([f.mean(1, keepdim=True), f.amax(1, keepdim=True)], 1)))
        return f * sp, sp[:, 0]
    def forward(s, f):
        return s.fc(s.drop(s.attend(f)[0].mean((2, 3))))[:, 0]


class MILHead(nn.Module):
    def __init__(s, c=K, hid=128, att=64):
        super().__init__()
        s.emb = nn.Sequential(nn.Linear(c, hid), nn.ReLU(), nn.Dropout(0.3))
        s.V = nn.Linear(hid, att); s.U = nn.Linear(hid, att); s.w = nn.Linear(att, 1); s.fc = nn.Linear(hid, 1)
    def attn(s, f):
        e = s.emb(f.flatten(2).transpose(1, 2))
        return torch.softmax(s.w(torch.tanh(s.V(e)) * torch.sigmoid(s.U(e)))[..., 0], 1), e
    def forward(s, f):
        a, e = s.attn(f)
        return s.fc((a[..., None] * e).sum(1))[:, 0]


HEADS = {'cam': CAMHead, 'cbam': CBAMHead, 'mil': MILHead}


def train_head(kind, X, y, tr, va, seed, bs, epochs=40, patience=8):
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    m = HEADS[kind]().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    pw = torch.tensor(np.sum(y[tr] == 0) / max(np.sum(y[tr] == 1), 1), device=dev)
    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        m.train(); order = rng.permutation(tr)
        for b in range(0, len(order), bs):
            sel = np.sort(order[b:b + bs])
            loss = F.binary_cross_entropy_with_logits(m(to_dev(X, sel)), torch.from_numpy(y[sel]).float().to(dev), pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pv = np.concatenate([torch.sigmoid(m(to_dev(X, np.sort(va[b:b + bs])))).cpu().numpy() for b in range(0, len(va), bs)])
        auc = roc_auc_score(y[np.concatenate([np.sort(va[b:b + bs]) for b in range(0, len(va), bs)])], pv)
        if auc > best: best, bad, best_state = auc, 0, {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    m.load_state_dict(best_state); m.eval()
    return m, best


def maps_for(kind, m, f):
    if kind == 'cam':
        with torch.no_grad(): return {'cam': m.maps(f)}
    if kind == 'cbam':
        f = f.clone().requires_grad_(True); g, = torch.autograd.grad(m(f).sum(), f)
        with torch.no_grad():
            return {'gradcam': F.relu((f * g.mean((2, 3), keepdim=True)).sum(1)), 'cbam_spatial': m.attend(f)[1]}
    with torch.no_grad():
        a, _ = m.attn(f); return {'mil_attention': a.view(f.shape[0], *f.shape[2:])}


def gt_crop_masks(meta):
    """Union GT of each test image in breast-crop space (760x456)."""
    cache = C + '/gt_test_crop760.npy'
    if os.path.exists(cache): return np.load(cache)
    imgs = {k: mps for k, fp, mps, pid, lb in D.test_images()}
    g = np.zeros((len(meta), 760, 456), bool)
    for i, mt in enumerate(meta):
        x0, y0, x1, y1 = mt['box']; full = np.zeros((mt['H'], mt['W']), bool)
        for mp in imgs[mt['key']]:
            mk = D.read_mask(mp, (mt['H'], mt['W']))
            if mk is not None: full |= mk
        g[i] = cv2.resize(full[y0:y1, x0:x1].astype(np.uint8) * 255, (456, 760), interpolation=cv2.INTER_NEAREST) >= 127
    np.save(cache, g); return g


def score(mp, gt):
    up = cv2.resize(mp.astype(np.float32), (456, 760), interpolation=cv2.INTER_LINEAR)
    py, px = np.unravel_index(int(np.argmax(up)), up.shape)
    lo, hi = float(up.min()), float(up.max())                 # min-max: shift/scale invariant (CAM can be < 0)
    pm = (up - lo) >= 0.5 * (hi - lo) if hi > lo else np.zeros(up.shape, bool)
    inter = np.count_nonzero(pm & gt); s = np.count_nonzero(pm) + np.count_nonzero(gt); u = s - inter
    return int(gt[py, px]), (inter / u if u else 0.0), (2 * inter / s if s else 0.0)


def main():
    mtr = json.load(open(C + '/meta_train.json')); mte = json.load(open(C + '/meta_test.json'))
    ytr = np.array([m['label'] for m in mtr]); yte = np.array([m['label'] for m in mte])
    tr, va = next(GroupShuffleSplit(1, test_size=0.15, random_state=2021).split(np.arange(len(mtr)), groups=[m['patient'] for m in mtr]))
    rows = D.test_rows(); key2i = {m['key']: i for i, m in enumerate(mte)}
    rk = rows['image_key'].map(key2i).values; yr = rows['label'].values
    gt = gt_crop_masks(mte)
    log(f'train {len(tr)} | val {len(va)} | test {len(mte)} | GT ready')
    RES, ALL = {}, {}
    for r in CFG['res']:
        X, explained = reduce(r); bs = BS[r]
        log(f'{r}: PCA {K} components retain {explained * 100:.1f}% of feature variance')
        RES[f'{r}_pca_explained'] = explained
        for kind in HEADS:
            for seed in SEEDS:
                m, vauc = train_head(kind, X['train'], ytr, tr, va, seed, bs)
                probs, hits = [], {}
                for b in range(0, len(mte), bs):
                    sel = np.arange(b, min(b + bs, len(mte))); f = to_dev(X['test'], sel)
                    with torch.no_grad(): probs.append(torch.sigmoid(m(f)).cpu().numpy())
                    for name, mp in maps_for(kind, m, f).items():
                        mp = mp.cpu().numpy()
                        for k, j in enumerate(sel): hits.setdefault(name, []).append(score(mp[k], gt[j]))
                p = np.concatenate(probs)
                res = dict(val_auc=vauc, auc_rows=roc_auc_score(yr, p[rk]), auc_images=roc_auc_score(yte, p))
                for name, h in hits.items():
                    h = np.array(h); ALL[f'{r}_{kind}_{name}_s{seed}'] = h
                    res[name] = dict(pointing=float(h[:, 0].mean()), iou=float(h[:, 1].mean()), dice=float(h[:, 2].mean()))
                ALL[f'{r}_{kind}_prob_s{seed}'] = p; RES[f'{r}_{kind}_s{seed}'] = res
                log(f'{r} {kind:<4} s{seed}: val AUC {vauc:.3f} | test AUC rows {res["auc_rows"]:.3f} images {res["auc_images"]:.3f} | '
                    + ' | '.join(f'{n} PG {res[n]["pointing"] * 100:.1f}% IoU {res[n]["iou"]:.3f}' for n in hits))
                del m; torch.cuda.empty_cache()
        json.dump(RES, open(OUT + f'/mc_heads{SUF}.json', 'w'), indent=2)
    np.savez_compressed(OUT + f'/mc_heads{SUF}.npz', **ALL, labels=yte, keys=np.array([m['key'] for m in mte]),
                        patients=np.array([m['patient'] for m in mte]))
    log('done')


if __name__ == '__main__':
    main()
