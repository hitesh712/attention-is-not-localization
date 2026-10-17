"""Cache frozen Mammo-CLIP (EfficientNet-B5) feature maps for CBIS-DDSM at three input resolutions.

Stronger-classifier / resolution experiment (Reviewers 1 & 2): a mammography-specific foundation encoder
(Mammo-CLIP, pretrained on UPMC + VinDr; CBIS-DDSM was not used) is applied frozen; small attention heads
are then trained on the cached maps (mc_heads.py). Preprocessing follows Mammo-CLIP: breast crop
(Otsu + largest component), direct resize to H x W, [0,1] scaling, z-normalisation, 3 channels.
The crop box is stored so that attention maps can be mapped back to original pixel coordinates.

Run with the Mammo-CLIP venv (torch + CUDA):
  D:/pipeline2_mammogram/pipeline2_mammogram/mammo_clip_dlora/.venv/Scripts/python.exe mc_features.py
"""
import os, sys, json, time, argparse
ap = argparse.ArgumentParser()
ap.add_argument('--encoder', default='mammoclip', choices=['mammoclip', 'imagenet'],
                help="imagenet = the same EfficientNet-B5 with natural-image weights (control for domain pretraining)")
ENC = ap.parse_args().encoder
os.environ.setdefault('HF_HUB_OFFLINE', '1' if ENC == 'mammoclip' else '0'); os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2, torch, timm
import cbis_data as D

WEIGHTS = 'G:/My Drive/mammo_clip_colab/model-epoch-7.pt.cleaned.pt'
OUT = 'D:/cbis-ddsm/_mc_cache' + ('' if ENC == 'mammoclip' else '_imagenet'); os.makedirs(OUT, exist_ok=True)
RES = {'r1520': (1520, 912), 'r760': (760, 456), 'r380': (380, 228)}      # (H, W)
# Mammo-CLIP pretraining statistics (pre_train_b5_clip.yaml); ImageNet control uses the ImageNet statistics
MEAN, STD = (0.3089279, 0.25053555408335154) if ENC == 'mammoclip' else (np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]))
T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def breast_box(gray):
    """Otsu + largest connected component bounding box (Mammo-CLIP / subproject crop_to_breast)."""
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, st, _ = cv2.connectedComponentsWithStats(th)
    if n <= 1: return 0, 0, gray.shape[1], gray.shape[0]
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x0, y0, w, h = st[k, 0], st[k, 1], st[k, 2], st[k, 3]
    return int(x0), int(y0), int(x0 + w), int(y0 + h)


def main():
    # ---- image lists: per mammogram; training images with conflicting mass/calc rows are dropped
    tr = D.train_rows()
    lab = tr.groupby('image_key')['label'].agg(['min', 'max'])
    keep = lab.index[lab['min'] == lab['max']]
    first = tr.drop_duplicates('image_key').set_index('image_key')
    train = [(k, first.at[k, 'full_path'], first.at[k, 'patient_id'], int(lab.at[k, 'max'])) for k in sorted(keep)]
    test = [(k, fp, pid, lb) for k, fp, mps, pid, lb in D.test_images()]
    log(f'train images {len(train)} (dropped {len(lab) - len(keep)} mixed mass+calc) | test images {len(test)}')

    if ENC == 'mammoclip':
        model = timm.create_model('tf_efficientnet_b5_ns', pretrained=False, num_classes=0, global_pool='')
        sd = torch.load(WEIGHTS, map_location='cpu', weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        log(f'Mammo-CLIP B5 loaded: {len(missing)} missing, {len(unexpected)} unexpected keys')
        assert not missing and not unexpected
    else:
        global MEAN, STD
        model = timm.create_model('tf_efficientnet_b5_ns', pretrained=True, num_classes=0, global_pool='')
        MEAN, STD = np.array(model.pretrained_cfg['mean']), np.array(model.pretrained_cfg['std'])     # the weights' own statistics
        log(f'ImageNet (noisy-student) EfficientNet-B5 loaded ({model.pretrained_cfg.get("tag")}); mean {MEAN} std {STD}')
    model = model.cuda().eval()

    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        shapes = {r: tuple(model.forward_features(torch.zeros(1, 3, H, W, device='cuda')).shape[1:]) for r, (H, W) in RES.items()}
    log('feature map shapes', shapes)

    for split, items in (('test', test), ('train', train)):
        mm = {r: np.lib.format.open_memmap(f'{OUT}/feat_{r}_{split}.npy', mode='w+', dtype=np.float16,
                                           shape=(len(items),) + shapes[r]) for r in RES}
        meta = []
        for i, (key, fp, pid, lb) in enumerate(items):
            gray = cv2.imread(fp, cv2.IMREAD_GRAYSCALE); Ho, Wo = gray.shape
            x0, y0, x1, y1 = breast_box(gray)
            crop = gray[y0:y1, x0:x1]
            for r, (H, W) in RES.items():
                img = cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
                x3 = np.repeat(img[None], 3, 0)                                   # grey -> 3 channels
                x3 = (x3 - np.reshape(MEAN, (-1, 1, 1))) / np.reshape(STD, (-1, 1, 1))
                t = torch.from_numpy(x3[None].astype(np.float32)).cuda()
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                    mm[r][i] = model.forward_features(t)[0].float().cpu().numpy().astype(np.float16)
            meta.append(dict(key=key, patient=pid, label=lb, H=Ho, W=Wo, box=[x0, y0, x1, y1]))
            if (i + 1) % 200 == 0 or i + 1 == len(items):
                log(f'{split} {i + 1}/{len(items)}')
        for r in RES: mm[r].flush()
        json.dump(meta, open(f'{OUT}/meta_{split}.json', 'w'))
    json.dump(dict(res=RES, shapes=shapes, encoder=ENC, mean=np.ravel(MEAN).tolist(), std=np.ravel(STD).tolist(),
                   weights=WEIGHTS if ENC == 'mammoclip' else 'timm tf_efficientnet_b5_ns (ImageNet)'),
              open(f'{OUT}/config.json', 'w'), indent=2)
    log('done')


if __name__ == '__main__':
    main()
