"""Apply the trained SegFormer detector (segformer_det.py) to the VinDr-Mammo detector inputs of vindr_prep.py.
Same normalisation and mixed-precision inference as in training-time evaluation; no retraining.
  D:/venvs/nnunet/Scripts/python.exe segformer_predict.py  ->  D:/segformer/mit_b2_s0/P_vindr.npz (keyed by image id)
"""
import json, argparse
import numpy as np, torch
import segmentation_models_pytorch as smp

ap = argparse.ArgumentParser()
ap.add_argument('--run', default='D:/segformer/mit_b2_s0'); ap.add_argument('--encoder', default='mit_b2')
args = ap.parse_args()
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda(); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
ids = json.load(open('D:/vindr_dicom/cache/ids.json')); X = np.load('D:/vindr_dicom/cache/X256.npy', mmap_mode='r')
model = smp.Segformer(encoder_name=args.encoder, encoder_weights=None, classes=1).cuda()
model.load_state_dict(torch.load(args.run + '/best.pt', map_location='cuda', weights_only=True)); model.eval()
out = {}
with torch.no_grad():
    for b in range(0, len(ids), 16):
        x = torch.from_numpy(np.ascontiguousarray(X[b:b + 16])).cuda().float()[:, None].repeat(1, 3, 1, 1) / 255.0
        with torch.autocast('cuda', dtype=torch.float16): p = torch.sigmoid(model((x - MEAN) / STD).float())[:, 0].cpu().numpy()
        for (_, i, _), q in zip(ids[b:b + 16], p): out[i] = q.astype(np.float32)
np.savez_compressed(args.run + '/P_vindr.npz', **out)
print(f'SegFormer maps for {len(out)} VinDr-Mammo images -> {args.run}/P_vindr.npz')
