# Attention Is Not Localization — code and results
DOI: https://doi.org/10.5281/zenodo.23571756
Code, split files and per-image / per-lesion outputs for

> H. Kag, M. Shukla and D. Kumar, "Attention Is Not Localization: A Leakage-Free, End-to-End Evaluation of
> Weakly-Supervised and Supervised Breast-Lesion Localization in Mammography," IEEE Access (under review).

Every number in the paper is produced by `revision/make_numbers.py` from the files in `results/`; nothing was
transcribed by hand.

## Contents

| Folder | What it holds |
|---|---|
| `notebooks/` | Original training / evaluation notebooks (weakly-supervised classifiers, attention-MIL, Attention U-Net v1–v3, ROI delineator, INbreast) |
| `revision/` | Revision experiments: re-scoring, Mammo-CLIP heads, GMIC/GLAM adapter, Mini-MIAS patient-grouped CV, pipeline-aware delineator, detector seeds, nnU-Net export, SegFormer, crop sensitivity, statistics, figures |
| `e2e/` | End-to-end detect-then-delineate evaluation (`e2e_common.py` holds the shared, frozen crop geometry and model code) |
| `splits/` | Patient-disjoint CBIS-DDSM split (incl. the 31 removed overlap patients), validation partition, Mini-MIAS folds |
| `results/` | Per-image and per-lesion outputs (`.npz`) and summaries (`.json`) behind every table |
| `env/` | `pip freeze` of the three environments |
| `weights_SHA256.txt` | Checksums of all trained weights. The weights themselves are attached to the GitHub release (v1.0) as assets: the Keras `.weights.h5` files, `segformer_mit_b2_s0_best.pt`, and `nnunet_2d_fold0.zip` (install with `nnUNetv2_install_pretrained_model_from_zip nnunet_2d_fold0.zip`) |
| `CITATION.cff` | How to cite this work (GitHub shows it as "Cite this repository") |

## Data (not redistributed)

* **CBIS-DDSM** — The Cancer Imaging Archive (JPEG conversion with `dicom_info.csv`).
* **Mini-MIAS** — mammographic image analysis society database (322 films, 161 women).
* **INbreast** — available from its authors on request; only the 107 mass images and 67 BI-RADS 1 images are used, for external testing only.
* **VinDr-Mammo** — original DICOM release 1.0.0 on PhysioNet (restricted access; sign the data use agreement). Only the
  1,363 images with a mass or suspicious-calcification box and 300 BI-RADS 1 images of the official test split are used, for
  external testing only. `e2e/download_vindr_zip.py` fetches exactly these files (about 4.5 GB instead of the 50.7 GB archive)
  by byte-range requests on the project ZIP and verifies each against PhysioNet's SHA-256 list; the selection is written to
  `selection.csv`. In line with the data use agreement, `results/` holds VinDr-Mammo summaries and per-image model outcomes
  keyed by image identifier, but no image data and no annotations.

Paths are set at the top of each script (`D:/cbis-ddsm`, `D:/INbreast Release 1.0`, ...); edit them for your machine.

## Environments

1. **TensorFlow 2.10 + DirectML** (`env/requirements_tensorflow_directml.txt`): EfficientNet-B0 classifiers, Attention U-Nets,
   delineators, end-to-end evaluation. Set `DML_VISIBLE_DEVICES` to the discrete GPU before importing TensorFlow.
2. **PyTorch 2.5.1 + CUDA 12.4** (`env/requirements_torch_mammoclip_gmic_glam.txt`): Mammo-CLIP features and heads, GMIC/GLAM.
3. **PyTorch 2.5.1 + CUDA 12.4 + nnU-Net 2.8.1 + segmentation-models-pytorch 0.5.0** (`env/requirements_torch_nnunet_segformer.txt`).

Third-party models (not included): Mammo-CLIP B5 checkpoint (`shawn24/Mammo-CLIP` on Hugging Face);
GMIC and GLAM code and weights from `github.com/nyukat/GMIC` (AGPL-3.0) and `github.com/nyukat/GLAM` (NYU licence for
non-commercial research and evaluation), placed in `D:/wsol_baselines/GMIC` and `D:/wsol_baselines/GLAM`; they are not
redistributed here. Our adapter (`revision/wsol_nyu.py`) imports their code unmodified (GLAM's patch selector is patched
at run time, see below).

## Reproducing the tables

| Paper item | Script | Output in `results/` |
|---|---|---|
| Weakly-supervised maps, CBIS (Table 2) | `revision/ws_reeval.py` | `ws_reeval.npz/.json` |
| Mini-MIAS patient-grouped CV (Table 2) | `revision/mias_patientwise.py` | `mias_cv.json` |
| Mammo-CLIP / ImageNet heads (Table 4, Fig. 3) | `revision/mc_features.py [--encoder imagenet]`, `revision/mc_heads.py [--encoder imagenet]` | `mc_heads[_imagenet].npz/.json` |
| GMIC / GLAM (Table 5) | `revision/wsol_nyu.py --model gmic|glam` | `wsol_gmic.npz/.json`, `wsol_glam.npz/.json` |
| Strata (Table 6) | `revision/make_numbers.py` | from the files above |
| Detectors (Table 7, Fig. 5) | `revision/detector_v3.py --seed s`, `revision/nnunet_prep.py` + nnU-Net CLI + `revision/nnunet_maps.py`, `revision/segformer_det.py`, then `e2e/e2e_cbis.py`, `e2e/e2e_inbreast.py` | `cbis_e2e_*.npz/.json`, `inbreast_e2e_*.npz/.json` |
| Conditional vs end-to-end (Table 8, Fig. 6) | `revision/jitter_roi.py --seed s` (pipeline-aware delineator), `e2e/e2e_cbis.py`, `e2e/e2e_inbreast.py` | as above |
| Crop-size sensitivity (Table 9) | `revision/crop_sensitivity.py` | `crop_sensitivity.json` |
| VinDr-Mammo (Table 10) | `revision/vindr_prep.py` (detector inputs), `e2e/e2e_vindr_dicom.py` (detection, box-level pipeline; `--minmax` for the sensitivity variant), `revision/segformer_predict.py`, nnU-Net CLI + `revision/nnunet_maps.py vindr`, `revision/wsol_nyu.py --datasets vindr` | `vindrdcm_e2e_*.npz/.json`, `wsol_gmic_vindr.*`, `wsol_glam_vindr.*` |
| Statistics | `revision/stats_revision.py`, `revision/make_numbers.py` | `stats_revision.json` |
| Figures | `revision/fig_revision.py` (Figs. 3, 5, 6), `revision/fig_flowchart_rev.py` (Fig. 1), `e2e/fig_e2e_examples.py` (Fig. 7) | — |

`revision/run_queue_a.ps1`, `run_queue_b.ps1`, `run_queue_c.ps1` and `run_queue_d.ps1` run everything in the order used for the paper
(strictly sequential, one GPU job at a time, on a 6 GB laptop GPU).

## Pitfalls we hit (and fixed)

* **Keras EfficientNet construction.** `EfficientNetB0(weights='imagenet')` inserts an extra input normalization layer that
  `weights=None` omits; a model rebuilt with `weights=None` and loaded from saved weights gives different outputs. Build
  with `weights='imagenet'`, then load the saved weights.
* **GLAM under modern PyTorch.** Its patch selector computes `index / width`, which was integer division in PyTorch 1.1
  and is true division now; `wsol_nyu.py` restores floor division.
* **CBIS-DDSM split.** The official mass and calcification splits share 31 patients; they are removed from training.
* **Mini-MIAS.** Files `mdb(2k-1)` and `mdb(2k)` are the two breasts of one woman; split by woman.
* **VinDr-Mammo PNG conversions.** Check that box annotations are registered with the images before using them; the
  conversion we originally used was not, and its results were discarded. Everything here uses the original DICOM files.
* **VinDr-Mammo display transform.** The stored DICOM pixel values must be passed through each file's own VOI LUT
  (Siemens: VOI LUT Sequence; Planmed, IMS Giotto: window centre / width; Planmed images are MONOCHROME1 and must be
  inverted). Plain min-max scaling leaves the breast in a narrow intensity band; see `revision/vindr_data.py`.
* **PhysioNet downloads from a script.** PhysioNet answers HTTP basic authentication only for wget-style clients.
