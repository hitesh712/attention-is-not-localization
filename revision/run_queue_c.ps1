# Revision job queue C - alternative detectors (nnU-Net, SegFormer; Reviewer 2 #3), dedicated WSOL models (GMIC, GLAM;
# Reviewer 1 #3 / Reviewer 2 #3), crop-size sensitivity (Reviewer 2 #8), the ImageNet-encoder control for the Mammo-CLIP
# experiment and the rule-based end-to-end example figure. Runs after queue B. Strictly sequential (one GPU job at a time).
# Order: short jobs first, the long nnU-Net training last, so that a problem there cannot delay the other results.
$TF = 'D:\pipeline2_mammogram\pipeline2_mammogram\.venv_tfdml\Scripts\python.exe'
$TORCH = 'D:\pipeline2_mammogram\pipeline2_mammogram\mammo_clip_dlora\.venv\Scripts\python.exe'
$NN = 'D:\venvs\nnunet\Scripts'
$REV = 'G:\My Drive\VS_CC\revision'; $E2E = 'G:\My Drive\VS_CC\e2e'
$RLOG = 'G:\My Drive\VS_CC\models\revision'; $ELOG = 'G:\My Drive\VS_CC\models\e2e'
$DS = 'Dataset501_CBISFull'; $RAWD = "D:\nnunet\nnUNet_raw\$DS"; $PRED = "D:\nnunet\nnUNet_preprocessed\$DS"
$JIT = 'G:/My Drive/VS_CC/models/cbis_roi_seg_jitter_s2021.weights.h5'
$SF = 'D:/segformer/mit_b2_s0'
$env:nnUNet_raw = 'D:\nnunet\nnUNet_raw'; $env:nnUNet_preprocessed = 'D:\nnunet\nnUNet_preprocessed'
$env:nnUNet_results = 'D:\nnunet\nnUNet_results'; $env:nnUNet_n_proc_DA = '3'; $env:nnUNet_def_n_proc = '2'; $env:nnUNet_compile = 'false'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'; $env:NO_ALBUMENTATIONS_UPDATE = '1'
$noise = 'tensorflow/c/logging.cc|DirectML device|dxgi|d3d12|Successfully opened|oneDNN|cpu_feature_guard|rebuild TensorFlow|UserWarning|warnings.warn'

function Step([string]$name, [scriptblock]$body, [string]$outfile) {
    Add-Content "$RLOG\queue_c.log" "$(Get-Date -Format s)  START $name"
    & $body 2>&1 | Where-Object { $_ -notmatch $noise } | Out-File -Encoding utf8 $outfile
    Add-Content "$RLOG\queue_c.log" "$(Get-Date -Format s)  END   $name (exit $LASTEXITCODE)"
}

function E2EPair([string]$n, [string]$pc, [string]$pi) {
    Step "cbis_${n}_pub" { Set-Location $E2E; & $TF -u e2e_cbis.py --pmap $pc --tag "${n}_pub" } "$ELOG\cbis_e2e_${n}_pub.log"
    Step "cbis_${n}_jit" { Set-Location $E2E; & $TF -u e2e_cbis.py --pmap $pc --roi $JIT --tag "${n}_jit" } "$ELOG\cbis_e2e_${n}_jit.log"
    Step "inb_${n}_pub"  { Set-Location $E2E; & $TF -u e2e_inbreast.py --pmap $pi --tag "${n}_pub" } "$ELOG\inbreast_e2e_${n}_pub.log"
    Step "inb_${n}_jit"  { Set-Location $E2E; & $TF -u e2e_inbreast.py --pmap $pi --roi $JIT --tag "${n}_jit" } "$ELOG\inbreast_e2e_${n}_jit.log"
}

while (-not ((Get-Content "$RLOG\queue_b.log" -Raw -ErrorAction SilentlyContinue) -match 'QUEUE B DONE')) { Start-Sleep 60 }
while (-not ((Get-Content 'D:\venvs\nnunet_torchfix.log' -Raw -ErrorAction SilentlyContinue) -match 'TORCHFIX DONE')) { Start-Sleep 60 }

# export + planning/preprocessing are CPU-only and were run ahead of the queue (10 Oct 2026, 15:11-15:19); redo only if missing
if (-not (Test-Path "$RAWD\splits_final.backup.json")) {
    Step 'nn_prep'  { Set-Location $REV; & $TF -u nnunet_prep.py } "$RLOG\nn_prep.log"
    Copy-Item "$PRED\splits_final.json" "$RAWD\splits_final.backup.json" -Force
}

Step 'segformer' { Set-Location $REV; & "$NN\python.exe" -u segformer_det.py --seed 0 } "$RLOG\segformer_s0.log"
E2EPair 'segformer' "$SF/P_cbis_test.npy" "$SF/P_inbreast.npz"
Step 'wsol_gmic' { Set-Location $REV; & $TORCH -u wsol_nyu.py --model gmic } "$RLOG\wsol_gmic.log"
Step 'wsol_glam' { Set-Location $REV; & $TORCH -u wsol_nyu.py --model glam } "$RLOG\wsol_glam.log"
Step 'crop_sens' { Set-Location $REV; & $TF -u crop_sensitivity.py } "$RLOG\crop_sensitivity.log"
Step 'fig_e2e_examples' { Set-Location $E2E; & $TF -u fig_e2e_examples.py } "$RLOG\fig_e2e_examples.log"
Step 'mc_features_imagenet' { Set-Location $REV; & $TORCH -u mc_features.py --encoder imagenet } "$RLOG\mc_features_imagenet.log"
Step 'mc_heads_imagenet'    { Set-Location $REV; & $TORCH -u mc_heads.py --encoder imagenet } "$RLOG\mc_heads_imagenet.log"

if (-not (Test-Path "$PRED\nnUNetPlans.json")) {
    Step 'nn_plan'  { & "$NN\nnUNetv2_plan_and_preprocess.exe" -d 501 -c 2d --verify_dataset_integrity -np 1 -npfp 1 -gpu_memory_target 5 } "$RLOG\nn_plan.log"
}
Copy-Item "$RAWD\splits_final.backup.json" "$PRED\splits_final.json" -Force      # patient-grouped fold 0
Step 'nn_train' { & "$NN\nnUNetv2_train.exe" 501 2d 0 -tr nnUNetTrainer_100epochs } "$RLOG\nn_train.log"
Step 'nn_pred_cbis' { & "$NN\nnUNetv2_predict.exe" -i "$RAWD\imagesTs" -o 'D:\nnunet\pred_cbis' -d 501 -c 2d -f 0 -tr nnUNetTrainer_100epochs --save_probabilities } "$RLOG\nn_pred_cbis.log"
Step 'nn_pred_inb'  { & "$NN\nnUNetv2_predict.exe" -i "$RAWD\imagesTs_inbreast" -o 'D:\nnunet\pred_inbreast' -d 501 -c 2d -f 0 -tr nnUNetTrainer_100epochs --save_probabilities } "$RLOG\nn_pred_inb.log"
Step 'nn_maps' { Set-Location $REV; & "$NN\python.exe" -u nnunet_maps.py } "$RLOG\nn_maps.log"
E2EPair 'nnunet' 'D:/nnunet/P_cbis_test.npy' 'D:/nnunet/P_inbreast.npz'
Step 'stats_final' { Set-Location $REV; & $TF -u stats_revision.py } "$RLOG\stats_final.log"
Add-Content "$RLOG\queue_c.log" "$(Get-Date -Format s)  QUEUE C DONE"
