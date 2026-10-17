# Revision job queue D - VinDr-Mammo (second external cohort, original DICOMs from PhysioNet; Reviewer 1 #2, Reviewer 2 #4).
# Waits for the download (e2e/download_vindr_zip.py; restarts it if it stopped before finishing), then runs, strictly
# sequentially: detector inputs -> the three detector architectures (+ two extra Attention U-Net seeds, + the min-max
# sensitivity variant) with the box-level end-to-end analysis -> GMIC and GLAM -> the pipeline-aware delineator.
$TF = 'D:\pipeline2_mammogram\pipeline2_mammogram\.venv_tfdml\Scripts\python.exe'
$TORCH = 'D:\pipeline2_mammogram\pipeline2_mammogram\mammo_clip_dlora\.venv\Scripts\python.exe'
$NN = 'D:\venvs\nnunet\Scripts'
$REV = 'G:\My Drive\VS_CC\revision'; $E2E = 'G:\My Drive\VS_CC\e2e'
$RLOG = 'G:\My Drive\VS_CC\models\revision'; $ELOG = 'G:\My Drive\VS_CC\models\e2e'; $M = 'G:/My Drive/VS_CC/models'
$VD = 'D:\vindr_dicom'; $RAWD = 'D:\nnunet\nnUNet_raw\Dataset501_CBISFull'
$env:nnUNet_raw = 'D:\nnunet\nnUNet_raw'; $env:nnUNet_preprocessed = 'D:\nnunet\nnUNet_preprocessed'
$env:nnUNet_results = 'D:\nnunet\nnUNet_results'; $env:nnUNet_def_n_proc = '2'; $env:nnUNet_compile = 'false'
$env:DML_VISIBLE_DEVICES = '0'; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'; $env:HF_HUB_OFFLINE = '1'
$noise = 'tensorflow/c/logging.cc|DirectML device|dxgi|d3d12|Successfully opened|oneDNN|cpu_feature_guard|rebuild TensorFlow|UserWarning|warnings.warn|FutureWarning'

function Step([string]$name, [scriptblock]$body, [string]$outfile) {
    Add-Content "$RLOG\queue_d.log" "$(Get-Date -Format s)  START $name"
    & $body 2>&1 | Where-Object { $_ -notmatch $noise } | Out-File -Encoding utf8 $outfile
    Add-Content "$RLOG\queue_d.log" "$(Get-Date -Format s)  END   $name (exit $LASTEXITCODE)"
}
function Vindr([string]$tag, [string[]]$extra) {
    Step "vindr_$tag" { Set-Location $E2E; & $TF -u e2e_vindr_dicom.py --tag $tag @extra } "$ELOG\vindrdcm_e2e_$tag.log"
}

# ---- 1. the download must be complete (the script is resumable; restart it if it is no longer running)
Add-Content "$RLOG\queue_d.log" "$(Get-Date -Format s)  waiting for the VinDr-Mammo download"
while (-not ((Get-Content "$VD\download.log" -Raw -ErrorAction SilentlyContinue) -match 'ALL DONE')) {
    $p = Get-Content "$VD\download.pid" -ErrorAction SilentlyContinue
    if (-not ($p -and (Get-Process -Id $p -ErrorAction SilentlyContinue))) {
        Add-Content "$RLOG\queue_d.log" "$(Get-Date -Format s)  download not running and not finished - restarting it"
        Start-Process -FilePath $TF -ArgumentList '-u', '"G:\My Drive\VS_CC\e2e\download_vindr_zip.py"' -WindowStyle Hidden `
            -RedirectStandardOutput "$VD\download.log" -RedirectStandardError "$VD\download_err.log"
        Start-Sleep 180
    }
    Start-Sleep 120
}
Add-Content "$RLOG\queue_d.log" "$(Get-Date -Format s)  download complete"

# ---- 2. detector inputs (DICOM display transform; min-max variant for the sensitivity analysis)
Step 'vindr_prep' { Set-Location $REV; & $TORCH -u vindr_prep.py } "$RLOG\vindr_prep.log"

# ---- 3. Attention U-Net v3: published model with the box-level end-to-end analysis; min-max variant and seeds 1, 2: detection only
Vindr 'pub' @('--roi', "$M/cbis_roi_seg.weights.h5")
Vindr 'minmax' @('--minmax')
Vindr 's1' @('--det', "$M/cbis_attention_unet_v3_s1.weights.h5")
Vindr 's2' @('--det', "$M/cbis_attention_unet_v3_s2.weights.h5")

# ---- 4. SegFormer and nnU-Net maps -> the same evaluation
Step 'segformer_vindr' { Set-Location $REV; & "$NN\python.exe" -u segformer_predict.py } "$RLOG\segformer_vindr.log"
Vindr 'segformer_pub' @('--pmap', 'D:/segformer/mit_b2_s0/P_vindr.npz', '--roi', "$M/cbis_roi_seg.weights.h5")
if (Test-Path 'D:\nnunet\pred_vindr') { Remove-Item -Recurse -Force 'D:\nnunet\pred_vindr' -Confirm:$false }
Step 'nn_pred_vindr' { & "$NN\nnUNetv2_predict.exe" -i "$RAWD\imagesTs_vindr" -o 'D:\nnunet\pred_vindr' -d 501 -c 2d -f 0 -tr nnUNetTrainer_100epochs --save_probabilities } "$RLOG\nn_pred_vindr.log"
Step 'nn_maps_vindr' { Set-Location $REV; & "$NN\python.exe" -u nnunet_maps.py vindr } "$RLOG\nn_maps_vindr.log"
Vindr 'nnunet_pub' @('--pmap', 'D:/nnunet/P_vindr.npz', '--roi', "$M/cbis_roi_seg.weights.h5")

# ---- 5. dedicated weakly-supervised localizers
Step 'wsol_gmic_vindr' { Set-Location $REV; & $TORCH -u wsol_nyu.py --model gmic --datasets vindr } "$RLOG\wsol_gmic_vindr.log"
Step 'wsol_glam_vindr' { Set-Location $REV; & $TORCH -u wsol_nyu.py --model glam --datasets vindr } "$RLOG\wsol_glam_vindr.log"

# ---- 6. pipeline-aware delineator with the published detector
Vindr 'jit' @('--roi', "$M/cbis_roi_seg_jitter_s2021.weights.h5")
Add-Content "$RLOG\queue_d.log" "$(Get-Date -Format s)  QUEUE D DONE"
