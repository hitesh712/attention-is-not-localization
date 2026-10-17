# Revision job queue B - runs after queue A; strictly sequential GPU jobs.
$TF  = 'D:\pipeline2_mammogram\pipeline2_mammogram\.venv_tfdml\Scripts\python.exe'
$REV = 'G:\My Drive\VS_CC\revision'; $E2E = 'G:\My Drive\VS_CC\e2e'
$RLOG = 'G:\My Drive\VS_CC\models\revision'; $ELOG = 'G:\My Drive\VS_CC\models\e2e'; $MOD = 'G:/My Drive/VS_CC/models'
$noise = 'tensorflow/c/logging.cc|DirectML device|dxgi|d3d12|Successfully opened|oneDNN|cpu_feature_guard|rebuild TensorFlow'

function Step([string]$name, [string]$py, [string]$dir, [string]$script, [string[]]$argz, [string]$outfile) {
    Add-Content "$RLOG\queue_b.log" "$(Get-Date -Format s)  START $name"
    Set-Location $dir
    & $py -u $script @argz 2>&1 | Where-Object { $_ -notmatch $noise } | Out-File -Encoding utf8 $outfile
    Add-Content "$RLOG\queue_b.log" "$(Get-Date -Format s)  END   $name (exit $LASTEXITCODE)"
}

while (-not ((Get-Content "$RLOG\queue_a.log" -Raw -ErrorAction SilentlyContinue) -match 'QUEUE A DONE')) { Start-Sleep 60 }

Step 'ws_reeval_fixed' $TF $REV 'ws_reeval.py'   @()                   "$RLOG\ws_reeval.log"
foreach ($s in 1, 2) {
    Step "detector_s$s" $TF $REV 'detector_v3.py' @('--seed', "$s")   "$RLOG\detector_s$s.log"
    Step "jitter_s$s"   $TF $REV 'jitter_roi.py'  @('--seed', "$s")   "$RLOG\jitter_s$s.log"
    $det = "$MOD/cbis_attention_unet_v3_s$s.weights.h5"; $roi = "$MOD/cbis_roi_seg_jitter_s$s.weights.h5"
    Step "cbis_s$s"     $TF $E2E 'e2e_cbis.py'     @('--det', $det, '--roi', $roi, '--tag', "s$s") "$ELOG\cbis_e2e_s$s.log"
    Step "inbreast_s$s" $TF $E2E 'e2e_inbreast.py' @('--det', $det, '--roi', $roi, '--tag', "s$s") "$ELOG\inbreast_e2e_s$s.log"
}
Step 'stats' $TF $REV 'stats_revision.py' @() "$RLOG\stats_revision.log"
Add-Content "$RLOG\queue_b.log" "$(Get-Date -Format s)  QUEUE B DONE"
