# Revision job queue A - strictly sequential GPU jobs (the laptop cannot run two at once).
$TF    = 'D:\pipeline2_mammogram\pipeline2_mammogram\.venv_tfdml\Scripts\python.exe'
$TORCH = 'D:\pipeline2_mammogram\pipeline2_mammogram\mammo_clip_dlora\.venv\Scripts\python.exe'
$REV = 'G:\My Drive\VS_CC\revision'; $E2E = 'G:\My Drive\VS_CC\e2e'
$LOG = 'G:\My Drive\VS_CC\models\revision'; $ELOG = 'G:\My Drive\VS_CC\models\e2e'
$JIT = 'G:/My Drive/VS_CC/models/cbis_roi_seg_jitter_s2021.weights.h5'
$noise = 'tensorflow/c/logging.cc|DirectML device|dxgi|d3d12|Successfully opened|oneDNN|cpu_feature_guard|rebuild TensorFlow'
$env:HF_HUB_OFFLINE = '1'; $env:NO_ALBUMENTATIONS_UPDATE = '1'

function Step($name, $py, $dir, $script, $argz, $log) {
    Add-Content "$LOG\queue_a.log" "$(Get-Date -Format s)  START $name"
    Set-Location $dir
    & $py -u $script @argz 2>&1 | Where-Object { $_ -notmatch $noise } | Out-File -Encoding utf8 $log
    Add-Content "$LOG\queue_a.log" "$(Get-Date -Format s)  END   $name (exit $LASTEXITCODE)"
}

# wait for the running weakly-supervised re-scoring to finish
while (-not ((Get-Content "$LOG\ws_reeval.log" -Raw -ErrorAction SilentlyContinue) -match 'SUMMARY|Traceback')) { Start-Sleep 30 }
Start-Sleep 20

Step 'mc_features'   $TORCH $REV 'mc_features.py'      @()                                   "$LOG\mc_features.log"
Step 'mc_heads'      $TORCH $REV 'mc_heads.py'         @()                                   "$LOG\mc_heads.log"
Step 'mias_cv'       $TF    $REV 'mias_patientwise.py' @()                                   "$LOG\mias_cv.log"
Step 'jitter_s2021'  $TF    $REV 'jitter_roi.py'       @('--seed', '2021')                   "$LOG\jitter_s2021.log"
Step 'cbis_pub'      $TF    $E2E 'e2e_cbis.py'         @('--tag', 'pub')                     "$ELOG\cbis_e2e_pub.log"
Step 'inbreast_pub'  $TF    $E2E 'e2e_inbreast.py'     @('--tag', 'pub')                     "$ELOG\inbreast_e2e_pub.log"
Step 'cbis_jit'      $TF    $E2E 'e2e_cbis.py'         @('--roi', $JIT, '--tag', 'jit')      "$ELOG\cbis_e2e_jit.log"
Step 'inbreast_jit'  $TF    $E2E 'e2e_inbreast.py'     @('--roi', $JIT, '--tag', 'jit')      "$ELOG\inbreast_e2e_jit.log"
Add-Content "$LOG\queue_a.log" "$(Get-Date -Format s)  QUEUE A DONE"
