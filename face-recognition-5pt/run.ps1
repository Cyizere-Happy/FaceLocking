# run.ps1 - one-shot launcher for the Falcon Eye / Eagle Eye recognition scanner.
# Usage:  .\run.ps1            (uses external camera, index 1)
#         .\run.ps1 -CamIndex 0   (built-in camera instead)

param(
    [int]$CamIndex = 1
)

$env:FALCON_CAM_INDEX = "$CamIndex"
Write-Host "Using camera index $CamIndex (set FALCON_CAM_INDEX)"

py -m src.recognize
