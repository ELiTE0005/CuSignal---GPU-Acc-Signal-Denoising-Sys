# Launch MSTAR pipeline HTML in browser and run training
param(
    [string]$MASTARRoot = "/data/MSTAR",
    [int]$Epochs = 40,
    [string]$OutputPath = "checkpoints/mstar_cnn.pt"
)

# Get the project root (parent of scripts folder)
$projectRoot = Split-Path -Parent $PSScriptRoot
$htmlFile = Join-Path $projectRoot "mstar_pipeline.html"

# Convert to file:// URI for browser
$htmlUri = "file:///" + $htmlFile.Replace('\', '/').Replace(':', '')
$htmlUri = $htmlUri -replace '///', 'C:/'

Write-Host "🚀 Opening MSTAR Pipeline Dashboard..."
Write-Host "📄 HTML: $htmlUri"

# Open in default browser
if ($PSVersionTable.PSVersion.Major -ge 6) {
    # PowerShell 7+
    Start-Process $htmlFile
} else {
    # PowerShell 5.1
    Invoke-Item $htmlFile
}

Write-Host ""
Write-Host "🐳 Starting Docker training..."
Write-Host "Command: docker exec cusignal-cusignal-dev-1 bash -lc `"cd /app/cusignal_project && python scripts/train_mstar_cnn.py --mstar-root $MASTARRoot --epochs $Epochs --out $OutputPath`""
Write-Host ""

# Run Docker training
docker exec cusignal-cusignal-dev-1 bash -lc "cd /app/cusignal_project && python scripts/train_mstar_cnn.py --mstar-root $MASTARRoot --epochs $Epochs --out $OutputPath"

Write-Host ""
Write-Host "✅ Training complete!"
