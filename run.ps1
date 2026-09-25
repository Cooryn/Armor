param(
    [ValidateSet('image', 'video')][string]$Mode = 'video',
    [ValidateSet('red', 'blue')][string]$Color = 'red',
    [string]$InputFile = 'video_2.avi'
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$pythonBase = & python -c "import sys; print(sys.base_prefix)"
$env:PATH = "$pythonBase\Library\bin;$env:PATH"
& "$PSScriptRoot\Armor.exe" $Mode $Color $InputFile
if ($LASTEXITCODE -ne 0) { throw 'Armor processing failed.' }
