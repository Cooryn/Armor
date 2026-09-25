param([switch]$PythonOnly)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$outputDir = Join-Path $PSScriptRoot 'outputs'
$tempDir = Join-Path $outputDir 'tmp'
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null
$pythonExe = Join-Path $repoRoot '.venv\Scripts\python.exe'
$oldTemp = $env:TEMP
$oldTmp = $env:TMP
$oldPath = $env:PATH
Push-Location $repoRoot
try {
    $env:TEMP = $tempDir
    $env:TMP = $tempDir
    & $pythonExe -B "$PSScriptRoot/run_python.py" |
        Tee-Object -FilePath (Join-Path $outputDir 'python-tests.log')
    $testExitCode = $LASTEXITCODE
    if ($testExitCode -ne 0) { throw 'Python tests failed.' }
    if (!$PythonOnly) {
        $pythonBase = & $pythonExe -c 'import sys; print(sys.base_prefix)'
        $env:PATH = "$pythonBase\Library\bin;$env:PATH"
        cmake -S $repoRoot -B "$PSScriptRoot/build" -G 'Visual Studio 17 2022' -A x64 `
            -DBUILD_TESTING=ON "-DOpenCV_DIR=$pythonBase/Library/cmake"
        if ($LASTEXITCODE -ne 0) { throw 'Test configuration failed.' }
        cmake --build "$PSScriptRoot/build" --config Release --target detector_tests --parallel
        if ($LASTEXITCODE -ne 0) { throw 'Test build failed.' }
        ctest --test-dir "$PSScriptRoot/build" -C Release --output-on-failure `
            --output-log "$outputDir/cpp-tests.log"
        if ($LASTEXITCODE -ne 0) { throw 'C++ tests failed.' }
    }
} finally {
    $env:TEMP = $oldTemp
    $env:TMP = $oldTmp
    $env:PATH = $oldPath
    Pop-Location
}
