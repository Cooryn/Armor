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
            -DBUILD_TESTING=ON "-DOpenCV_DIR=$pythonBase/Library/cmake" "-DCMAKE_PREFIX_PATH=$pythonBase/Library"
        if ($LASTEXITCODE -ne 0) { throw 'Test configuration failed.' }
        cmake --build "$PSScriptRoot/build" --config Release --target detector_tests predictor_tests camera_gimbal_tests camera_gimbal pose_base predictor predictor_polar predictor_armor --parallel
        if ($LASTEXITCODE -ne 0) { throw 'Test build failed.' }
        ctest --test-dir "$PSScriptRoot/build" -C Release --output-on-failure `
            --output-log "$outputDir/cpp-tests.log"
        if ($LASTEXITCODE -ne 0) { throw 'C++ tests failed.' }
        $comparison = Join-Path $outputDir ('predictor-equivalence-' + [guid]::NewGuid().ToString('N'))
        & $pythonExe -B "$PSScriptRoot/verify_predictor_equivalence.py" --python-root $repoRoot `
            --bin-dir "$repoRoot" --work-dir $comparison
        if ($LASTEXITCODE -ne 0) { throw 'Predictor equivalence tests failed.' }
        $cameraSimulation = Join-Path $outputDir ('camera-simulation-' + [guid]::NewGuid().ToString('N'))
        & $pythonExe -B "$PSScriptRoot/camera_pipeline.py" --work-dir $cameraSimulation
        if ($LASTEXITCODE -ne 0) { throw 'Camera coordinate/control simulation failed.' }
    }
} finally {
    $env:TEMP = $oldTemp
    $env:TMP = $oldTmp
    $env:PATH = $oldPath
    Pop-Location
}
