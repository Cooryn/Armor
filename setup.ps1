$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$pythonBase = & python -c "import sys; print(sys.base_prefix)"
if (!(Test-Path '.venv\Scripts\python.exe')) {
    python -m venv --system-site-packages .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed.' }
}
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed.' }
cmake -S . -B build -G 'Visual Studio 17 2022' -A x64 -DBUILD_TESTING=OFF "-DOpenCV_DIR=$pythonBase/Library/cmake" "-DCMAKE_PREFIX_PATH=$pythonBase/Library"
if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed.' }
cmake --build build --config Release --parallel
if ($LASTEXITCODE -ne 0) { throw 'C++ build failed.' }
