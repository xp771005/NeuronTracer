@echo off
REM NeuronTracer v3: build the exe on this computer. Run from Anaconda Prompt.
REM Everything (environment, download caches, build output, test results) goes under BASE, not C:.
set BASE=D:\ntb
set ENV=%BASE%\env
set CONDA_PKGS_DIRS=%BASE%\conda_pkgs
set PIP_CACHE_DIR=%BASE%\pip_cache
cd /d "%~dp0"
if not exist "%BASE%" mkdir "%BASE%"

call conda --version >nul 2>&1
if errorlevel 1 (
  echo Conda not found. Open "Anaconda Prompt" from the Start menu and run this file again from there.
  pause
  exit /b 1
)

echo.
echo === Step 1 of 6: build environment in %ENV% ===
if not exist "%ENV%\python.exe" (
  call conda create -p "%ENV%" -c conda-forge python=3.11 tk -y
)
call conda activate "%ENV%"
if errorlevel 1 ( echo Could not activate %ENV%. & pause & exit /b 1 )
pip install numpy==1.26.4 scipy==1.13.1 scikit-image==0.24.0 onnxruntime==1.18.1 pillow networkx pyinstaller pyinstaller-hooks-contrib
if errorlevel 1 ( echo pip install failed. & pause & exit /b 1 )

echo.
echo === Step 2 of 6: test with Python ===
if exist "%BASE%\test" rmdir /s /q "%BASE%\test"
python src\app.py --cli sample\test_input.png "%BASE%\test\python"
if errorlevel 1 ( echo Python test failed. & pause & exit /b 1 )

echo.
echo === Step 3 of 6: build the exe (2 to 5 minutes) ===
pyinstaller --noconfirm --distpath "%BASE%\dist" --workpath "%BASE%\build" NeuronTracer.spec
if not exist "%BASE%\dist\NeuronTracer\NeuronTracer.exe" ( echo Build failed. & pause & exit /b 1 )

echo.
echo === Step 4 of 6: test the exe ===
start /wait "" "%BASE%\dist\NeuronTracer\NeuronTracer.exe" --cli "%~dp0sample\test_input.png" "%BASE%\test\exe"

echo.
echo === Step 5 of 6: compare Python and exe results ===
fc "%BASE%\test\python\test_input_neuron_out\test_input_metrics.csv" "%BASE%\test\exe\test_input_neuron_out\test_input_metrics.csv" >nul
if errorlevel 1 (
  echo DIFFERENT: the exe and Python results do not match. Send the two folders in %BASE%\test to check.
) else (
  echo SAME: the exe gives the same result as Python.
)

echo.
echo === Step 6 of 6: check that the edit and review windows work inside the exe ===
del /q "%BASE%\dist\NeuronTracer\editor_selfcheck_ok.txt" "%BASE%\dist\NeuronTracer\annotator_selfcheck_ok.txt" 2>nul
start /wait "" "%BASE%\dist\NeuronTracer\NeuronTracer.exe" --check-editor
start /wait "" "%BASE%\dist\NeuronTracer\NeuronTracer.exe" --check-annotator
if exist "%BASE%\dist\NeuronTracer\editor_selfcheck_ok.txt" (echo EDITOR OK) else (echo EDITOR FAILED: the Edit tracing window does not work in the exe)
if exist "%BASE%\dist\NeuronTracer\annotator_selfcheck_ok.txt" (echo REVIEW OK) else (echo REVIEW FAILED: the review window does not work in the exe)
del /q "%BASE%\dist\NeuronTracer\editor_selfcheck_ok.txt" "%BASE%\dist\NeuronTracer\annotator_selfcheck_ok.txt" 2>nul
echo.
echo Done. Success = SAME + EDITOR OK + REVIEW OK above.
echo The new software is in %BASE%\dist\NeuronTracer\  ^(run NeuronTracer.exe; copy the whole folder^)
pause
