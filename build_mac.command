#!/bin/bash
# NeuronTracer: build the macOS app on a Mac. Double-click this file in Finder (or run it in Terminal).
# Needs: python3 from python.org (3.11, includes Tk) or a conda python 3.11 with tk. Everything goes under $BASE.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
BASE="$HOME/ntb_mac"; mkdir -p "$BASE"
PY=$(command -v python3.11 || command -v python3)
echo "=== Step 1 of 6: environment in $BASE/env (python: $PY) ==="
if [ ! -x "$BASE/env/bin/python" ]; then "$PY" -m venv "$BASE/env"; fi
source "$BASE/env/bin/activate"
python -c "import tkinter" || { echo "This python has no Tk. Install python 3.11 from python.org and run again."; exit 1; }
pip install -q --upgrade pip
pip install -q numpy==1.26.4 scipy==1.13.1 scikit-image==0.24.0 onnxruntime==1.18.1 pillow networkx pyinstaller pyinstaller-hooks-contrib
echo "=== Step 2 of 6: test with Python ==="
rm -rf "$BASE/test"; python src/app.py --cli sample/test_input.png "$BASE/test/python"
echo "=== Step 3 of 6: build the app (2 to 5 minutes) ==="
pyinstaller --noconfirm --distpath "$BASE/dist" --workpath "$BASE/build" NeuronTracer.spec
APP="$BASE/dist/NeuronTracer.app/Contents/MacOS/NeuronTracer"
[ -x "$APP" ] || { echo "Build failed."; exit 1; }
echo "=== Step 4 of 6: test the app ==="
"$APP" --cli "$HERE/sample/test_input.png" "$BASE/test/app"
echo "=== Step 5 of 6: compare Python and app results ==="
if cmp -s "$BASE/test/python/test_input_neuron_out/test_input_metrics.csv" "$BASE/test/app/test_input_neuron_out/test_input_metrics.csv"; then echo "SAME: the app gives the same result as Python."; else echo "DIFFERENT: send the two folders in $BASE/test to check."; fi
echo "=== Step 6 of 6: check the edit and review windows inside the app ==="
D="$BASE/dist/NeuronTracer.app/Contents/MacOS"; rm -f "$D/editor_selfcheck_ok.txt" "$D/annotator_selfcheck_ok.txt"
"$APP" --check-editor; "$APP" --check-annotator
[ -f "$D/editor_selfcheck_ok.txt" ] && echo "EDITOR OK" || echo "EDITOR FAILED"
[ -f "$D/annotator_selfcheck_ok.txt" ] && echo "REVIEW OK" || echo "REVIEW FAILED"
echo "Done. Success = SAME + EDITOR OK + REVIEW OK. The app is $BASE/dist/NeuronTracer.app (zip the whole .app to share)."
echo "If macOS refuses to open it on another Mac: right-click the app, Open, then Open again (unsigned app)."
