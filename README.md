# NeuronTracer

Neurite tracing for single filled neurons (Pang lab, Rutgers). Python 3.11 + Tk + ONNX Runtime.

## Builds
Every push to `main` builds the program on GitHub's machines for macOS (Apple silicon and Intel) and Windows,
checks that the built program gives the same result as Python on `sample/test_input.png` (SAME), and runs the
edit and review window self-checks. Download the zips from the Actions run (Artifacts) or, for a tag `v*`, from Releases.

The segmentation model `src/neurite_model.onnx` (237 MB) is not in the repository. It is attached once to a release
named `model`; the workflow downloads it from there. To replace the model, upload the new file to that release.

## Build by hand
Windows: `build_v3.bat` (Anaconda Prompt). macOS: double-click `build_mac.command` (needs python.org Python 3.11).

## Run from source
```
pip install numpy==1.26.4 scipy==1.13.1 scikit-image==0.24.0 onnxruntime==1.18.1 pillow networkx
python src/app.py
```
