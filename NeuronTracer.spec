# -*- mode: python ; coding: utf-8 -*-
import os, sys
from PyInstaller.utils.hooks import collect_all

# paths are relative to this spec file and to the active conda env, so it builds on any machine
HERE = os.path.dirname(os.path.abspath(SPEC))
datas = [(os.path.join(HERE, 'src', 'neurite_model.onnx'), '.'), (os.path.join(HERE, 'src', 'cnn_crossing.onnx'), '.')]
binaries = [(os.path.join(sys.prefix, 'Library', 'bin', '*.dll'), '.')] if sys.platform == 'win32' else []
hiddenimports = ['PIL._tkinter_finder', 'theme', 'annotator', 'editor', 'test_editor']   # windows imported lazily inside functions
tmp_ret = collect_all('onnxruntime')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('skimage')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('scipy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('numpy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    [os.path.join(HERE, 'src', 'app.py')],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='NeuronTracer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='NeuronTracer',
)
if sys.platform == 'darwin':
    app = BUNDLE(coll, name='NeuronTracer.app', icon=None, bundle_identifier='edu.rutgers.panglab.neurontracer',
                 info_plist={'NSHighResolutionCapable': True, 'LSMinimumSystemVersion': '12.0'})
