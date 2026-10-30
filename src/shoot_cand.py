"""Open the editor on the cell-172 demo folder, go to the To-check item that carries the tracer's join candidate, screenshot,
press J, screenshot again. Runs on the Xvfb display :77."""
import sys, os, time
sys.path.insert(0, '/work/nvme/bfin/ypu2/neuron_morph/yuqi/app_v3_update/src')
import tkinter as tk
from PIL import ImageGrab
import editor
root = tk.Tk(); root.withdraw()
w = editor.Editor(root, folder='/work/nvme/bfin/ypu2/neuron_morph/yuqi/rules_round/editor_demo')
w.geometry('1600x1000+0+0'); w.update(); time.sleep(1.5); w.update()
cands = [i for i, it in enumerate(w.issues) if it.get('cand')]
print('issues', len(w.issues), 'with tracer candidate', len(cands), [w.issues[i]['label'] + ' | ' + w.issues[i].get('note', '') for i in cands])
if cands:
    w.go_check(cands[0]); w.update(); time.sleep(1.0); w.update()
    ImageGrab.grab(bbox=(0, 0, 1600, 1000), xdisplay=':77').save('shot_cand_before.png')
    w.join_current_candidate(); w.update(); time.sleep(1.0); w.update()
    ImageGrab.grab(bbox=(0, 0, 1600, 1000), xdisplay=':77').save('shot_cand_after.png')
    print('after join: loose issues', sum(1 for it in w.issues if it['kind'] == 'loose'), 'edits', len(w.edits), 'hint:', w.hint_var.get() if hasattr(w, 'hint_var') else '')
root.destroy(); print('done')
