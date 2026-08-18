# -*- coding: utf-8 -*-
"""Execute the exploration notebook headlessly and dump every figure to PNG.

Used to verify the notebook actually runs inside the container and that the
3D-label / image projection lines up before handing it to the user.
"""
import json
import os
import sys
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NB = sys.argv[1] if len(sys.argv) > 1 else "/workspace/VAD/explore_etri.ipynb"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/workspace/VAD/work_dirs/nb_out"
os.makedirs(OUT, exist_ok=True)

_fig_n = [0]
_orig_show = plt.show


def _show(*a, **k):
    for num in plt.get_fignums():
        _fig_n[0] += 1
        path = os.path.join(OUT, "fig_%02d.png" % _fig_n[0])
        plt.figure(num).savefig(path, bbox_inches="tight", dpi=110)
        print("  [saved] %s" % path)
    plt.close("all")


plt.show = _show

ns = {"__name__": "__main__", "display": print}
cells = json.load(open(NB, encoding="utf-8"))["cells"]
code_cells = [c for c in cells if c["cell_type"] == "code"]

for i, c in enumerate(code_cells):
    src = c["source"]
    src = "\n".join(l for l in src.split("\n") if not l.strip().startswith("%"))
    print("=" * 60)
    print("### code cell %d/%d" % (i + 1, len(code_cells)))
    try:
        exec(compile(src, "<cell%d>" % i, "exec"), ns)
    except Exception:
        print("!!! FAILED")
        traceback.print_exc()
        sys.exit(1)

print("=" * 60)
print("ALL CELLS OK - %d figures written to %s" % (_fig_n[0], OUT))
