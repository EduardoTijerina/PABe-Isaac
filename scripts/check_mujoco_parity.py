"""Parity check: the backend-based episode (this project, MuJoCo backend) must reproduce CBF-repro's original
episode bit for bit. Run with CBF-repro's venv: PABE_ROBOT=<g1|revo2> python scripts/check_mujoco_parity.py [models.pkl]"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, "/mnt/data/CBF-repro")
from pabe_cbf.core import METHODS as M_old                     # noqa: E402
from pabe_cbf.g1.episode import G1Episode as Old               # noqa: E402
from pabe_cbf.g1.geometry import G1_GEOM as GEOM_old           # noqa: E402
from pabe_cbf.scenarios import Participant as Pa_old, Scenario as Sc_old, FAMILIES_ALL  # noqa: E402
from pabe.core import METHODS as M_new                         # noqa: E402
from pabe.episode import G1Episode as New                      # noqa: E402
from pabe.geometry import G1_GEOM as GEOM_new                  # noqa: E402
from pabe.scenarios import Participant as Pa_new, Scenario as Sc_new  # noqa: E402
from backends.mujoco import MujocoBackend                       # noqa: E402

models = None
if len(sys.argv) > 1:
    import pickle
    models = pickle.load(open(sys.argv[1], "rb"))
methods = ["B0", "B1"] + (["B2", "B9"] if models is not None else [])
be = MujocoBackend()
n = bad = 0
for fam in FAMILIES_ALL:
    for pid in (4000, 4007):
        for meth in methods:
            o = Old(M_old[meth], Sc_old(fam, Pa_old(pid, shift=pid == 4007, geom=GEOM_old), 0), models, record=True)
            ro, _, _ = o.run()
            e = New(M_new[meth], Sc_new(fam, Pa_new(pid, shift=pid == 4007, geom=GEOM_new), 0), models, be, record=True)
            rn, _, _ = e.run()
            same = all((ro[k] == rn[k]) or (isinstance(ro[k], float) and np.isnan(ro[k]) and np.isnan(rn[k]))
                       for k in ro if k != "robot") \
                and len(o.record) == len(e.record) \
                and all(np.array_equal(a["qpos"], b["qpos"]) and np.array_equal(a["u"], b["u"])
                        for a, b in zip(o.record, e.record))
            n += 1; bad += not same
            if not same:
                print("DIFF", fam, pid, meth, ro["R_reason"], rn["R_reason"], len(o.record), len(e.record))
print(f"{os.environ.get('PABE_ROBOT')}: {n - bad}/{n} episodes bit-identical ({', '.join(methods)})")
