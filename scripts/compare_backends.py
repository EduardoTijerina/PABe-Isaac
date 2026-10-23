"""Compara episodio a episodio dos ficheros de scripts/run_episodes.py (p. ej. MuJoCo frente a Isaac Sim):
desenlace, fuerza pico, tiempos de contacto y fin, y desviación de la trayectoria de la palma.
Uso: python scripts/compare_backends.py results/cmp_mujoco.json results/cmp_isaac.json"""
import json
import sys

import numpy as np

A, B = (json.load(open(p)) for p in sys.argv[1:3])
key = lambda r: (r["family"], r["pid"], r["ep"], r["method"])
B = {key(r): r for r in B}
rows, same = [], 0
print(f"{'episode':<38} {'outcome A':>10} {'outcome B':>10} {'peakF A':>7} {'peakF B':>7} {'t_C A':>6} {'t_C B':>6} "
      f"{'T A':>6} {'T B':>6} {'palm dev (cm)':>13}")
for a in A:
    b = B.get(key(a))
    if b is None:
        continue
    pa, pb = np.array(a["palm"]), np.array(b["palm"])
    n = min(len(pa), len(pb))
    dev = np.linalg.norm(pa[:n] - pb[:n], axis=1)
    ok = a["R_reason"] == b["R_reason"] and a["V_any"] == b["V_any"]
    same += ok
    f = lambda x: "-" if x is None else f"{x:.2f}"
    print(f"{'/'.join(map(str, key(a))):<38} {str(a['R_reason']):>10} {str(b['R_reason']):>10} {a['peak_F']:>7.1f} "
          f"{b['peak_F']:>7.1f} {f(a['t_C']):>6} {f(b['t_C']):>6} {a['T']:>6.2f} {b['T']:>6.2f} "
          f"{np.median(dev) * 100:>6.1f} / {dev.max() * 100:<5.1f}{'' if ok else '   <- differs'}")
    rows.append((a, b, dev))
print(f"\nsame outcome and violation flag: {same}/{len(rows)}")
dF = [abs(a["peak_F"] - b["peak_F"]) for a, b, _ in rows if a["R_reason"] == b["R_reason"]]
dT = [abs(a["T"] - b["T"]) for a, b, _ in rows if a["R_reason"] == b["R_reason"]]
print(f"where outcomes agree: |peak F| diff median {np.median(dF):.1f} N (max {max(dF):.1f}); "
      f"|T| diff median {np.median(dT):.2f} s (max {max(dT):.2f})")
print(f"palm trajectory deviation: median over episodes {np.median([np.median(d) for _, _, d in rows]) * 100:.1f} cm, "
      f"worst {max(d.max() for _, _, d in rows) * 100:.1f} cm")
print(f"wall time per episode: A {np.mean([a['wall'] for a, _, _ in rows]):.1f} s, B {np.mean([b['wall'] for _, b, _ in rows]):.1f} s")
