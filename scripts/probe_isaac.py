"""Comprobaciones del backend Isaac antes de correr episodios:
  1. estructura: GDL, cuerpos, forma del jacobiano, base fija, punto de la palma (frente a MuJoCo)
  2. jacobiano de PhysX: ¿filas lineales en el origen del eslabón o en su centro de masas? (diferencias finitas)
  3. signo del sesgo: con tau = sesgo el brazo debe quedarse quieto; con tau = 0 debe caer
Uso: ~/isaac_env/bin/python scripts/probe_isaac.py
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "vendor")]
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})
import numpy as np  # noqa: E402

from backends.isaac import IsaacBackend  # noqa: E402

be = IsaacBackend()
be.reset()
print(f"[probe] dof={be.n_dof} bodies={len(be.body_names)} jac row_off={be.row_off} col_off={be.col_off}")
print(f"[probe] finger dofs present: {[be.dof_names[i] for i in be.fing]}")
mimic = [n for n in be.dof_names if 'distal' in n and n.startswith('right')]
print(f"[probe] right distal dofs: {mimic}")
pel = be.body_names.index("pelvis") if "pelvis" in be.body_names else 0
print(f"[probe] pelvis at {be._P[pel].round(4)} (expected [0 0 0.793])")
print(f"[probe] palm point at stand {be.palm().round(4)}  (MuJoCo revo2: [0.067 -0.263 0.674])")
print(f"[probe] torso at {be._P[be.b_torso].round(4)}  arm limits lo {be.arm_lo.round(2)} hi {be.arm_hi.round(2)}")

# 2. jacobiano: desplazamiento de un paso con velocidad articular conocida (sesgo compensado)
be._read(); com = be._coms()
J = be._jac()
errs = {"origin": [], "com": []}
q0 = be.arm_q()
for j in range(7):
    be.reset()
    qd = np.zeros(be.n_dof); qd[be.arm[j]] = 0.2
    be.art.set_joint_velocities(qd[None])
    be._read(); P0 = be._P.copy(); J = be._jac(); com = be._coms()
    be.set_arm_torque(np.clip(be.arm_bias(), -be.tau_lim, be.tau_lim))
    be.step()
    for b in (be.b_palm, be.cap_bodies["upperarm"][1]):
        v = (be._P[b] - P0[b]) / 0.002
        row = J[b + be.row_off]
        Jl, Ja = row[:3, be.cols], row[3:, be.cols]
        u = np.zeros(7); u[j] = 0.2
        v_origin = Jl @ u
        v_com = (Jl - np.array([[0, -c[2], c[1]], [c[2], 0, -c[0]], [-c[1], c[0], 0]]) @ Ja) @ u \
            if False else None
        c = P0[b] - com[b]                       # origen respecto del CdM
        S = np.array([[0, -c[2], c[1]], [c[2], 0, -c[0]], [-c[1], c[0], 0]])
        v_com = (Jl - S @ Ja) @ u                # si Jl es del CdM, el origen se mueve así
        errs["origin"].append(np.linalg.norm(v - v_origin)); errs["com"].append(np.linalg.norm(v - v_com))
print(f"[probe] jacobian origin-hypothesis err {np.mean(errs['origin']):.4f} m/s, com-hypothesis err {np.mean(errs['com']):.4f} m/s")

# 3. sesgo: 0,5 s con tau = sesgo frente a tau = 0
for label, use in (("tau = bias", True), ("tau = 0", False)):
    be.reset(); q0 = be.arm_q()
    for _ in range(250):
        be.set_arm_torque(np.clip(be.arm_bias(), -be.tau_lim, be.tau_lim) if use else np.zeros(7))
        be.step()
    print(f"[probe] {label}: arm drift after 0.5 s = {np.abs(be.arm_q() - q0).max():.4f} rad")
app.close()
