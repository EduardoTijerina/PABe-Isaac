"""Compara la dinámica del brazo en PhysX con la de MuJoCo (results/mujoco_ref.json) en las mismas posturas:
matriz de masas, sesgo (gravedad en reposo), jacobianos de la palma y del codo (hipótesis origen / CdM), y
las ganancias reales de los servos del brazo. Uso: ~/isaac_env/bin/python -u scripts/probe_isaac_vs_mujoco.py"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "vendor")]
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})
import numpy as np  # noqa: E402

from backends.isaac import IsaacBackend  # noqa: E402

ref = json.load(open(os.path.join(ROOT, "results", "mujoco_ref.json")))
be = IsaacBackend()
be.reset()
kp, kd = (np.asarray(x)[0] for x in be.art.get_gains())
print(f"[cmp] arm gains kp {kp[be.arm].round(3)} kd {kd[be.arm].round(3)}; a leg kp {kp[0]:.1f} kd {kd[0]:.2f}")
sv = getattr(be.art, "_physics_sim_view", None)
print(f"[cmp] kinematic update available: {sv is not None and hasattr(sv, 'update_articulations_kinematic')}")
for name in ("stand", "bent"):
    r = ref[name]
    q = be.q_stand.copy(); q[be.arm] = r["q"]
    be.art.set_joint_positions(q[None]); be.art.set_joint_velocities(np.zeros((1, be.n_dof)))
    if sv is not None and hasattr(sv, "update_articulations_kinematic"):
        sv.update_articulations_kinematic()
    be._read()
    rel = lambda a, b: np.linalg.norm(np.asarray(a) - np.asarray(b)) / max(np.linalg.norm(b), 1e-9)
    print(f"[cmp] {name}: |q_isaac - q| = {np.abs(be.arm_q() - r['q']).max():.2e}  palm err "
          f"{np.linalg.norm(be.palm() - r['palm']) * 1000:.2f} mm  elbow err "
          f"{np.linalg.norm(be._P[be.cap_bodies['upperarm'][1]] - r['elbow']) * 1000:.2f} mm")
    print(f"[cmp] {name}: M rel err {rel(be.arm_M(), r['M']):.3f}  bias rel err {rel(be.arm_bias(), r['bias']):.3f}"
          f"  (isaac bias {np.round(be.arm_bias(), 3)})")
    for at_com in (False, True):
        be.jac_at_com = at_com
        k = be.kin()
        print(f"[cmp] {name} jac {'CoM' if at_com else 'origin'}: Jp rel err {rel(k['Jp'], r['Jp']):.4f}  "
              f"Jelbow rel err {rel(k['caps']['upperarm'][4], r['Jelbow']):.4f}  Jr rel err {rel(k['Jr'], r['Jr']):.4f}")
    be.jac_at_com = True
print(f"[cmp] total mass now {np.asarray(be.art._physics_view.get_masses())[0].sum():.2f} kg (MuJoCo 33.78)")
for label, use in (("bias", True), ("zero", False)):
    be.reset(); q0 = be.arm_q(); t0 = be.world.current_time
    for _ in range(250):
        be.set_arm_torque(np.clip(be.arm_bias(), -be.tau_lim, be.tau_lim) if use else np.zeros(7))
        be.step()
    print(f"[cmp] drop tau={label}: sim time {be.world.current_time - t0:.3f} s, arm change {np.round(be.arm_q() - q0, 3)}"
          f"  (MuJoCo {np.round(ref['drop'][label], 3)})")
app.close()
