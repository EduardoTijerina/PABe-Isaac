"""¿Se aplican los pares del brazo? Caída libre (tau = 0) desde el inicio y respuesta a un par constante en el codo,
frente a la aceleración esperada tau / M. Uso: ~/isaac_env/bin/python -u scripts/probe_effort.py"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "vendor")]
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})
import numpy as np  # noqa: E402

from backends.isaac import IsaacBackend  # noqa: E402

be = IsaacBackend()
art = be.art
for label in ("zero (fresh)", "elbow +1 Nm over bias"):
    be.reset(); q0 = be.arm_q()
    print(f"[eff] {label}: control mode / dof props:", getattr(art, "_dof_types", None))
    for s in range(100):
        tau = np.zeros(7)
        if label.startswith("elbow"):
            tau = np.clip(be.arm_bias(), -be.tau_lim, be.tau_lim); tau[3] += 1.0
        be.set_arm_torque(tau)
        if s == 0:
            print(f"[eff]   applied efforts read back: {np.round(np.asarray(art.get_applied_joint_efforts())[0, be.arm], 3)}")
        be.step()
    dq = be.arm_q() - q0
    print(f"[eff]   after 0.2 s arm change {np.round(dq, 4)}; qd {np.round(be.arm_qd(), 3)}")
    if label.startswith("elbow"):
        print(f"[eff]   expected elbow change ~ 0.5*(1/M33)*t^2 = {0.5 / be.arm_M()[3, 3] * 0.04:.3f} rad")
props = art._physics_view.get_dof_properties() if hasattr(art._physics_view, "get_dof_properties") else None
print("[eff] dof max velocities arm:", np.round(np.asarray(art.get_joint_max_velocities())[0, be.arm], 2) if hasattr(art, "get_joint_max_velocities") else "n/a")
print("[eff] dof max efforts arm:", np.round(np.asarray(art.get_max_efforts())[0, be.arm], 2))
print("[eff] friction coefficients arm:", np.round(np.asarray(art.get_friction_coefficients())[0, be.arm], 3))
print("[eff] armatures arm:", np.round(np.asarray(art.get_armatures())[0, be.arm], 3))
app.close()
