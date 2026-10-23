import json, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "vendor")]
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import numpy as np
from backends.isaac import IsaacBackend
ref = json.load(open(os.path.join(ROOT, "results", "mujoco_mass.json")))
be = IsaacBackend(); be.reset()
pv = be.art._physics_view
mass = np.asarray(pv.get_masses())[0]; inert = np.asarray(pv.get_inertias())[0]
for n, r in ref.items():
    i = be.body_names.index(n)
    print(f"[mass] {n:<28} isaac {mass[i]:8.3f} kg  mujoco {r['mass']:8.3f}  inertia diag isaac {np.round(inert[i].reshape(3,3).diagonal(),5)} mujoco {np.round(r['inertia'],5)}")
print(f"[mass] total isaac {mass.sum():.2f} kg")
import omni.usd
from pxr import UsdPhysics, PhysxSchema
st = omni.usd.get_context().get_stage()
for p in st.Traverse():
    if p.IsA(UsdPhysics.Scene):
        s = UsdPhysics.Scene(p); print("[mass] scene", p.GetPath(), "gravity", s.GetGravityDirectionAttr().Get(), s.GetGravityMagnitudeAttr().Get())
    if p.GetName() == "right_elbow_joint":
        print("[mass] elbow joint attrs:", {a.GetName(): a.Get() for a in p.GetAttributes() if any(k in a.GetName() for k in ("damping", "riction", "stiffness", "armature", "maxJointVelocity", "maxForce"))})
app.close()
