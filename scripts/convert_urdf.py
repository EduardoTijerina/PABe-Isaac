"""URDF G1 + Revo2 -> USD para Isaac Sim (assets/g1_revo2.usd).

Base fija (como en MuJoCo), sin fusionar las articulaciones fijas (right_base_link y las falanges existen como
cuerpos), articulaciones mimic de los dedos, sin autocolisión e inercias del URDF. Uso (Python de Isaac Sim):
    ~/isaac_env/bin/python scripts/convert_urdf.py [--urdf PATH] [--out assets/g1_revo2.usd]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
ap = argparse.ArgumentParser()
ap.add_argument("--urdf", default=None)
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "assets", "g1_revo2.usd"))
a = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import omni.kit.commands  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.asset.importer.urdf")
from isaacsim.asset.importer.urdf import _urdf  # noqa: E402

from pabe.robot import REVO2_URDF  # noqa: E402

urdf = os.path.abspath(a.urdf or REVO2_URDF)
out = os.path.abspath(a.out)
os.makedirs(os.path.dirname(out), exist_ok=True)
_, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
cfg.merge_fixed_joints = False
cfg.fix_base = True
cfg.import_inertia_tensor = True
cfg.self_collision = False
cfg.parse_mimic = True
cfg.make_default_prim = True
cfg.create_physics_scene = False
cfg.distance_scale = 1.0
cfg.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
_, path = omni.kit.commands.execute("URDFParseAndImportFile", urdf_path=urdf, import_config=cfg, dest_path=out)
print(f"[convert] {urdf}\n[convert] -> {out} (robot prim {path})", flush=True)
app.close()
