"""Desglose del coste de un paso de física (2 ms) en la escena de la persona viva: world.step(render=False) frente a
PhysX directo (simulate + fetch_results), lecturas del tensor API que hace el episodio en cada subpaso, escritura del
estado a USD (/physics/updateToUsd) y colisionadores estáticos de la oficina. Comprueba además que el paso directo
deja el mismo estado que world.step (misma trayectoria del brazo en caída libre desde el mismo estado).
Uso: ~/isaac_env/bin/python -u scripts/probe_physics_cost.py"""
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "vendor")]
import isaacsim  # noqa: E402
from isaacsim import SimulationApp  # noqa: E402
KIT = os.path.join(os.path.dirname(isaacsim.__file__), "apps", "isaacsim.exp.action_and_event_data_generation.base.kit")
app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"}, experience=KIT)

import carb  # noqa: E402
import numpy as np  # noqa: E402
from omni.physx import get_physx_simulation_interface  # noqa: E402

from backends.isaac import DT_PHYS, IsaacBackend  # noqa: E402
from groot.live_human import LiveHuman, load_cfg, setup_scene  # noqa: E402

cfg = load_cfg()
sp = {}
be = IsaacBackend(pre_reset=lambda w, st: sp.update(zip(("p", "a", "s"), setup_scene(w, st, cfg))), floor=False)
hu = LiveHuman(be.world, sp["a"], sp["s"], cfg)
be.reset()
si = get_physx_simulation_interface()
st = carb.settings.get_settings()
T = [0.0]


def ms(fn, n=200):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e3


def direct():
    si.simulate(DT_PHYS, T[0]); si.fetch_results(); T[0] += DT_PHYS


def free_fall(stepper, n=100):
    """Brazo sin par desde el estado de reposo: trayectoria articular tras n pasos."""
    be.reset()
    for _ in range(n):
        be.set_arm_torque(np.zeros(7))
        stepper()
    return be.arm_q()


print("[cost] ajustes:", {k: st.get(k) for k in ("/physics/updateToUsd", "/physics/updateVelocitiesToUsd",
                                                 "/physics/fabricEnabled", "/physics/numThreads")}, flush=True)
ws = lambda: be.world.step(render=False)  # noqa: E731
print(f"[cost] world.step(render=False): {ms(ws):.2f} ms", flush=True)
print(f"[cost] simulate+fetch_results:   {ms(direct):.2f} ms", flush=True)
for name, fn in (("arm_q", be.arm_q), ("arm_qd", be.arm_qd), ("arm_M", be.arm_M), ("arm_bias", be.arm_bias),
                 ("palm", be.palm), ("_read", be._read), ("kin", be.kin)):
    print(f"[cost] {name:<9} {ms(fn, 100):.2f} ms", flush=True)
qa, qb = free_fall(ws), free_fall(direct)
print(f"[cost] caída libre 0,2 s: |world.step - directo| = {np.abs(qa - qb).max():.2e} rad "
      f"(desplazamiento {np.abs(qa - be.q_stand[be.arm]).max():.3f} rad)", flush=True)
st.set_bool("/physics/updateToUsd", False)
st.set_bool("/physics/updateVelocitiesToUsd", False)
print(f"[cost] sin escritura a USD: world.step {ms(ws):.2f} ms, directo {ms(direct):.2f} ms", flush=True)
qc = free_fall(direct)
print(f"[cost]   caída libre igual que antes: {np.abs(qc - qb).max():.2e} rad", flush=True)
from pxr import UsdPhysics  # noqa: E402
n_col = sum(1 for p in be.stage.Traverse() if str(p.GetPath()).startswith("/World/Room")
            and p.HasAPI(UsdPhysics.CollisionAPI))
print(f"[cost] colisionadores en la oficina: {n_col}", flush=True)
be.stage.GetPrimAtPath("/World/Room").SetActive(False)
for _ in range(3):
    app.update()
be.art.initialize()
be.reset()
print(f"[cost] oficina desactivada: world.step {ms(ws):.2f} ms, directo {ms(direct):.2f} ms", flush=True)
app.close()
