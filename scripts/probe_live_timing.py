"""Coste por ciclo de control con la persona viva y opciones para abaratarlo: cuánto cuestan los 5 pasos de física y
world.render(), si el personaje sigue animándose con la vista desactivada (ViewportAPI.updates_enabled = False) y a
qué ritmo avanza la animación con rendering_dt = 20 ms y un render cada 2 ciclos.
Uso: ~/isaac_env/bin/python -u scripts/probe_live_timing.py [--no-office]"""
import argparse
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "vendor")]
ap = argparse.ArgumentParser()
ap.add_argument("--no-office", action="store_true")
a = ap.parse_args()

import isaacsim  # noqa: E402
from isaacsim import SimulationApp  # noqa: E402
KIT = os.path.join(os.path.dirname(isaacsim.__file__), "apps", "isaacsim.exp.action_and_event_data_generation.base.kit")
app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"}, experience=KIT)

import numpy as np  # noqa: E402

from backends.isaac import IsaacBackend  # noqa: E402
from groot.live_human import LiveHuman, load_cfg, setup_scene  # noqa: E402

cfg = load_cfg()
sp = {}
be = IsaacBackend(pre_reset=lambda w, st: sp.update(zip(("p", "a", "s"), setup_scene(w, st, cfg, office=not a.no_office))),
                  floor=a.no_office)
hu = LiveHuman(be.world, sp["a"], sp["s"], cfg)
hu.warmup()
hu.place(0.80, 0.0)
be.reset()


def timed(fn, n):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e3


def phys5():
    for _ in range(5):
        be.set_arm_torque(be.arm_bias())
        be.step()


def extend_run(render_every, label, ticks=250):
    """Lanza 'extend' y mide cuándo la mano llega al 50 % del avance (perfil: 0,90 s)."""
    hu.rest(); hu.place(0.80, 0.0)
    hu.start("extend")
    x0 = hu.joint(hu.j_wrist)[0]
    t_half, tic = None, time.perf_counter()
    for k in range(1, ticks + 1):
        phys5()
        if k % render_every == 0:
            be.world.render()
        x = hu.joint(hu.j_wrist)[0]
        if t_half is None and x0 - x > 0.5 * 0.26:
            t_half = k * 0.01
    ms = (time.perf_counter() - tic) / ticks * 1e3
    print(f"[timing] {label}: {ms:.1f} ms/ciclo; mano al 50 % en t={t_half} s (perfil 0,90 s); "
          f"avance final {x0 - hu.joint(hu.j_wrist)[0]:.3f} m", flush=True)


print(f"[timing] oficina={'no' if a.no_office else 'sí'}", flush=True)
print(f"[timing] 5 pasos de física: {timed(phys5, 100):.1f} ms; world.render(): {timed(be.world.render, 50):.1f} ms",
      flush=True)
extend_run(1, "render cada ciclo (referencia)")
from omni.kit.viewport.utility import get_active_viewport  # noqa: E402
vp = get_active_viewport()
if vp is not None:
    vp.updates_enabled = False
    print(f"[timing] vista desactivada: world.render() {timed(be.world.render, 50):.1f} ms", flush=True)
    extend_run(1, "vista desactivada, render cada ciclo")
be.world.set_simulation_dt(physics_dt=0.002, rendering_dt=0.02)
extend_run(2, "vista desactivada, rendering_dt 20 ms, render cada 2 ciclos")
app.close()
