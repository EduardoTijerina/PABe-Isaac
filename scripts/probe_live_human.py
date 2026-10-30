"""Prueba de la persona viva (groot/live_human.py) en el backend Isaac, sin método ni política: el brazo sólo
compensa la gravedad. Comprueba que el personaje se anima con el paso del backend (5 pasos de física + 1
world.render() por ciclo de control), que world.render() no avanza la física, la temporización de cada variante
frente a su perfil y dónde queda la mano (alcance de PABe).
Uso: ~/isaac_env/bin/python -u scripts/probe_live_human.py [--gui] [--variants extend lunge ...] [--dist 0.80]
       [--no-office] [--shots]   (clips de groot/families.CLIPS; 'extend+retract@3.0' lanza retract a los 3 s; --shots
                                  guarda capturas laterales en los instantes clave de KEY_T)"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "vendor")]
ap = argparse.ArgumentParser()
ap.add_argument("--gui", action="store_true")
ap.add_argument("--variants", nargs="+", default=["extend", "hesitate", "withdraw", "partial"])
ap.add_argument("--dist", type=float, default=0.80)
ap.add_argument("--secs", type=float, default=8.0)
ap.add_argument("--no-office", action="store_true")
ap.add_argument("--shots", action="store_true")
ap.add_argument("--out", default=os.path.join(ROOT, "results", "probe_live_human.json"))
a = ap.parse_args()

import isaacsim  # noqa: E402
from isaacsim import SimulationApp  # noqa: E402
KIT = os.path.join(os.path.dirname(isaacsim.__file__), "apps", "isaacsim.exp.action_and_event_data_generation.base.kit")
app = SimulationApp({"headless": not a.gui, "renderer": "RayTracedLighting"}, experience=KIT)

import numpy as np  # noqa: E402

from backends.isaac import IsaacBackend  # noqa: E402
from groot.live_human import LiveBackend, LiveHuman, load_cfg, setup_scene  # noqa: E402
from pabe import geometry as G  # noqa: E402
import handshake_variants as HV  # noqa: E402
from groot.families import CLIPS  # noqa: E402

KEY_T = {"withdraw_pre": [1.1, 2.2], "withdraw_contact": [3.9], "hesitation": [1.8, 5.2], "lunge": [1.15, 2.2],
         "ambiguous_withdraw": [4.7], "extend+retract@3.0": [3.4, 4.0], "extend": [2.5]}

cfg = load_cfg()
spawned = {}
t0 = time.time()
be = IsaacBackend(pre_reset=lambda w, st: spawned.update(zip(("prim", "agent", "skel"),
                                                             setup_scene(w, st, cfg, office=not a.no_office))),
                  floor=a.no_office)
print(f"[probe] escena lista en {time.time() - t0:.0f} s; agente={spawned['agent']} skel={spawned['skel']}", flush=True)
human = LiveHuman(be.world, spawned["agent"], spawned["skel"], cfg)
print(f"[probe] articulaciones: muñeca={human.j_wrist} codo={human.j_elbow} hombro={human.j_shoulder}", flush=True)
print(f"[probe] esqueleto ({len(human.tokens)}): {human.tokens}", flush=True)
human.warmup()
live = LiveBackend(be, human)

out = dict(joints=[human.j_wrist, human.j_elbow, human.j_shoulder], variants={})
if a.shots:   # la vista se coloca antes: si se mueve en el mismo fotograma de la captura, sale la vista anterior
    from isaacsim.core.utils.viewports import set_camera_view
    set_camera_view(eye=[0.45, -1.9, 1.3], target=[0.7, 0.0, 1.15])
for v in a.variants:
    stop, facing = human.place(a.dist, 0.0)
    live.reset()
    q0 = be.arm_q()
    clip, _, later = v.partition("+")
    human.start(clip)
    extra = (later.split("@")[0], float(later.split("@")[1])) if later else None
    shots = list(KEY_T.get(v, [])) if a.shots else []
    tp0, tw0 = be.world.current_time, time.time()
    n = int(a.secs / 0.01)
    for k in range(n):
        for _ in range(5):
            be.set_arm_torque(be.arm_bias())          # sólo compensación de gravedad
            live.step()
        t_now = (k + 1) * 0.01
        if extra and t_now >= extra[1]:
            human.send(extra[0], t_now); extra = None
        if shots and t_now >= shots[0] - 1e-9:
            from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
            os.makedirs(os.path.join(ROOT, "results", "clip_shots"), exist_ok=True)
            capture_viewport_to_file(get_active_viewport(), os.path.join(ROOT, "results", "clip_shots",
                                                                         f"{v.replace('@', '_')}_{shots.pop(0):.2f}s.png"))
    dt_phys, wall = be.world.current_time - tp0, time.time() - tw0
    ts = np.array(human.buf_t)
    hand = np.array([human.hand(t) for t in ts])
    sh = np.array([human.shoulder(t) for t in ts])
    z = hand[:, 2]
    rise = next((t for t, zz in zip(ts, z) if zz > z[0] + 0.05), None)
    prof = CLIPS[clip][1]
    print(f"[probe] {v}: física {dt_phys:.2f} s para {n * 0.01:.2f} s de control (render no debe sumar), "
          f"{wall / n * 1e3:.1f} ms/ciclo; deriva del brazo {np.abs(be.arm_q() - q0).max():.3f} rad", flush=True)
    print(f"        mano t=0 {np.round(hand[0], 3)}  máx z {z.max():.3f} en t={ts[z.argmax()]:.2f} s  "
          f"empieza a subir en t={rise}  final {np.round(hand[-1], 3)}", flush=True)
    print(f"        hombro {np.round(sh[0], 3)} (modelo {G.H_SHOULDER}); distancia mano-ROBOT_REF máx "
          f"{np.linalg.norm(hand - G.ROBOT_REF, axis=1).min():.3f} m (REACH_MAX 1.2); perfil {prof}", flush=True)
    for tt in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0):
        if tt <= ts[-1]:
            i = int(np.argmin(np.abs(ts - tt)))
            print(f"        t={tt:4.1f}  mano {np.round(hand[i], 3)}  alfa_perfil(t-lat)="
                  f"{HV.profile_alpha(prof, tt - human.latency):.2f}", flush=True)
    out["variants"][v] = dict(t=ts.tolist(), hand=hand.tolist(), shoulder=sh.tolist(), stop=list(map(float, stop)),
                              phys_s=dt_phys, ctrl_s=n * 0.01, ms_per_tick=wall / n * 1e3)
    human.rest()

with open(a.out, "w") as f:
    json.dump(out, f)
print("[probe] guardado", a.out, flush=True)
if a.gui:
    while app.is_running():
        app.update()
app.close()
