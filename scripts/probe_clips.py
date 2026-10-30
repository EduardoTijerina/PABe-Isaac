"""Prueba de los clips de groot/author_clips.py --candidates sobre Nathan (espacio vacío): para cada candidata del brazo,
posición final de la mano, holgura del antebrazo y la mano respecto del pecho (esqueleto) y dos capturas (vista del
G1 y lateral); para la cabeza, la dirección de la mirada (ojos) en cada segmento de HeadProbe y la dirección hacia
la cabeza del G1.
Uso: ~/isaac_env/bin/python -u scripts/probe_clips.py"""
import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "vendor")]
import isaacsim  # noqa: E402
from isaacsim import SimulationApp  # noqa: E402
KIT = os.path.join(os.path.dirname(isaacsim.__file__), "apps", "isaacsim.exp.action_and_event_data_generation.base.kit")
app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": 960, "height": 720}, experience=KIT)

import numpy as np  # noqa: E402

from backends.isaac import IsaacBackend  # noqa: E402
from groot.live_human import LiveHuman, load_cfg, setup_scene  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport  # noqa: E402

CLIPS = os.path.join(ROOT, "assets", "human_clips")
SHOTS = os.path.join(ROOT, "results", "clip_shots")
os.makedirs(SHOTS, exist_ok=True)
cands = sorted(glob.glob(f"{CLIPS}/cand*.skelanim.usd"))
cfg = load_cfg(own_clips=False)
clips = cfg["milestone3"]["variants"]["clips"]
for p in cands + [f"{CLIPS}/head_probe.skelanim.usd"]:
    clips[os.path.basename(p).split(".")[0]] = {"anim_usd": p}
sp = {}
be = IsaacBackend(pre_reset=lambda w, st: sp.update(zip(("p", "a", "s"), setup_scene(w, st, cfg, office=False))))
hu = LiveHuman(be.world, sp["a"], sp["s"], cfg)
hu.warmup()
hu.place(0.80, 0.0)
be.reset()
pre = hu.j_wrist[: -len("hand_r")]
J = lambda leaf: hu.joint(pre + leaf)  # noqa: E731
heads = [n for n in be.body_names if "head" in n.lower() or "logo" in n.lower()]
print("[clips] cuerpos del G1 con 'head'/'logo':", heads, flush=True)
# el origen de head_link coincide con el de la pelvis (el URDF lo monta así): objetivo fijo a la altura de los ojos
g1_head = np.array([0.05, 0.0, 1.22])
print("[clips] objetivo de la mirada (cabeza del G1):", g1_head, flush=True)
vp = get_active_viewport()


def render(n):
    for _ in range(n):
        be.world.render()


def shot(name, eye, target):
    set_camera_view(eye=eye, target=target)
    render(5)
    capture_viewport_to_file(vp, os.path.join(SHOTS, name))
    render(10)


def seg_dist(p, a, b):
    ab = b - a
    s = np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0)
    return np.linalg.norm(p - (a + s * ab))


def clearance():
    """Distancia mínima de puntos del antebrazo y de la mano al eje del tronco (cadera -> cuello)."""
    a, b = J("spine_01"), J("neck")
    e, w = J("lowerarm_r"), J("hand_r")
    hand = w + 0.08 * (w - e) / np.linalg.norm(w - e)
    pts = [e + s * (w - e) for s in np.linspace(0.0, 1.0, 6)] + [hand]
    return min(seg_dist(p, a, b) for p in pts), hand


def gaze():
    v = sum(J(f"eye_end_{s}") - J(f"eye_{s}") for s in "lr")
    return v / np.linalg.norm(v)


def gaze_err():
    """Ángulo (grados) entre la mirada y la dirección de los ojos a la cabeza del G1."""
    e = 0.5 * (J("eye_l") + J("eye_r"))
    w = (g1_head - e) / np.linalg.norm(g1_head - e)
    return float(np.degrees(np.arccos(np.clip(gaze() @ w, -1, 1))))


out = dict(g1_head=g1_head.tolist(), cands={}, head={})
d0, h0 = clearance()
eye0 = 0.5 * (J("eye_l") + J("eye_r"))
want = (g1_head - eye0) / np.linalg.norm(g1_head - eye0)
print(f"[clips] reposo: holgura tronco {d0:.3f} m, mano {np.round(h0, 3)}; error de mirada {gaze_err():.1f}°",
      flush=True)
shot("idle_g1view.png", [-0.75, -0.05, 1.25], [0.8, 0.0, 1.15])
from groot.author_clips import CANDIDATES  # noqa: E402
for i, p in enumerate(cands):
    hu.hi.send_custom_command(hu.agent, f"Cand{i}", 60.0)
    render(260)
    d, hand = clearance()
    out["cands"][i] = dict(clearance=d, hand=hand.tolist(), shoulder=J("upperarm_r").tolist())
    out["cands"][i]["gaze_err_deg"] = gaze_err()
    out["cands"][i]["params"] = [CANDIDATES[i][0], CANDIDATES[i][1], CANDIDATES[i][2], len(CANDIDATES[i][3]) > 0]
    print(f"[clips] cand{i} {out['cands'][i]['params']}: holgura tronco {d:.3f} m (reposo {d0:.3f}); "
          f"mano {np.round(hand, 3)}; error de mirada {gaze_err():.1f}°", flush=True)
    shot(f"cand{i}_g1view.png", [-0.75, -0.05, 1.25], [0.8, 0.0, 1.15])
    shot(f"cand{i}_side.png", [0.45, -1.9, 1.3], [0.7, 0.0, 1.15])
    hu.rest(80)
if os.environ.get("HEAD_PROBE"):
    hu.hi.send_custom_command(hu.agent, "HeadProbe", 60.0)
t = 0.0
for k, lab in enumerate(["+X", "-X", "+Y", "-Y", "+Z", "-Z"] if os.environ.get("HEAD_PROBE") else []):
    render(int(round((1.5 * k + 0.9 - t) / 0.01)))
    t = 1.5 * k + 0.9
    g = gaze()
    out["head"][lab] = g.tolist()
    print(f"[clips] Head {lab} 25°: mirada {np.round(g, 3)}", flush=True)
json.dump(out, open(os.path.join(ROOT, "results", "probe_clips.json"), "w"))
print("[clips] capturas en", SHOTS, flush=True)
app.close()
