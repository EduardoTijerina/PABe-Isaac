"""Episodios PABe con la persona viva (Nathan, escena de oficina de GR00T) en Isaac Sim: el brazo lo mueve el
controlador nominal de PABe y el filtro del método elegido; la persona representa una familia de saludo de PABe
(i-ix) o una variante de GR00T (groot/families.py).

  ~/isaac_env/bin/python -u scripts/run_live_episode.py --families i_stable iv_hesitation --methods B1
Opciones: --families (por defecto las 9 de PABe; también extend hesitate withdraw partial),
--methods (B0 B1 sin modelos; B2..B9 con --models),
--pids, --dist / --bearing (colocación de la persona), --render-every / --no-viewport / --no-office (velocidad),
--gui, --out, --save-replay F.pkl (guarda los episodios para repetirlos), --replay F.pkl (con --gui: sólo repite
episodios guardados, sin simularlos), --video-dir D (con --replay y sin ventana: un MP4 por episodio, numerados como
los de MuJoCo: 01_i_stable.mp4 ...; --video-frame T: sólo una imagen de prueba en t = T del primero).
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "vendor")]
ap = argparse.ArgumentParser()
ap.add_argument("--families", nargs="+", default=None)
ap.add_argument("--methods", nargs="+", default=["B1"])
ap.add_argument("--pids", nargs="+", type=int, default=[4000])
ap.add_argument("--models", default=None)
# a 0,80 m su mano extendida queda en x ~ 0,41, al límite del alcance del brazo del G1: la palma se queda a 0,5-2 mm
# del umbral de agarre (R_C + TOUCH) y el saludo no se cierra; a 0,75 m queda en x ~ 0,36, el punto de encuentro
# de pabe.geometry.G1_GEOM
ap.add_argument("--dist", type=float, default=0.75)
ap.add_argument("--bearing", type=float, default=0.0, help="grados")
ap.add_argument("--render-every", type=int, default=1)
ap.add_argument("--no-viewport", action="store_true")
ap.add_argument("--no-office", action="store_true")
ap.add_argument("--usd-sync", action="store_true", help="escribir el estado de PhysX en USD en cada paso (lento)")
ap.add_argument("--gui", action="store_true")
ap.add_argument("--save-replay", default=None)
ap.add_argument("--replay", default=None)
ap.add_argument("--video-dir", default=None)
ap.add_argument("--video-frame", type=float, default=None)
ap.add_argument("--out", default=os.path.join(ROOT, "results", "live_episodes.json"))
a = ap.parse_args()

import isaacsim  # noqa: E402
from isaacsim import SimulationApp  # noqa: E402
KIT = os.path.join(os.path.dirname(isaacsim.__file__), "apps", "isaacsim.exp.action_and_event_data_generation.base.kit")
app = SimulationApp({"headless": not a.gui, "renderer": "RayTracedLighting"}, experience=KIT)

import numpy as np  # noqa: E402

from backends.isaac import IsaacBackend  # noqa: E402
from groot.families import PABE_FAMILIES  # noqa: E402
from groot.live_human import LiveBackend, LiveHuman, LiveHumanScenario, load_cfg, setup_scene  # noqa: E402
from pabe.core import METHODS  # noqa: E402
from pabe.episode import G1Episode  # noqa: E402

cfg = load_cfg()
sp = {}
t0 = time.time()
be = IsaacBackend(pre_reset=lambda w, st: sp.update(zip(("p", "a", "s"), setup_scene(w, st, cfg,
                                                                                       office=not a.no_office))),
                  floor=a.no_office, rendering_dt=a.render_every * 0.01,
                  usd_sync=a.usd_sync or a.gui or bool(a.video_dir))   # con ventana o vídeo el robot debe verse
human = LiveHuman(be.world, sp["a"], sp["s"], cfg)
human.warmup()
live = LiveBackend(be, human, render_every=a.render_every, viewport=not a.no_viewport)
print(f"[live] escena lista en {time.time() - t0:.0f} s (render cada {a.render_every} ciclos, "
      f"vista {'no' if a.no_viewport else 'sí'}, oficina {'no' if a.no_office else 'sí'})", flush=True)
models = None
if a.models:
    import pickle
    models = pickle.load(open(a.models, "rb"))

KEYS = ["R_reason", "success", "false_cancel", "V_any", "peak_F", "T", "n_infeas", "t_closure", "t_R",
        "release_fail", "worst_link_deficit"]
out, replays = [], []
if a.replay:
    import pickle
    replays = pickle.load(open(a.replay, "rb"))
for fam in ([] if a.replay else a.families or PABE_FAMILIES):
    for pid in a.pids:
        for meth in a.methods:
            human.rest()
            human.place(a.dist, np.radians(a.bearing))
            sc = LiveHumanScenario(human, fam, pid, 0)
            live.clip = sc.clip
            e = G1Episode(METHODS[meth], sc, models, live, record=True)
            tw = time.time()
            res, _, _ = e.run()
            wall = time.time() - tw
            row = {k: (None if (isinstance(res[k], float) and np.isnan(res[k])) else
                       (res[k].item() if hasattr(res[k], "item") else res[k])) for k in KEYS}
            row.update(family=fam, pid=pid, method=meth, wall=wall, ms_per_tick=wall / (res["T"] / 0.01) * 1e3,
                       commands=list(human.log), t_W=sc.t_W,
                       t_C=next((x["t"] for x in e.record if x["mode"] == "C"), None),
                       palm=[x["palm"].round(4).tolist() for x in e.record[::5]],
                       hand=[x["hand"].round(4).tolist() for x in e.record[::5]])
            out.append(row)
            replays.append((f"{fam} {meth}: {row['R_reason']}, {row['peak_F']:.1f} N",
                            [dict(t=x["t"], qpos=x["qpos"]) for x in e.record], list(human.log)))
            print(f"[live] {fam:<24} pid{pid} {meth}: {row['R_reason']} success={row['success']} V={row['V_any']} "
                  f"peakF={row['peak_F']:.1f} T={row['T']:.2f} t_C={row['t_C']} "
                  f"({wall:.0f} s, {row['ms_per_tick']:.1f} ms/ciclo)", flush=True)
if out:
    json.dump(out, open(a.out, "w"))
    print(f"[live] {len(out)} episodios -> {a.out}", flush=True)
if a.save_replay:
    import pickle
    pickle.dump(replays, open(a.save_replay, "wb"))
    print(f"[live] repeticiones -> {a.save_replay}", flush=True)
VIEW = dict(eye=[0.8, 2.15, 1.42], target=[0.42, 0.0, 1.05])   # tres cuartos desde el lado derecho de la persona


def replay(label, rec, log, realtime=True, on_tick=None):
    """Repite un episodio guardado: la persona repite sus órdenes en los mismos instantes (un render por ciclo de
    10 ms la anima al ritmo del registro) y el robot sigue las posiciones registradas."""
    print(f"[live] repetición: {label}", flush=True)
    human.rest()
    human.place(a.dist, np.radians(a.bearing))
    be.reset()
    pending = list(log)
    t0 = time.time()
    for k, x in enumerate(rec):
        if app.is_exiting():
            return
        while pending and pending[0][0] <= x["t"] + 1e-9:
            human.send(pending.pop(0)[1], x["t"])
        be.art.set_joint_positions(np.asarray(x["qpos"], float)[None])
        be.art.set_joint_velocities(np.zeros((1, be.n_dof)))
        be.world.step(render=False, update_fabric=True)
        be.world.render()
        if on_tick is not None and on_tick(k, x["t"]) is False:
            return
        wait = x["t"] - (time.time() - t0)
        if realtime and wait > 0:
            time.sleep(wait)


if a.video_dir and a.replay and not a.gui:
    import subprocess
    import cv2
    from isaacsim.core.utils.viewports import set_camera_view
    from isaacsim.sensors.camera import Camera
    W, H, EVERY = 1280, 720, 3                                  # 100/3 imágenes por segundo
    cam = Camera(prim_path="/World/VideoCam", resolution=(W, H))
    cam.initialize()
    cam.set_clipping_range(0.05, 100.0)
    from pxr import UsdGeom
    UsdGeom.Camera(be.stage.GetPrimAtPath("/World/VideoCam")).GetFocalLengthAttr().Set(14.0)   # gran angular
    set_camera_view(eye=VIEW["eye"], target=VIEW["target"], camera_prim_path="/World/VideoCam")
    for _ in range(10):
        be.world.render()
    os.makedirs(a.video_dir, exist_ok=True)

    def caption(img, lines):
        for n, txt in enumerate(lines):
            org = (24, 44 + 36 * n)
            cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 5, cv2.LINE_AA)
            cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        return img

    for n, (label, rec, log) in enumerate(replays, 1):
        fam = label.split()[0]
        if a.video_frame is not None:                       # imagen de prueba del encuadre
            def grab(k, t):
                if t >= a.video_frame - 1e-9:
                    path = os.path.join(a.video_dir, f"frame_{fam}_{a.video_frame:.2f}s.png")
                    cv2.imwrite(path, caption(cam.get_rgba()[:, :, :3][:, :, ::-1].copy(), [label, f"t = {t:.2f} s"]))
                    print("[live] imagen ->", path, flush=True)
                    return False
            replay(label, rec, log, realtime=False, on_tick=grab)
            break
        path = os.path.join(a.video_dir, f"{n:02d}_{fam}.mp4")
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                               "-s", f"{W}x{H}", "-r", f"{100 / EVERY:.4f}", "-i", "-", "-pix_fmt", "yuv420p",
                               "-crf", "20", path], stdin=subprocess.PIPE)

        def write(k, t):
            if k % EVERY == 0:
                rgb = cam.get_rgba()
                if rgb is not None and rgb.size:
                    ff.stdin.write(caption(rgb[:, :, :3][:, :, ::-1].copy(), [label, f"t = {t:.2f} s"]).tobytes())
        replay(label, rec, log, realtime=False, on_tick=write)
        ff.stdin.close(); ff.wait()
        print(f"[live] vídeo -> {path}", flush=True)

if a.gui:
    from isaacsim.core.utils.viewports import set_camera_view
    set_camera_view(**VIEW)
    print("[live] repitiendo los episodios en tiempo real; cerrar la ventana de Isaac Sim para salir", flush=True)
    while not app.is_exiting():
        for r in replays:
            if app.is_exiting():
                break
            replay(*r)
            for _ in range(100):
                be.world.render()
app.close()
