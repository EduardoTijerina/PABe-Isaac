"""Corre una lista de episodios en un backend y guarda resultados y trayectorias de la palma (JSON).

  MuJoCo:  PABE_ROBOT=revo2 <python con mujoco> scripts/run_episodes.py --backend mujoco --out results/mj.json
  Isaac:   ~/isaac_env/bin/python -u scripts/run_episodes.py --backend isaac --out results/isaac.json [--gui]
Opciones: --families, --pids, --eps, --methods (B0 B1 sin modelos; B2..B9 con --models), --approach (caminata).
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "vendor")]
ap = argparse.ArgumentParser()
ap.add_argument("--backend", choices=["mujoco", "isaac"], required=True)
ap.add_argument("--families", nargs="+", default=["i_stable", "iv_hesitation", "iii_withdraw_contact", "x_walk_hesitation"])
ap.add_argument("--pids", nargs="+", type=int, default=[4000, 4001, 4002])
ap.add_argument("--eps", type=int, default=1)
ap.add_argument("--methods", nargs="+", default=["B0", "B1"])
ap.add_argument("--models", default=None)
ap.add_argument("--approach", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--gui", action="store_true", help="Isaac: ventana con el humano visual (más lento; cámara desde "
                                                   "el lado derecho de la persona)")
a = ap.parse_args()

if a.backend == "isaac":
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": not a.gui})
import numpy as np  # noqa: E402

from pabe.core import METHODS  # noqa: E402
from pabe.episode import G1Episode  # noqa: E402
from pabe.geometry import G1_GEOM  # noqa: E402
from pabe.scenarios import Participant, Scenario  # noqa: E402

if a.backend == "isaac":
    from backends.isaac import IsaacBackend
    be = IsaacBackend(human_visual=a.gui)
else:
    from backends.mujoco import MujocoBackend
    be = MujocoBackend()
models = None
if a.models:
    import pickle
    models = pickle.load(open(a.models, "rb"))

KEYS = ["R_reason", "success", "false_cancel", "V_any", "peak_F", "T", "n_infeas", "jerk_p95", "t_closure", "t_R",
        "variant", "t_handoff", "approach_abort", "release_fail", "worst_link_deficit"]
out = []
t_all = time.time()
for fam in a.families:
    for pid in a.pids:
        for ep in range(a.eps):
            for meth in a.methods:
                sc = Scenario(fam, Participant(pid, geom=G1_GEOM), ep)
                e = G1Episode(METHODS[meth], sc, models, be, record=True, approach=a.approach, render=a.gui)
                t0 = time.time()
                res, _, _ = e.run()
                row = {k: (None if (isinstance(res[k], float) and np.isnan(res[k])) else
                           (res[k].item() if hasattr(res[k], "item") else res[k])) for k in KEYS}
                row.update(family=fam, pid=pid, ep=ep, method=meth, wall=time.time() - t0,
                           t_C=next((x["t"] for x in e.record if x["mode"] == "C"), None),
                           palm=[x["palm"].round(4).tolist() for x in e.record[::5]])
                out.append(row)
                print(f"[{a.backend}] {fam:<22} pid{pid} ep{ep} {meth}: {row['R_reason']} success={row['success']} "
                      f"V={row['V_any']} peakF={row['peak_F']:.1f} T={row['T']:.2f} ({row['wall']:.1f}s wall)", flush=True)
json.dump(out, open(a.out, "w"))
print(f"[{a.backend}] {len(out)} episodes in {time.time() - t_all:.0f}s -> {a.out}", flush=True)
if a.backend == "isaac":
    if a.gui:
        # repetición a velocidad real del último episodio, en bucle hasta cerrar la ventana (la simulación en
        # vivo va ~7-10 veces más lenta que el tiempo real)
        from pabe.episode import PALM_FACE
        print("[isaac] replaying the last episode in real time; close the Isaac Sim window to exit", flush=True)
        rec = e.record
        walk = any(x["x_r"] > 0 for x in rec)
        if walk:
            # caminata: el robot (base fija en la física) se desplaza aquí solo visualmente; la persona ya está en
            # el marco del robot, así que se le aplica el mismo desplazamiento. Cámara más abierta.
            from pxr import Gf, UsdGeom
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            hum = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Human"))
            hum_t = next((op for op in hum.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate),
                         None) or hum.AddTranslateOp()
            from isaacsim.core.utils.viewports import set_camera_view
            set_camera_view(eye=[0.2, 3.6, 1.7], target=[0.2, 0.0, 0.9])
            pz = be.art.get_world_poses()[0][0, 2]
            # piernas: oscilación de marcha cinemática, como en el visor de MuJoCo (pabe/walk_anim.py)
            from pabe import walk_anim as W
            knots_r, knots_h = W.robot_knots(rec), sc.walk.knots_h
            dof = {n: i for i, n in enumerate(be.dof_names)}
        while not app.is_exiting():
            t0 = time.time()
            for x in rec:
                if app.is_exiting():
                    break
                q = np.array(x["qpos"], float)
                be.set_human(x["pose"], PALM_FACE)
                if walk:
                    env_r, ph_r = W.gait(knots_r, x["t"], W.STRIDE_R)
                    env_h, ph_h = W.gait(knots_h, x["t"], W.STRIDE_H)
                    W.robot_gait(q, dof, ph_r, env_r)
                    be.art.set_world_poses(positions=np.array([[-x["x_r"], 0.0,
                                                                 pz - W.BOB * env_r * np.sin(ph_r) ** 2]]))
                    hum_t.Set(Gf.Vec3d(-float(x["x_r"]), 0.0, -W.BOB * env_h * np.sin(ph_h) ** 2))
                    # marco de /World/Human = marco del robot: la cadera va en x_h + x_r
                    be.set_human_legs(W.human_legs(sc.walk.human(x["t"]) + x["x_r"], ph_h, env_h, 0.0))
                be.art.set_joint_positions(q[None])
                be.art.set_joint_velocities(np.zeros((1, be.n_dof)))
                be.world.step(render=False, update_fabric=True)
                be.world.render()
                wait = x["t"] - (time.time() - t0)
                if wait > 0:
                    time.sleep(wait)
            for _ in range(60):                       # ~1 s de pausa entre repeticiones
                be.world.render()
    app.close()
