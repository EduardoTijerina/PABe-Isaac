"""Clips de saludo de la persona viva para ESTA simulación (no toca los de ~/gr00t_project/assets, que usa la
recogida de datos de GR00T). Misma receta que gr00t_project/stage2_collect/author_handshake_clip.py (deltas
locales sobre la pose 0 del clip de reposo, mano abierta injertada del clip de saludo, perfiles alfa de
handshake_variants), con dos cambios:

  brazo   parámetros propios (POSE): en el original la elevación baja (12°) seguida de 30° de aducción barre el
          antebrazo por delante del pecho y lo atraviesa
  cabeza  rotación local de Head (y Neck2) para que mire a la cabeza del G1, que es más bajo

Sin SimulationApp (pxr + numpy):
  EXTS=~/isaac_env/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311
  PYTHONPATH=$EXTS LD_LIBRARY_PATH=$EXTS/bin ~/isaac_env/bin/python groot/author_clips.py [--candidates | --head-probe]
"""
import argparse
import os
import sys

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdSkel

PROJ = os.path.expanduser("~/gr00t_project")
sys.path.insert(0, os.path.join(PROJ, "stage2_collect"))
from author_handshake_clip import (HAND_BLEND, HAND_DONOR, HAND_DONOR_FRAME, axis_angle, np_to_quatf,  # noqa: E402
                                   q_to_np, qmul, slerp)
from handshake_variants import VARIANTS, profile_alpha  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from groot.families import CLIPS, clip_file  # noqa: E402

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "human_clips"))
IDLE = f"{PROJ}/assets/stand_idle_loop.skelanim.usd"
X, Y, Z = np.eye(3)

# pose elegida (scripts/probe_clips.py, 2026-10-29; candidata 3 de la segunda ronda): brazo = elevación local +Y,
# luego aducción +Z (original 12°, 30°: holgura del antebrazo al eje del tronco 0,128 m, lo atraviesa; ésta 0,217 m,
# en reposo 0,222 m), codo recto; cabeza baja ~33° y gira ~14° hacia el G1 (error de mirada 45,7° -> 7,1°).
# Mano ofrecida en (0,454, 0,083, 1,048) con la persona a 0,80 m.

# candidatas para la prueba: (elevación, aducción, flexión del codo, cabeza). Cabeza (scripts/probe_clips.py,
# 25° por eje): +Y baja la mirada ~27°, +X la gira hacia la izquierda de la persona (-y del mundo)
HEAD_A = [(Y, 33.0), (X, 14.0)]
HEAD_B = [(Y, 28.0), (X, 10.0)]
POSE = dict(raise_deg=32.0, across_deg=25.0, bend_deg=0.0, head=HEAD_A)
CANDIDATES = [(12, 30, 20, []), (28, 18, 10, HEAD_A), (28, 25, 0, HEAD_A), (32, 25, 0, HEAD_A),
              (35, 22, 0, HEAD_B), (30, 28, 5, HEAD_B), (35, 30, 0, HEAD_A)]


def load_idle():
    src = Usd.Stage.Open(IDLE)
    anim = UsdSkel.Animation(src.GetPrimAtPath(str(src.GetDefaultPrim().GetPath())))
    joints = list(anim.GetJointsAttr().Get())
    return dict(src=src, joints=joints, rot0=[q_to_np(q) for q in anim.GetRotationsAttr().Get(0.0)],
                tr0=list(anim.GetTranslationsAttr().Get(0.0)), scales=anim.GetScalesAttr().Get(),
                tcps=src.GetTimeCodesPerSecond())


def find(joints, leaf):
    m = [j for j in joints if j.split("/")[-1] == leaf]
    if len(m) != 1:
        raise RuntimeError(f"{leaf}: {m}")
    return m[0]


def targets_for(idle, raise_deg, across_deg, bend_deg, head):
    """Rotaciones locales objetivo (alfa = 1) por articulación."""
    j, r0 = idle["joints"], idle["rot0"]
    idx = {x: i for i, x in enumerate(j)}
    up, lo, hd = find(j, "R_UpArm"), find(j, "R_LoArm"), find(j, "Head")
    t = {up: qmul(r0[idx[up]], qmul(axis_angle(Z, across_deg), axis_angle(Y, raise_deg))),
         lo: qmul(r0[idx[lo]], axis_angle(Y, bend_deg))}
    if head:
        d = np.array([1.0, 0.0, 0.0, 0.0])
        for ax, deg in head:
            d = qmul(axis_angle(ax, deg), d)
        t[hd] = qmul(r0[idx[hd]], d)
    # mano abierta: dedos de la mano derecha mezclados con el clip de saludo (igual que el original)
    donor = Usd.Stage.Open(HAND_DONOR)
    da = UsdSkel.Animation(donor.GetPrimAtPath(str(donor.GetDefaultPrim().GetPath())))
    drot = [q_to_np(q) for q in da.GetRotationsAttr().Get(HAND_DONOR_FRAME)]
    didx = {x.split("/")[-1]: i for i, x in enumerate(da.GetJointsAttr().Get())}
    for x in j:
        leaf = x.split("/")[-1]
        if leaf.startswith("R_") and "Finger" in leaf and leaf in didx:
            t[x] = slerp(r0[idx[x]], drot[didx[leaf]], HAND_BLEND)
    return t


def write_clip(idle, path, command, keyframes):
    """keyframes: [(t_s, {articulación: cuaternión})]; las que faltan quedan en la pose de reposo."""
    j, r0, tcps = idle["joints"], idle["rot0"], idle["tcps"]
    end_f = int(round(keyframes[-1][0] * tcps))
    st = Usd.Stage.CreateNew(path)
    st.SetTimeCodesPerSecond(tcps); st.SetFramesPerSecond(tcps)
    st.SetStartTimeCode(0); st.SetEndTimeCode(end_f)
    UsdGeom.SetStageUpAxis(st, UsdGeom.GetStageUpAxis(idle["src"]))
    UsdGeom.SetStageMetersPerUnit(st, UsdGeom.GetStageMetersPerUnit(idle["src"]))
    out = UsdSkel.Animation.Define(st, "/Root")
    prim = out.GetPrim()
    st.SetDefaultPrim(prim)
    out.GetJointsAttr().Set(j)
    out.GetTranslationsAttr().Set(idle["tr0"])
    if idle["scales"] is not None:
        out.GetScalesAttr().Set(idle["scales"])
    for ts, pose in keyframes:
        out.GetRotationsAttr().Set([np_to_quatf(pose.get(x, r0[i])) for i, x in enumerate(j)],
                                   Usd.TimeCode(int(round(ts * tcps))))
    for k, typ, v in (("CustomCommandName", Sdf.ValueTypeNames.String, command),
                      ("CustomCommandTemplate", Sdf.ValueTypeNames.String, "Timing"),
                      ("CustomCommandAnimStartTime", Sdf.ValueTypeNames.Float, 0.0),
                      ("CustomCommandAnimEndTime", Sdf.ValueTypeNames.Float, float(keyframes[-1][0])),
                      ("CustomCommandAnimLoop", Sdf.ValueTypeNames.Bool, False),
                      ("CustomCommandAnimBackwards", Sdf.ValueTypeNames.Bool, False)):
        prim.CreateAttribute(k, typ).Set(v)
    st.GetRootLayer().Save()


def profile_keys(idle, targets, profile):
    """Fotogramas cada 3 códigos de tiempo con slerp reposo -> objetivo según el perfil alfa."""
    j, r0, tcps = idle["joints"], idle["rot0"], idle["tcps"]
    idx = {x: i for i, x in enumerate(j)}
    end_f = int(round(profile[-1][0] * tcps))
    keys = []
    for f in sorted(set(list(range(0, end_f + 1, 3)) + [end_f])):
        a = profile_alpha(profile, f / tcps)
        keys.append((f / tcps, {x: slerp(r0[idx[x]], q, a) for x, q in targets.items()}))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", action="store_true", help="clips de prueba del brazo (Cand0..) y de la cabeza")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    idle = load_idle()
    if a.candidates:
        prof = VARIANTS["extend"]["profile"]
        for i, (r, c, b, h) in enumerate(CANDIDATES):
            write_clip(idle, f"{OUT}/cand{i}.skelanim.usd", f"Cand{i}", profile_keys(idle, targets_for(idle, r, c, b, h), prof))
        # cabeza: ±25° sobre cada eje local de Head, 1,5 s por segmento (pose fija en todo el segmento)
        hd = find(idle["joints"], "Head")
        r0h = idle["rot0"][idle["joints"].index(hd)]
        keys, t = [], 0.0
        for ax in (X, Y, Z):
            for s in (+1, -1):
                q = {hd: qmul(r0h, axis_angle(ax, s * 25.0))}
                keys += [(t, q), (t + 1.45, q)]
                t += 1.5
        write_clip(idle, f"{OUT}/head_probe.skelanim.usd", "HeadProbe", keys)
        print(f"{len(CANDIDATES)} candidatas + head_probe -> {OUT}")
        return
    tg = targets_for(idle, POSE["raise_deg"], POSE["across_deg"], POSE["bend_deg"], POSE["head"])
    for name, (command, profile) in CLIPS.items():
        write_clip(idle, f"{OUT}/{clip_file(name)}", command, profile_keys(idle, tg, profile))
    print(f"clips {list(CLIPS)} con {POSE} -> {OUT}")


if __name__ == "__main__":
    main()
