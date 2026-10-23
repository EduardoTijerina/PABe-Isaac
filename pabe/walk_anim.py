"""Animación de marcha para la repetición visual (misma que pabe_cbf/g1/walk_in.py de CBF-repro).

La simulación tiene base fija y no hay controlador de locomoción: la marcha es una oscilación cinemática de piernas
(y brazo izquierdo del robot) ligada a la distancia recorrida, que solo se aplica al dibujar. El brazo derecho y
el saludo son los del episodio físico.
"""
import numpy as np

from .scenarios import _knots_interp

STRIDE_R = 0.8       # ciclo de marcha completo (dos pasos) del G1 (m)
STRIDE_H = 1.2       # ídem, persona (pasos de 0,6 m)
V_GAIT = 0.5         # velocidad a la que la oscilación de marcha alcanza su amplitud completa (m/s)
BOB = 0.012          # descenso vertical de la pelvis a mitad de paso (m)
H_HIP = np.array([0.78, 0.09, 0.88])   # cadera derecha humana en su puesto (la izquierda es simétrica en y)
H_LEG_LEN = 0.84


def gait(knots, t, stride):
    """Amplitud (velocidad normalizada) y fase (ligada a la distancia recorrida: el pie no resbala)."""
    e = 1e-3
    v = abs(_knots_interp(t + e, knots) - _knots_interp(t - e, knots)) / (2 * e)
    return min(1.0, v / V_GAIT), 2 * np.pi * (knots[0][1] - _knots_interp(t, knots)) / stride


def robot_knots(record):
    """Marcha reactiva del robot registrada en el episodio (nodos con velocidad, cada 5 ciclos)."""
    ts = np.array([x["t"] for x in record]); xs = np.array([x["x_r"] for x in record])
    vs = np.gradient(xs, ts) if len(ts) > 1 else np.zeros_like(xs)
    idx = sorted(set(range(0, len(ts), 5)) | {len(ts) - 1})
    return [(float(ts[i]), float(xs[i]), float(vs[i])) for i in idx]


def robot_gait(q, dof, ph, env):
    """Oscilación de marcha sobre la postura (env = 0: sin cambios). No toca el brazo derecho.
    dof: nombre de articulación -> índice en q."""
    sn, cs = np.sin(ph), np.cos(ph)
    for side, sgn in (("right", 1.0), ("left", -1.0)):
        swing = max(0.0, sgn * cs)                      # fase de vuelo de esa pierna
        q[dof[f"{side}_hip_pitch_joint"]] -= 0.35 * sgn * sn * env
        q[dof[f"{side}_knee_joint"]] += 0.6 * swing * env
        q[dof[f"{side}_ankle_pitch_joint"]] -= 0.3 * swing * env
    q[dof["left_shoulder_pitch_joint"]] -= 0.3 * sn * env  # brazo opuesto a la pierna
    return q


def human_legs(x_h, ph, env, dz):
    """Piernas de la persona (marco del mundo, robot en su puesto): [(centro, eje unitario)] derecha e izquierda."""
    out = []
    for sgn in (1.0, -1.0):
        a = 0.35 * env * np.sin(ph) * sgn
        hip = np.array([H_HIP[0] + x_h, sgn * H_HIP[1], H_HIP[2] + dz])
        u = np.array([-np.sin(a), 0.0, -np.cos(a)])    # la persona mira hacia -x
        out.append((hip + 0.5 * H_LEG_LEN * u, u))
    return out
