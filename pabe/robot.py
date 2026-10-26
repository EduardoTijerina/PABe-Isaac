"""Constantes del robot comunes a todos los simuladores (sin dependencias de MuJoCo ni de Isaac).

Robot (variable de entorno PABE_ROBOT, leída al importar; los procesos hijos la heredan):
  revo2  URDF G1 29 GDL + manos BrainCo Revo2 (51 GDL) de PABE_REVO2_URDF (por defecto en este proyecto:
         es el único con recurso en Isaac Sim)
  g1     MuJoCo Menagerie, G1 29 GDL + manos Dex3 (43 GDL), el del manuscrito (sólo backend MuJoCo)
"""
import os

import numpy as np

ROBOTS = ("g1", "revo2")
ROBOT = os.environ.get("PABE_ROBOT", "revo2")
if ROBOT not in ROBOTS:
    raise ValueError(f"PABE_ROBOT={ROBOT!r}: debe ser uno de {ROBOTS}")
REVO2_URDF = os.environ.get("PABE_REVO2_URDF", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "robots", "g1_with_revo2", "g1_with_revo2_hands.urdf"))

ARM_JOINTS = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
              "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
# límites de par del brazo (N·m): muñeca pitch/yaw 5, el resto 25
ARM_TAU = np.array([5.0 if ("wrist_pitch" in n or "wrist_yaw" in n) else 25.0 for n in ARM_JOINTS])
# mano derecha: articulaciones accionadas en el agarre, su cierre y los cuerpos de la base de los dedos medio/índice
HANDS = {
    "g1": dict(joints=["right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
                       "right_hand_middle_0_joint", "right_hand_middle_1_joint", "right_hand_index_0_joint",
                       "right_hand_index_1_joint"],
               closed=np.array([0.0, 0.5, -0.9, 1.1, 1.2, 1.1, 1.2]),
               finger_base=("right_hand_middle_0_link", "right_hand_index_0_link"),
               palm_normal=(0.0, 1.0, 0.0)),
    # Revo2: las falanges distales siguen a las proximales (mimic del URDF)
    "revo2": dict(joints=["right_thumb_metacarpal_joint", "right_thumb_proximal_joint", "right_index_proximal_joint",
                          "right_middle_proximal_joint", "right_ring_proximal_joint", "right_pinky_proximal_joint"],
                  closed=np.array([0.3, 0.6, 0.9, 0.9, 0.9, 0.9]),
                  finger_base=("right_middle_proximal_link", "right_index_proximal_link"),
                  palm_normal=(0.0, 0.0, -1.0)),     # montada con giro de 90° (rpy 0 pi/2 0)
}
FINGER_JOINTS = HANDS[ROBOT]["joints"]
FINGER_CLOSED = HANDS[ROBOT]["closed"]
# normal de la palma en el marco de la muñeca (lado hacia el que se cierran los dedos; medido en el modelo)
PALM_NORMAL = np.array(HANDS[ROBOT]["palm_normal"])
# Cuerpos que delimitan las cápsulas de seguridad del robot (aproximación conservadora de los eslabones)
CAPSULES = {                       # nombre: (cuerpo_a, cuerpo_b, radio)
    "upperarm": ("right_shoulder_roll_link", "right_elbow_link", 0.050),
    "forearm": ("right_elbow_link", "right_wrist_roll_link", 0.045),
    "wrist": ("right_wrist_roll_link", "right_wrist_pitch_link", 0.042),   # sin el eslabón de la palma
}
PALM_BODY = "right_wrist_yaw_link"
TORSO_BODY = "torso_link"
PELVIS_Z = 0.793
PALM_FRAC = 0.65                   # centro de la palma: 65 % del camino muñeca -> base de los dedos medio/índice
# postura de pie: articulaciones a cero salvo rodilla/cadera levemente flexionadas y brazos relajados
STAND_POSE = [("left_hip_pitch_joint", -0.1), ("right_hip_pitch_joint", -0.1), ("left_knee_joint", 0.3),
              ("right_knee_joint", 0.3), ("left_ankle_pitch_joint", -0.2), ("right_ankle_pitch_joint", -0.2),
              ("left_shoulder_roll_joint", 0.25), ("right_shoulder_roll_joint", -0.30),
              ("left_elbow_joint", 0.9), ("right_elbow_joint", 0.9), ("right_shoulder_pitch_joint", 0.25),
              ("left_shoulder_pitch_joint", 0.25)]


def results_dir(backend):
    return f"results/{backend}_{ROBOT}"


def check_results_dir(out, backend):
    """Impide mezclar resultados (y puntos de control) de robots o simuladores distintos en un mismo directorio."""
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "robot.txt")
    tag = f"{backend}_{ROBOT}"
    if os.path.exists(path):
        prev = open(path).read().strip()
        if prev != tag:
            raise RuntimeError(f"{out} contiene resultados de {prev}, no de {tag}; usa --out {results_dir(backend)}")
    else:
        with open(path, "w") as f:
            f.write(tag + "\n")
