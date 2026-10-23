"""Backend MuJoCo (referencia): reproduce exactamente el episodio de CBF-repro.

Interfaz de backend que usa pabe.episode.G1Episode (la implementa también backends.isaac):
  reset()                      postura de pie, brazo y dedos sostenidos
  kin()                        cinemática del brazo: cápsulas (con jacobianos traslacionales de sus extremos),
                               palma (punto, jacobianos traslacional y rotacional), marcos de la muñeca y del torso
  arm_q(), arm_qd()            articulaciones del brazo derecho (7)
  arm_M(), arm_bias()          bloque del brazo de la matriz de masas y fuerzas de sesgo (Coriolis + gravedad)
  set_arm_torque(tau)          par del brazo (ya acotado por el episodio)
  set_grip(closed)             mano cerrada / abierta
  palm(), body_pos(name)       punto de la palma; origen del cuerpo 'forearm'
  apply_force(where, p, f)     fuerza externa en el punto p ('palm' o 'forearm'); se aplica en el siguiente step()
  step()                       un paso de física (2 ms)
  set_human(pose)              humano visual (sin física)
  qpos()                       estado completo (para reproducir el episodio)
  arm_lo, arm_hi, tau_lim      límites articulares y de par del brazo
"""
import os

import mujoco
import numpy as np

from pabe import geometry as G
from pabe.robot import FINGER_CLOSED, ROBOT
from .mujoco_model import G1Index, build, stand_qpos

MJB_PATH = os.path.join(os.path.dirname(__file__), "..", "third_party", f"g1_{ROBOT}_pabe_scene.mjb")
_MODEL = None


def get_model():
    """Modelo compilado una sola vez y cacheado en binario MJB (borrarlo tras cambiar mujoco_model.build)."""
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(MJB_PATH):
            m = build()
            tmp = MJB_PATH + f".{os.getpid()}.tmp"
            mujoco.mj_saveModel(m, tmp, None)
            os.replace(tmp, MJB_PATH)
        _MODEL = mujoco.MjModel.from_binary_path(MJB_PATH)
    return _MODEL


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros_like(v)


class MujocoBackend:
    name = "mujoco"

    def __init__(self):
        self.mm = mm = get_model()
        self.d = mujoco.MjData(mm)
        self.ix = ix = G1Index(mm)
        self.arm_lo, self.arm_hi = ix.arm_lo, ix.arm_hi
        self.tau_lim = mm.actuator_ctrlrange[ix.arm_act, 1]
        self._M = np.zeros((mm.nv, mm.nv))
        self._forearm = ix.cap_bodies["forearm"][0]

    def reset(self):
        mm, d, ix = self.mm, self.d, self.ix
        mujoco.mj_resetData(mm, d)
        qs = stand_qpos(mm)
        d.qpos[:] = qs
        d.ctrl[:] = qs[ix.act_q]
        mujoco.mj_forward(mm, d)

    def kin(self):
        mm, d, ix = self.mm, self.d, self.ix
        jac = {}
        for k, (ba, bb, r) in ix.cap_bodies.items():
            for b in (ba, bb):
                if b not in jac:
                    jp = np.zeros((3, mm.nv))
                    mujoco.mj_jac(mm, d, jp, None, d.xpos[b], b)
                    jac[b] = jp[:, ix.arm_v]
        caps = {k: (d.xpos[ba].copy(), d.xpos[bb].copy(), r, jac[ba], jac[bb]) for k, (ba, bb, r) in ix.cap_bodies.items()}
        jp, jr = np.zeros((3, mm.nv)), np.zeros((3, mm.nv))
        palm = d.site_xpos[ix.palm_site].copy()
        mujoco.mj_jac(mm, d, jp, jr, palm, ix.palm)
        return dict(caps=caps, palm=palm, Jp=jp[:, ix.arm_v], Jr=jr[:, ix.arm_v],
                    wrist_pos=d.xpos[ix.palm], wrist_R=d.xmat[ix.palm].reshape(3, 3),
                    torso_pos=d.xpos[ix.torso], torso_R=d.xmat[ix.torso].reshape(3, 3))

    def torso_pos(self):
        return self.d.xpos[self.ix.torso]

    def arm_q(self):
        return self.d.qpos[self.ix.arm_q].copy()

    def arm_qd(self):
        return self.d.qvel[self.ix.arm_v]

    def arm_M(self):
        mujoco.mj_fullM(self.mm, self.d, self._M)
        return self._M[np.ix_(self.ix.arm_v, self.ix.arm_v)]

    def arm_bias(self):
        return self.d.qfrc_bias[self.ix.arm_v]

    def set_arm_torque(self, tau):
        self.d.ctrl[self.ix.arm_act] = tau

    def set_grip(self, closed):
        self.d.ctrl[self.ix.fing_act] = FINGER_CLOSED if closed else 0.0

    def palm(self):
        return self.d.site_xpos[self.ix.palm_site]

    def body_pos(self, name):
        assert name == "forearm"
        return self.d.xpos[self._forearm]

    def apply_force(self, where, point, f):
        body = self.ix.palm if where == "palm" else self._forearm
        mujoco.mj_applyFT(self.mm, self.d, f, np.zeros(3), point, body, self.d.qfrc_applied)

    def step(self):
        mujoco.mj_step(self.mm, self.d)
        self.d.qfrc_applied[:] = 0.0

    def qpos(self):
        return self.d.qpos.copy()

    def set_human(self, pose, palm_face):
        mm, d, ix = self.mm, self.d, self.ix
        caps = G.human_capsules(pose)
        for name, key in (("h_uarm", "uarm"), ("h_farm", "farm")):
            a, b, _ = caps[key]
            d.mocap_pos[ix.mocap[name]] = 0.5 * (a + b)
            d.mocap_quat[ix.mocap[name]] = G.quat_from_z(b - a)
        d.mocap_pos[ix.mocap["h_hand"]] = pose["hand"]
        # elipsoide de la mano (sólo visual; la física usa una esfera): eje z a lo largo de la mano y eje fino (y,
        # normal de la palma) horizontal, mirando a la palma del robot: saludo con el pulgar arriba
        z = unit(pose["hand"] - pose["wrist"])
        y = unit(-palm_face - (-palm_face @ z) * z)
        Rh = np.column_stack([G.cross3(y, z), y, z])
        mujoco.mju_mat2Quat(d.mocap_quat[ix.mocap["h_hand"]], Rh.flatten())
        o = pose.get("off", np.zeros(3))
        ta, tb, _ = G.H_TORSO
        d.mocap_pos[ix.mocap["h_torso"]] = 0.5 * (ta + tb) + o
        d.mocap_pos[ix.mocap["h_head"]] = G.H_HEAD[0] + o
        d.mocap_pos[ix.mocap["h_legs"]] = np.array([0.78, 0.0, 0.48]) + o
