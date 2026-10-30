"""Backend Isaac Sim 5.1 (PhysX): G1 + Revo2 desde assets/g1_revo2.usd (scripts/convert_urdf.py), base fija.

Misma interfaz que backends.mujoco.MujocoBackend (documentada allí). Equivalencias con el modelo MuJoCo:
  * física a 500 Hz (physics_dt = 2 ms), sin contactos (colisiones del robot desactivadas, sin suelo físico);
  * brazo derecho en par puro (ganancias 0) con par calculado del episodio: M y sesgo de PhysX;
  * resto de articulaciones con servos de posición kp = 500 y amortiguamiento crítico, armadura 0,01;
  * punto de la palma = PALM_FRAC del camino muñeca -> base de los dedos medio/índice, como en MuJoCo.
Diferencias conocidas: PhysX no tiene el frictionloss de MuJoCo (0,3 N·m) y su integrador es otro.

El SimulationApp debe crearse ANTES de importar este módulo (scripts/run_isaac_episode.py).
"""
import os

import numpy as np

from pabe import geometry as G
from pabe import walk_anim
from pabe.robot import (ARM_JOINTS, ARM_TAU, CAPSULES, FINGER_CLOSED, FINGER_JOINTS, HANDS, PALM_BODY, PALM_FRAC,
                        PELVIS_Z, REVO2_URDF, ROBOT, STAND_POSE, TORSO_BODY)

ASSET = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "g1_revo2.usd"))
ROBOT_PATH = "/World/G1"
DT_PHYS = 0.002
KP_HOLD, ARMATURE = 500.0, 0.01


def _quat_xyzw_to_R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _quats_xyzw_to_R(q):
    """_quat_xyzw_to_R para todos los eslabones a la vez (mismas operaciones elemento a elemento: mismo resultado)."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3), dtype=q.dtype)
    R[:, 0, 0], R[:, 0, 1], R[:, 0, 2] = 1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)
    R[:, 1, 0], R[:, 1, 1], R[:, 1, 2] = 2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)
    R[:, 2, 0], R[:, 2, 1], R[:, 2, 2] = 2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)
    return R


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros_like(v)


class IsaacBackend:
    name = "isaac"

    def __init__(self, usd=ASSET, human_visual=False, jac_at_com=True, pre_reset=None, floor=True,
                 rendering_dt=DT_PHYS * 5, usd_sync=True):
        """jac_at_com: las filas lineales del jacobiano de PhysX son las del centro de masas de cada eslabón
        (comprobado frente a MuJoCo en scripts/probe_isaac_vs_mujoco.py: error relativo 0,0000).
        pre_reset(world, stage): se llama justo antes del único world.reset() (p. ej. para añadir una escena y un
        personaje animado, groot/live_human.py). floor: suelo visual propio (False si la escena ya trae uno).
        rendering_dt: paso de cada world.render() (el grafo de animación avanza esto por actualización).
        usd_sync=False: PhysX no escribe el estado en USD tras cada paso (/physics/updateToUsd); el paso cuesta ~1,3 ms
        en vez de ~10 ms y la física es idéntica (scripts/probe_physics_cost.py). Para lotes sin ventana: la vista
        puede no reflejar el robot."""
        if ROBOT != "revo2":
            raise RuntimeError("el backend Isaac sólo tiene el recurso del G1 + Revo2 (PABE_ROBOT=revo2)")
        from isaacsim.core.api import World
        from isaacsim.core.prims import Articulation
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import Gf, UsdGeom, UsdLux, UsdPhysics
        import carb
        import omni.usd
        if not usd_sync:
            st = carb.settings.get_settings()
            st.set_bool("/physics/updateToUsd", False)
            st.set_bool("/physics/updateVelocitiesToUsd", False)

        self.world = World(physics_dt=DT_PHYS, rendering_dt=rendering_dt, stage_units_in_meters=1.0, backend="numpy")
        stage = omni.usd.get_context().get_stage()
        add_reference_to_stage(usd_path=usd, prim_path=ROBOT_PATH)
        xf = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT_PATH))
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, PELVIS_Z))
        # sin contactos, como en MuJoCo: se desactivan todas las colisiones del robot (no hay suelo físico)
        for prim in stage.Traverse():
            if str(prim.GetPath()).startswith(ROBOT_PATH) and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
        # eslabones sin <inertial> en el URDF (puntas, "touch", sensores): PhysX les daría masa por defecto
        # (+14 kg en total, varios en la mano derecha); como en MuJoCo, masa despreciable
        for name in _massless_links(REVO2_URDF):
            prim = stage.GetPrimAtPath(f"{ROBOT_PATH}/{name}")
            if prim.IsValid():
                mapi = UsdPhysics.MassAPI.Apply(prim)
                mapi.CreateMassAttr(1e-4)
                mapi.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-9, 1e-9, 1e-9))
        light = UsdLux.DistantLight.Define(stage, "/World/Light")
        light.CreateIntensityAttr(2500)
        if floor:
            fl = UsdGeom.Mesh.Define(stage, "/World/Floor")          # suelo sólo visual
            fl.CreatePointsAttr([(-5, -5, 0), (5, -5, 0), (5, 5, 0), (-5, 5, 0)])
            fl.CreateFaceVertexCountsAttr([4]); fl.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
            fl.CreateDisplayColorAttr([(0.15, 0.25, 0.35)])
        self.stage = stage
        self.human = self._make_human(stage) if human_visual else None
        self._n_steps = 0
        if human_visual:   # cámara desde el lado derecho de la persona (su brazo no queda tapado por el torso)
            from isaacsim.core.utils.viewports import set_camera_view
            set_camera_view(eye=[0.95, 1.55, 1.35], target=[0.42, 0.02, 0.98])

        self.art = Articulation(prim_paths_expr=ROBOT_PATH, name="g1")
        self.world.scene.add(self.art)
        if pre_reset is not None:
            pre_reset(self.world, stage)
        self.world.reset()
        art = self.art
        self.dof_names = list(art.dof_names)
        self.body_names = list(art.body_names)
        self.n_dof = len(self.dof_names)
        dof = {n: i for i, n in enumerate(self.dof_names)}
        body = {n: i for i, n in enumerate(self.body_names)}
        self.arm = np.array([dof[n] for n in ARM_JOINTS])
        self.fing = np.array([dof[n] for n in FINGER_JOINTS])
        self.cap_bodies = {k: (body[a], body[b], r) for k, (a, b, r) in CAPSULES.items()}
        self.b_palm, self.b_torso = body[PALM_BODY], body[TORSO_BODY]
        self.b_forearm = self.cap_bodies["forearm"][0]
        fb = HANDS[ROBOT]["finger_base"]
        self.b_fb = (body[fb[0]], body[fb[1]])
        lim = np.asarray(art.get_dof_limits())[0]
        self.arm_lo, self.arm_hi = lim[self.arm, 0], lim[self.arm, 1]
        self.tau_lim = ARM_TAU.copy()
        self.q_stand = np.zeros(self.n_dof)
        for n, v in STAND_POSE:
            if n in dof:
                self.q_stand[dof[n]] = v
        # jacobiano: base fija (L-1, 6, n) o flotante (L, 6, n+6)
        L = len(self.body_names)
        shape = tuple(art.get_jacobian_shape())[-3:]
        if shape == (L - 1, 6, self.n_dof):
            self.row_off, self.col_off = -1, 0
        elif shape == (L, 6, self.n_dof + 6):
            self.row_off, self.col_off = 0, 6
        else:
            raise RuntimeError(f"forma del jacobiano {shape} inesperada (L={L}, n={self.n_dof})")
        self.cols = self.arm + self.col_off
        # servos: brazo en par puro; el resto en posición con kp = 500 y amortiguamiento crítico; armadura como en
        # MuJoCo. Sólo en la vista física: world.reset() los recargaría del USD, así que no se vuelve a llamar
        # (cada episodio se reinicia fijando el estado articular, ver reset()).
        art.set_armatures(np.full((1, self.n_dof), ARMATURE))
        art.set_joint_positions(self.q_stand[None])
        M = np.asarray(art.get_mass_matrices())[0]
        diag = np.diag(M)[self.col_off:]
        kp = np.full(self.n_dof, KP_HOLD)
        kd = 2.0 * np.sqrt(KP_HOLD * np.maximum(diag, 1e-4))
        kp[self.arm], kd[self.arm] = 0.0, 0.0
        art.set_gains(kps=kp[None], kds=kd[None])
        self.jac_at_com = jac_at_com
        self._f = np.zeros((1, L, 3)); self._p = np.zeros((1, L, 3)); self._f_any = False
        self.palm_local = None

    # ------------------------------------------------------------------------------------------
    def _make_human(self, stage):
        from pxr import Gf, UsdGeom
        col, shirt, pants = (0.82, 0.70, 0.60), (0.55, 0.62, 0.72), (0.30, 0.32, 0.38)
        parts = {}
        for name, kind, size, c in (("torso", "capsule", (0.16, 0.48), shirt), ("head", "sphere", (0.10,), col),
                                    ("uarm", "capsule", (0.05, 0.30), shirt), ("farm", "capsule", (0.04, 0.27), col),
                                    ("hand", "ellipsoid", (0.045, 0.025, 0.06), col),
                                    ("rleg", "capsule", (0.065, 0.74), pants),
                                    ("lleg", "capsule", (0.065, 0.74), pants)):
            path = f"/World/Human/{name}"
            if kind == "capsule":
                g = UsdGeom.Capsule.Define(stage, path); g.CreateRadiusAttr(size[0]); g.CreateHeightAttr(size[1])
                g.CreateAxisAttr("Z")
            else:
                g = UsdGeom.Sphere.Define(stage, path); g.CreateRadiusAttr(1.0 if kind == "ellipsoid" else size[0])
            g.CreateDisplayColorAttr([c])
            xf = UsdGeom.Xformable(g.GetPrim())
            ops = (xf.AddTranslateOp(), xf.AddOrientOp(), xf.AddScaleOp())
            ops[2].Set(Gf.Vec3f(*size) if kind == "ellipsoid" else Gf.Vec3f(1, 1, 1))
            parts[name] = ops
        return parts

    def set_human(self, pose, palm_face):
        if self.human is None:
            return
        from pxr import Gf
        caps = G.human_capsules(pose)

        def put(name, pos, R=None):
            t, o, _ = self.human[name]
            t.Set(Gf.Vec3d(*map(float, pos)))
            if R is not None:
                q = _R_to_quat_wxyz(R)
                o.Set(Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3])))
        for name in ("uarm", "farm"):
            a, b, _ = caps[name]
            put(name, 0.5 * (a + b), _R_from_z(b - a))
        z = unit(pose["hand"] - pose["wrist"])
        y = unit(-palm_face - (-palm_face @ z) * z)
        put("hand", pose["hand"], np.column_stack([G.cross3(y, z), y, z]))
        o = pose.get("off", np.zeros(3))
        ta, tb, _ = G.H_TORSO
        put("torso", 0.5 * (ta + tb) + o)
        put("head", G.H_HEAD[0] + o)
        x0 = float(o[0]) if np.ndim(o) else 0.0
        self.set_human_legs(walk_anim.human_legs(x0, 0.0, 0.0, 0.0))

    def set_human_legs(self, legs):
        """Piernas de la persona: [(centro, eje unitario)] derecha e izquierda (ver pabe.walk_anim)."""
        from pxr import Gf
        for name, (c, u) in zip(("rleg", "lleg"), legs):
            t, o, _ = self.human[name]
            t.Set(Gf.Vec3d(*map(float, c)))
            q = _R_to_quat_wxyz(_R_from_z(u))
            o.Set(Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3])))

    # ------------------------------------------------------------------------------------------
    def reset(self):
        """Postura de pie en reposo (sin world.reset(): base fija y ningún otro cuerpo dinámico)."""
        art = self.art
        art.set_joint_positions(self.q_stand[None])
        art.set_joint_velocities(np.zeros((1, self.n_dof)))
        art.set_joint_position_targets(self.q_stand[None])
        art.set_joint_efforts(np.zeros((1, self.n_dof)))
        self._f[:] = 0.0; self._f_any = False
        self._grip = None
        self.set_grip(False)
        self._read()
        if self.palm_local is None:
            Rw, pw = self._R[self.b_palm], self._P[self.b_palm]
            base = 0.5 * sum(Rw.T @ (self._P[b] - pw) for b in self.b_fb)
            self.palm_local = PALM_FRAC * base

    def _read(self):
        tf = np.asarray(self.art._physics_view.get_link_transforms())[0]
        self._P = tf[:, :3].astype(float)
        self._R = _quats_xyzw_to_R(tf[:, 3:7])

    def _jac(self):
        return np.asarray(self.art.get_jacobians())[0]

    def _point_jac(self, J, b, p):
        """Jacobiano traslacional (3 x 7, columnas del brazo) del punto p fijo al cuerpo b."""
        row = J[b + self.row_off]
        Jl, Ja = row[:3, self.cols], row[3:, self.cols]
        o = self._com[b] if self.jac_at_com else self._P[b]
        return Jl - _skew(p - o) @ Ja

    def kin(self):
        self._read()
        if self.jac_at_com:
            self._com = self._coms()
        J = self._jac()
        jac = {}
        for k, (ba, bb, r) in self.cap_bodies.items():
            for b in (ba, bb):
                if b not in jac:
                    jac[b] = self._point_jac(J, b, self._P[b])
        caps = {k: (self._P[ba].copy(), self._P[bb].copy(), r, jac[ba], jac[bb]) for k, (ba, bb, r) in self.cap_bodies.items()}
        palm = self.palm()
        row = J[self.b_palm + self.row_off]
        return dict(caps=caps, palm=palm, Jp=self._point_jac(J, self.b_palm, palm), Jr=row[3:, self.cols],
                    wrist_pos=self._P[self.b_palm], wrist_R=self._R[self.b_palm],
                    torso_pos=self._P[self.b_torso], torso_R=self._R[self.b_torso])

    def _coms(self):
        c = np.asarray(self.art._physics_view.get_coms())[0]      # marco local de cada eslabón: (pos, quat)
        return [self._P[i] + self._R[i] @ c[i, :3] for i in range(len(self.body_names))]

    def palm(self):
        return self._P[self.b_palm] + self._R[self.b_palm] @ self.palm_local

    def body_pos(self, name):
        assert name == "forearm"
        return self._P[self.b_forearm]

    def arm_q(self):
        return np.asarray(self.art.get_joint_positions())[0, self.arm].astype(float)

    def arm_qd(self):
        return np.asarray(self.art.get_joint_velocities())[0, self.arm].astype(float)

    def arm_M(self):
        # PhysX simula la armadura pero no la incluye en la matriz que devuelve (MuJoCo sí): se suma aquí
        M = np.asarray(self.art.get_mass_matrices())[0]
        return M[np.ix_(self.cols, self.cols)] + ARMATURE * np.eye(len(self.cols))

    def arm_bias(self):
        c = np.asarray(self.art.get_coriolis_and_centrifugal_forces())[0]
        g = np.asarray(self.art.get_generalized_gravity_forces())[0]
        return (c + g)[self.cols]

    def set_arm_torque(self, tau):
        self.art.set_joint_efforts(np.asarray(tau, float)[None], joint_indices=self.arm)

    def set_grip(self, closed):
        if closed == self._grip:
            return
        self._grip = closed
        self.art.set_joint_position_targets((FINGER_CLOSED if closed else np.zeros(len(self.fing)))[None],
                                            joint_indices=self.fing)

    def apply_force(self, where, point, f):
        b = self.b_palm if where == "palm" else self.b_forearm
        self._f[0, b] += f
        self._p[0, b] = point
        self._f_any = True

    def step(self):
        if self._f_any:
            self.art._physics_view.apply_forces_and_torques_at_position(
                self._f.astype(np.float32), None, self._p.astype(np.float32), np.arange(1), True)
            self._f[:] = 0.0; self._f_any = False
        self.world.step(render=False)
        self._read()
        if self.human is not None:              # con ventana: se redibuja una vez por ciclo de control (10 ms)
            self._n_steps += 1
            if self._n_steps % 5 == 0:
                self.world.render()

    def qpos(self):
        return np.asarray(self.art.get_joint_positions())[0].copy()


def _massless_links(urdf):
    import xml.etree.ElementTree as ET
    return [l.get("name") for l in ET.parse(urdf).getroot().findall("link") if l.find("inertial") is None]


def _R_from_z(v):
    z = unit(v)
    a = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = unit(np.cross(a, z))
    return np.column_stack([x, np.cross(z, x), z])


def _R_to_quat_wxyz(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])
