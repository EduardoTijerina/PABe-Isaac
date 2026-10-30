"""Persona animada (personaje Renderpeople + IRA de ~/gr00t_project) dentro del mundo del backend Isaac.

Reutiliza sin copiar el código de la escena de recogida de datos de GR00T (~/gr00t_project/stage2_collect:
human_ira, handshake_variants), así que la persona hace exactamente las variantes con que se generaron los datos:
extend, hesitate, withdraw, partial.

  setup_scene(world, stage)    escena de oficina, luces de relleno, NavMesh y personaje (antes de world.reset())
  LiveHuman                    registro, calentamiento, colocación, comando de la variante y muestreo del esqueleto
  LiveBackend                  envoltorio del IsaacBackend: tras cada ciclo de control (5 pasos de física) actualiza
                               la aplicación (anima el personaje) y muestrea su esqueleto
  LiveHumanScenario            el Scenario que lee el episodio: mano y cuerpo desde el esqueleto vivo, intención
                               verdadera (z, autorización, t_W) desde los perfiles de handshake_variants

Requisitos: SimulationApp con la experiencia isaacsim.exp.action_and_event_data_generation.base.kit (sin ella el
grafo de animación nunca se inicializa; ver human_ira.py).
"""
import os
import sys

import numpy as np
import yaml

from pabe import geometry as G
from pabe.scenarios import Participant, Scenario

GR00T_DIR = os.path.expanduser("~/gr00t_project")
sys.path.insert(0, os.path.join(GR00T_DIR, "stage2_collect"))
import handshake_variants as HV  # noqa: E402  (numpy puro)
from groot.families import CLIPS, FAMILIES, T_FULL, clip_file  # noqa: E402

CFG_PATH = os.path.join(GR00T_DIR, "configs", "scene_config.yaml")
CTRL_DT = 0.01                        # periodo de control del episodio (pabe.episode.DT)
HAND_FWD = 0.08                       # muñeca -> centro de la mano, a lo largo del antebrazo
# retardo entre la orden de la variante y el inicio visible del clip, medido con el paso del backend
# (scripts/probe_live_human.py: 50 % del avance a 0,89 s frente a 0,90 s del perfil). El 0,5 s de scene_config.yaml
# corresponde a los fotogramas grabados de la recogida de datos, no a este lazo.
ANIM_LATENCY = 0.0


CLIPS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "human_clips"))


def load_cfg(own_clips=True):
    """Configuración de la escena de GR00T. own_clips: usar los clips de esta simulación (groot/author_clips.py:
    brazo que no atraviesa el pecho y cabeza que mira al G1) en vez de los de ~/gr00t_project/assets, con los
    mismos nombres de orden."""
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    if own_clips:
        m3 = cfg["milestone3"]
        for name in CLIPS:                      # todos los clips de groot/families.py, registrados al crear al personaje
            path = os.path.join(CLIPS_DIR, clip_file(name))
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path}: generar con groot/author_clips.py")
            if name == "extend":
                m3["extend"]["anim_usd"] = path
            else:
                m3["variants"]["clips"].setdefault(name, {})["anim_usd"] = path
    return cfg


def setup_scene(world, stage, cfg, office=True):
    """Oficina (misma colocación que la escena de GR00T), luces y personaje con sus comandos de variante.
    Devuelve (prim, agent_name, skelroot). Llamar como pre_reset del IsaacBackend."""
    from isaacsim.storage.native import get_assets_root_path
    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import Gf, UsdGeom, UsdLux
    import human_ira

    root = get_assets_root_path()
    sc = cfg["scene"]
    if office:
        add_reference_to_stage(usd_path=root + sc["room_usd"], prim_path="/World/Room")
        room = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Room"))
        room.ClearXformOpOrder()
        room.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in sc.get("room_translate", [0, 0, 0])]))
        if float(sc.get("room_rotate_z_deg", 0.0) or 0.0):
            room.AddRotateZOp().Set(float(sc["room_rotate_z_deg"]))
        fill = float(sc.get("dome_light", 0) or 0)
        for name, pos in (("EncounterFill", (0.0, 0.0, 2.5)), ("WalkInFill", (2.2, 0.0, 2.5))):
            sl = UsdLux.SphereLight.Define(stage, f"/World/{name}")
            sl.CreateRadiusAttr().Set(0.4)
            sl.CreateIntensityAttr().Set(fill * 20.0)
            UsdGeom.Xformable(sl.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*pos))
    human_ira.enable_ira_extensions()
    human_ira.exclude_from_navmesh("/World/G1")
    if office:
        human_ira.ensure_navmesh_volume(stage, "/World/Room")
        human_ira.bake_navmesh()
    prim, agent = human_ira.spawn_human(cfg, root, attach_behavior=True)
    return prim, agent, human_ira.find_skelroot_path(prim)


def _skeleton_tokens(skelroot):
    import omni.usd
    from pxr import Usd, UsdSkel
    stage = omni.usd.get_context().get_stage()
    for p in Usd.PrimRange(stage.GetPrimAtPath(skelroot)):
        if p.IsA(UsdSkel.Skeleton):
            joints = UsdSkel.Skeleton(p).GetJointsAttr().Get()
            if joints:
                return [str(j).split("/")[-1] for j in joints]
    return []


def _right_joint(tokens, keys):
    """Articulación del lado derecho cuyo nombre corto termina en una de las claves (p. ej. 'lowerarm_r'); si no hay,
    la primera que la contenga. Lado derecho: '_r' final, 'r_' inicial o 'right' (no basta una 'r' inicial: los rigs
    de Renderpeople empiezan por 'rp_', ver human_ira.find_hand_joint_name)."""
    def right(low):
        return low.endswith("_r") or low.startswith("r_") or "right" in low
    for k in keys:
        for t in tokens:
            low = t.lower()
            if right(low) and (low.endswith(f"_{k}_r") or low == f"{k}_r" or low == f"r_{k}"
                               or low.endswith(f"right{k}") or low.endswith(f"right_{k}")):
                return t
    for k in keys:
        for t in tokens:
            low = t.lower()
            if right(low) and k in low and not any(f in low for f in ("index", "middle", "ring", "pinky", "thumb",
                                                                          "twist", "corrective")):
                return t
    return None


class LiveHuman:
    """El personaje vivo: registro con AgentManager, colocación, variantes y buffer temporal del esqueleto."""

    def __init__(self, world, agent, skelroot, cfg):
        import human_ira
        self.hi, self.world, self.agent, self.skelroot, self.cfg = human_ira, world, agent, skelroot, cfg
        self.latency = ANIM_LATENCY
        self.hold_s = float(cfg["milestone3"]["extend"].get("hold_s", 60.0))
        if not human_ira.wait_for_agent_registered(agent, lambda: world.step(render=True), max_frames=240):
            raise RuntimeError("el personaje no se registró en AgentManager (¿experiencia SDG?)")
        self.tokens = _skeleton_tokens(skelroot)
        self.j_wrist = human_ira.find_hand_joint_name(skelroot)
        self.j_elbow = _right_joint(self.tokens, ("lowerarm", "forearm", "elbow"))
        self.j_shoulder = _right_joint(self.tokens, ("upperarm", "shoulder"))
        if None in (self.j_wrist, self.j_elbow, self.j_shoulder):
            raise RuntimeError(f"articulaciones no encontradas: muñeca={self.j_wrist} codo={self.j_elbow} "
                               f"hombro={self.j_shoulder}; esqueleto: {self.tokens}")
        self.buf_t, self.buf = [], []
        self.log = []                         # órdenes del episodio [(t, orden)] (para repetirlo)

    # ---------------- control del personaje ----------------
    def joint(self, name):
        return self.hi.get_hand_world_position(self.skelroot, name)

    def warmup(self):
        """Ejecuta una vez la orden de saludo sin grabar (el rig personalizado deja de animarse tras un arranque en
        frío; ver collect_handshake_data.warmup_human)."""
        self.hi.send_custom_command(self.agent, HV.COMMAND_NAME, 2.0)
        for _ in range(40):
            self.world.step(render=True)
        self.rest()

    def rest(self, frames=60):
        """Vuelve a reposo (Idle) y deja que el grafo mezcle la transición."""
        self.hi.send_idle(self.agent)
        for _ in range(frames):
            self.world.render()

    def place(self, distance_m, bearing_rad, frames=20):
        stop, facing = self.hi.compute_stop_point(self.cfg, bearing_rad, distance_m)
        self.hi.set_character_world_pose(self.skelroot, stop, facing)
        for _ in range(frames):
            self.world.render()
        return stop, facing

    def start(self, clip):
        """Lanza el clip (nombre de groot/families.CLIPS) en t = 0 del episodio, con el buffer vacío."""
        self.buf_t, self.buf, self.log = [], [], []
        self.send(clip, 0.0)
        self.sample(0.0)

    def send(self, clip, t):
        """Orden del clip en t (interrumpe la actual: inject_command(force_inject=True))."""
        self.hi.send_custom_command(self.agent, CLIPS[clip][0], self.hold_s)
        self.log.append((t, clip))

    # ---------------- muestreo y consulta ----------------
    def sample(self, t):
        w, e, s = self.joint(self.j_wrist), self.joint(self.j_elbow), self.joint(self.j_shoulder)
        if w is None or e is None or s is None:
            if not self.buf:
                raise RuntimeError("sin pose del esqueleto (¿personaje desregistrado?)")
            w, e, s = self.buf[-1]
        self.buf_t.append(t)
        self.buf.append((w, e, s))

    def _interp(self, t, i):
        bt = self.buf_t
        if t <= bt[0]:
            return self.buf[0][i]
        n = len(bt) - 1
        if t >= bt[n]:                        # futuro: extrapolación con la última velocidad
            if n == 0:
                return self.buf[0][i]
            v = (self.buf[n][i] - self.buf[n - 1][i]) / max(bt[n] - bt[n - 1], 1e-9)
            return self.buf[n][i] + v * min(t - bt[n], 0.05)
        k = int(np.searchsorted(bt, t)) - 1
        a = (t - bt[k]) / max(bt[k + 1] - bt[k], 1e-9)
        return (1 - a) * self.buf[k][i] + a * self.buf[k + 1][i]

    def hand(self, t):
        w, e = self._interp(t, 0), self._interp(t, 1)
        d = w - e
        return w + HAND_FWD * d / (np.linalg.norm(d) + 1e-9)

    def shoulder(self, t):
        return self._interp(t, 2)


class LiveBackend:
    """IsaacBackend + persona viva. Cada render_every ciclos de control (sub pasos de física cada uno) actualiza la
    aplicación sin avanzar la física (world.render(): el grafo de animación sólo se evalúa en las actualizaciones de
    Kit) y muestrea el esqueleto; hand(t) interpola entre muestras.
      render_every  ciclos de control por actualización; el backend debe crearse con rendering_dt =
                    render_every·10 ms para que la animación siga en tiempo real (cambiarlo después invalida la física)
      viewport      False: la vista no se dibuja (lotes sin ventana ni vídeo); el grafo sigue evaluándose
      clip          clip (groot/families.CLIPS) que se lanza en cada reset() (t = 0 del episodio)"""

    def __init__(self, be, human, sub=5, render_every=1, viewport=True, clip=None):
        self.be, self.human, self.sub, self.every, self.clip = be, human, sub, render_every, clip
        self._n = 0
        if abs(be.world.get_rendering_dt() - render_every * CTRL_DT) > 1e-9:
            raise ValueError(f"crear el backend con rendering_dt={render_every * CTRL_DT} para render_every={render_every} "
                             f"(el mundo tiene {be.world.get_rendering_dt()})")
        if not viewport:
            from omni.kit.viewport.utility import get_active_viewport
            vp = get_active_viewport()
            if vp is not None:
                vp.updates_enabled = False

    def __getattr__(self, name):
        return getattr(self.be, name)

    def reset(self):
        self.be.reset()
        self._n = 0
        if self.clip is not None:
            self.human.start(self.clip)

    def step(self):
        self.be.step()
        self._n += 1
        if self._n % (self.sub * self.every) == 0:
            self.be.world.render()
            self.human.sample(self._n // self.sub * CTRL_DT)


class LiveHumanScenario(Scenario):
    """Scenario de PABe con la persona viva. Se construye como i_stable del participante (ruidos de percepción y de
    observación, mismo generador) y sustituye la cinemática y la intención verdadera según groot/families.FAMILIES:

      mano      centro de la mano del esqueleto (muñeca + 8 cm a lo largo del antebrazo), con el retardo y el ruido
                de percepción de siempre aplicados por el episodio; en vi, más el golpe lateral de 3 cm
      cuerpo    desplazamiento rígido que lleva el hombro del modelo (geometry.H_SHOULDER) al hombro derecho vivo; el
                brazo, el torso y la cabeza de las barreras se derivan como en el resto de familias
      z         E salvo las ventanas H de la familia y W desde t_W (retiradas, o la reacción en ix)
      v / vi    oclusión de 0,6 s cerca de la llegada y tacto retrasado / empujón en el antebrazo y golpe en la mano,
                con los mismos sorteos que pabe.scenarios respecto de la llegada de la mano (1,8 s)
      ix        persona reactiva: si el robot invade su espacio sin autorización o tira fuerte, el episodio llama a
                react(t) y la persona lanza el clip de retirada
    """

    def __init__(self, human, family, pid, ep):
        # duración máxima la de i_stable (P.T_MAX = 7 s, como en las familias guionizadas; los ruidos están
        # dimensionados para ella)
        super().__init__("i_stable", Participant(pid, geom=G.G1_GEOM), ep)
        f = FAMILIES[family]
        self.human, self.family, self.clip = human, family, f["clip"]
        self.knots, self.bumps, self.shake, self.ext_pulses = [], [], [np.inf, np.inf], []
        self.authorized = f.get("authorized", True)
        self.t_auth = f.get("t_auth") if self.authorized else None
        self.t_W = f.get("t_W")
        self.z_windows = list(f.get("z", []))
        self.t_lunge = f.get("t_lunge")
        self.ambiguous = f.get("ambiguous", False)
        self.reactive = f.get("reactive", False)
        self.tactile_delay = f.get("tactile_delay", 0.0)
        rng = np.random.default_rng([pid, ep, 4242])
        if f.get("occlusion"):
            t_occ = T_FULL - 0.2 + rng.uniform(0.0, 1.0)
            self.occlusion = (t_occ, t_occ + 0.6)
        if f.get("perturb"):
            self.bumps.append((T_FULL + rng.uniform(0.8, 1.4), 0.2, 0.03 * self.g.lat * rng.choice([-1, 1])))
            self.ext_pulses = [(T_FULL + rng.uniform(0.3, 2.0), 0.15)]
        self.t_closure = None

    def hand(self, t):
        p = self.human.hand(t)
        for (tb, dur, vec) in self.bumps:
            if tb <= t <= tb + dur:
                p = p + vec * np.sin(np.pi * (t - tb) / dur) ** 2
        return p

    def body(self, t):
        return self.human.shoulder(t) - G.H_SHOULDER

    def closure(self, t):
        """El robot completó el apretón. La persona viva sigue su clip (no se retira por guion)."""
        self.t_closure = t

    def react(self, t):
        """(ix) Retirada provocada por el robot: z = W desde t y la persona lanza el clip de retirada."""
        if self.t_W is None:
            self.t_W = t
            self.human.send("retract", t)
