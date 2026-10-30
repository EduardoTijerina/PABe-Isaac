"""Las familias de saludo de PABe (i-ix) y las variantes de GR00T representadas por la persona viva (Nathan).

CLIPS: clip de animación por nombre -> (orden de omni.anim.people, perfil alfa). El perfil es [(t_s, alfa)] con
interpolación coseno entre reposo (alfa 0) y la pose de saludo de groot/author_clips.POSE (alfa 1); alfa > 1
extrapola la pose (embestida). groot/author_clips.py genera un clip por entrada.

FAMILIES: familia -> qué clip lanza la persona en t = 0 y qué cambia en el Scenario de PABe (intención verdadera,
autorización, oclusión, empujones, persona reactiva). Tiempos medidos con la persona a 0,80 m y el método B1
(scripts/run_live_episode.py): mano extendida a 1,8 s, entrada en contacto (modo C) ~1,9 s, agarre ~3,05 s,
cierre 1,2 s después (P.T_SHAKE). Las retiradas en contacto (iii, viii) caen entre el agarre y el cierre.
Pure numpy: sin omni/pxr.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.expanduser("~/gr00t_project"), "stage2_collect"))
import handshake_variants as HV  # noqa: E402

from pabe.scenarios import H  # noqa: E402

T_FULL = 1.8                       # mano extendida del todo (perfil extend)
T_AUTH = 0.7 * T_FULL              # autorización al 70 % del alcance, como en pabe.scenarios
REACT_W = 0.2                      # retardo entre la decisión de retirarse (t_W) y el movimiento

CLIPS = {
    # variantes de GR00T (mismos perfiles y órdenes que handshake_variants)
    **{name: (v["command"], v["profile"]) for name, v in HV.VARIANTS.items()},
    # familias de PABe
    "withdraw_pre": ("WithdrawPre", [(0.0, 0.0), (1.0, 0.6), (1.2, 0.6), (2.0, 0.0), (6.0, 0.0)]),
    "withdraw_contact": ("WithdrawContact", [(0.0, 0.0), (T_FULL, 1.0), (3.5, 1.0), (4.3, 0.0), (7.0, 0.0)]),
    # tirón en contacto de ~2 cm (alfa 0,955; con 0,93 eran 3 cm, por encima del tope de 2,7 cm en que el muelle del
    # agarre supera la regla de tirón explícito de 8 N, ver pabe.scenarios iv_hesitation)
    # pausa más corta que la variante de GR00T (compromiso a 2,8 s en vez de 3,4): con ella el agarre llegaba a ~5 s y
    # el cierre caía después de los 7 s del episodio
    "hesitation": ("Hesitation", [(0.0, 0.0), (1.0, 0.55), (2.0, 0.55), (2.8, 1.0), (4.9, 1.0), (5.2, 0.955),
                                  (5.5, 1.0), (8.0, 1.0)]),
    "lunge": ("Lunge", [(0.0, 0.0), (1.0, 0.65), (1.15, 1.3), (1.45, 0.85), (2.0, 1.0), (7.0, 1.0)]),
    "ambiguous_withdraw": ("AmbiguousWithdraw", [(0.0, 0.0), (T_FULL, 1.0), (3.7, 1.0), (5.7, 0.0), (8.0, 0.0)]),
    "retract": ("Retract", [(0.0, 1.0), (0.7, 0.0), (3.0, 0.0)]),    # persona reactiva (ix): desde extendida
}

FAMILIES = {
    "i_stable": dict(clip="extend", t_auth=T_AUTH),
    "ii_withdraw_pre": dict(clip="withdraw_pre", authorized=False, t_W=1.2 - REACT_W),
    "iii_withdraw_contact": dict(clip="withdraw_contact", t_auth=T_AUTH, t_W=3.5 - REACT_W),
    "iv_hesitation": dict(clip="hesitation", t_auth=2.5, z=[(0.9, 2.0, H), (4.85, 5.5, H)]),
    "v_occlusion": dict(clip="extend", t_auth=T_AUTH, occlusion=True, tactile_delay=0.25),
    "vi_perturbation": dict(clip="extend", t_auth=T_AUTH, perturb=True),
    "vii_lunge": dict(clip="lunge", t_auth=T_AUTH, t_lunge=1.0),
    "viii_ambiguous_withdraw": dict(clip="ambiguous_withdraw", t_auth=T_AUTH, t_W=3.7 - 0.4, ambiguous=True),
    "ix_reactive": dict(clip="extend", t_auth=T_AUTH, reactive=True),
    # variantes de GR00T como escenarios (misma intención que en la recogida de datos)
    "extend": dict(clip="extend", t_auth=T_AUTH),
    "hesitate": dict(clip="hesitate", t_auth=HV.commit_time(HV.VARIANTS["hesitate"]["profile"]) - 0.3,
                     z=[(1.1, 2.4, H)]),
    "withdraw": dict(clip="withdraw", t_auth=T_AUTH, t_W=HV.retract_start(HV.VARIANTS["withdraw"]["profile"]) - REACT_W),
    "partial": dict(clip="partial", authorized=False, z=[(0.5, float("inf"), H)]),
}
PABE_FAMILIES = list(FAMILIES)[:9]


def clip_file(name):
    return f"handshake_{name}.skelanim.usd" if name in HV.VARIANTS else f"{name}.skelanim.usd"
