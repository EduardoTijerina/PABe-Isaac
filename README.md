# PABe-Isaac

PABe-CBF (handshake safety filter + approach system) for the Unitree G1 with BrainCo Revo2 hands, with the
episode loop independent of the simulator. Two backends implement the same small interface:

| Backend | Python | Use |
| --- | --- | --- |
| `backends/mujoco.py` | CBF-repro venv (`/mnt/data/CBF-repro/.venv`) | reference; reproduces CBF-repro bit for bit |
| `backends/isaac.py` | Isaac Sim 5.1 (`~/isaac_env`), plus `vendor/` (quadprog) | PhysX, same robot from the same URDF |

The simulator-independent code is in `pabe/` (copied from `/mnt/data/CBF-repro/pabe_cbf`, October 2026):
the QP and methods (`core.py`), beliefs (`belief.py`), scenarios incl. walking (`scenarios.py`), the approach
system (`approach.py`), geometry, robot constants (`robot.py`) and the episode loop (`episode.py`).

## Setup

```bash
cd /mnt/data/PABe-Isaac
~/isaac_env/bin/pip install --no-cache-dir --no-deps --target vendor quadprog   # once
~/isaac_env/bin/python -u scripts/convert_urdf.py                               # URDF -> assets/g1_revo2.usd, once
# trained models (approach + handshake) from CBF-repro, for the walking scenario and B2..B9
~/isaac_env/bin/python scripts/import_models.py /mnt/data/CBF-repro/results/g1/models.pkl results/models_g1.pkl
```

The G1 + Revo2 URDF and the meshes it uses are in `robots/g1_with_revo2/` (Unitree meshes BSD-3-Clause, Revo2
meshes Apache-2.0; licences alongside); `PABE_REVO2_URDF` points elsewhere. The converted USD (`assets/`), the
imported models (`results/models_g1.pkl`) and the comparison results are committed too, so the Isaac side runs
without the conversion step; `vendor/` (quadprog built for the Isaac Python) is not. The MuJoCo Menagerie G1 is in
`third_party/unitree_g1` (BSD-3-Clause, see its `LICENSE`).

## Run

```bash
# same episodes on both simulators, then compare
PABE_ROBOT=revo2 /mnt/data/CBF-repro/.venv/bin/python scripts/run_episodes.py --backend mujoco --out results/cmp_mujoco.json
~/isaac_env/bin/python -u scripts/run_episodes.py --backend isaac --out results/cmp_isaac.json
python3 scripts/compare_backends.py results/cmp_mujoco.json results/cmp_isaac.json
# watch in the Isaac Sim window (slower)
~/isaac_env/bin/python -u scripts/run_episodes.py --backend isaac --families iv_hesitation --pids 4000 --methods B1 --gui --out results/gui.json
```

```bash
# walking scenario: both walk toward each other, the person hesitates (stop_back), then they shake hands
~/isaac_env/bin/python -u scripts/run_episodes.py --backend isaac --families x_walk_hesitation --pids 4006 --methods B1 \
    --models results/models_g1.pkl --approach P3 --gui --out results/gui_walk.json
```

In the window the walking legs are a kinematic animation (`pabe/walk_anim.py`); the physics has a fixed base.
Always pass `-u` to the Isaac Python: `SimulationApp.close()` exits before buffered output is flushed.

## Checks

| Script | What it verifies |
| --- | --- |
| `scripts/check_mujoco_parity.py` | the backend-based loop equals CBF-repro's original episode bit for bit (both robots, B0/B1/B2/B9) |
| `scripts/probe_isaac.py` | Isaac structure: 51 DOF, fixed base, pelvis height, palm point |
| `scripts/probe_isaac_vs_mujoco.py` | PhysX vs MuJoCo in the same poses: Jacobians, mass matrix, gravity torques, free fall |
| `scripts/probe_effort.py` | arm torques are applied as commanded |

## PhysX details that matter (found with the probes)

- Linear Jacobian rows are at each link's **centre of mass**, not its origin; `backends/isaac.py` shifts them.
- URDF links without `<inertial>` (finger tips, "touch" and sensor links) get a **default mass** in PhysX
  (+14 kg); they are set to negligible mass, as in MuJoCo.
- `world.reset()` **reloads drive gains from the USD** (stiffness 625): it is called once at start-up and episodes
  reset by setting joint states.
- The reported mass matrix **omits armature**; it is added back for computed torque.
- PhysX has no equivalent of MuJoCo's joint `frictionloss` (0.3 N·m): a known difference.

Phase 2 (not started): an Isaac Lab backend with many robots in parallel, for the full benchmark.
