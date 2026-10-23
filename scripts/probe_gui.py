"""Prueba de ventana: abre Isaac Sim con el robot y mantiene la ventana renderizando; informa del estado de
is_running() para depurar el bucle de espera. Uso: DISPLAY=:0 ~/isaac_env/bin/python -u scripts/probe_gui.py"""
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "vendor")]
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": False})
from backends.isaac import IsaacBackend  # noqa: E402

be = IsaacBackend(human_visual=True)
be.reset()
print(f"[gui] after init: app.is_running()={app.is_running()} is_exiting={app.is_exiting()}", flush=True)
t0, n = time.time(), 0
while time.time() - t0 < 600:
    be.world.render()
    n += 1
    if n in (1, 10, 100) or n % 1000 == 0:
        print(f"[gui] frame {n}: is_running={app.is_running()} is_exiting={app.is_exiting()}", flush=True)
    if app.is_exiting():
        break
print(f"[gui] loop ended after {n} frames, {time.time() - t0:.0f} s", flush=True)
app.close()
