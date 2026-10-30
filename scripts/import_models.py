"""Importa models.pkl de CBF-repro (clases en pabe_cbf.*) al paquete pabe de este proyecto.
Uso: <python> scripts/import_models.py /mnt/data/belief-entropy/mujoco/results/g1/models.pkl results/models_g1.pkl"""
import os
import pickle
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "vendor")]


class _U(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("pabe_cbf"):
            module = "pabe." + module.split(".")[-1] if module != "pabe_cbf" else "pabe"
        elif module.startswith("numpy._core"):                # numpy 2 -> 1 (Python de Isaac)
            try:
                __import__("numpy._core")
            except ImportError:
                module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


src, dst = sys.argv[1:3]
m = _U(open(src, "rb")).load()
pickle.dump(m, open(dst, "wb"))
print(f"{type(m).__module__}.{type(m).__name__} approach={getattr(m, 'approach', None) is not None} -> {dst}")
