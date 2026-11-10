"""Medicion de tiempos por seccion y seleccion del backend de computo (CPU o GPU).

Uso dentro de un notebook (primera celda de codigo, ANTES de importar scikit-learn):

    import timing
    BACKEND = timing.activar_backend()          # lee la variable de entorno EYP_BACKEND
    CRONO = timing.Cronometro("03_model_development")
    CRONO.seccion("Preparacion")
    ...
    CRONO.seccion("M3 - Random Forest")
    ...
    CRONO.guardar()                             # results/tiempos/<notebook>__<backend>__<equipo>.csv

Backends:
    EYP_BACKEND=cpu (por defecto)  scikit-learn normal.
    EYP_BACKEND=gpu                activa cuml.accel (RAPIDS cuML) para que scikit-learn use la GPU NVIDIA
                                   sin cambiar el codigo. Requiere Linux o WSL2 con cuML instalado.
                                   Los estimadores o parametros que cuML no soporta se ejecutan en CPU.

Este modulo no importa scikit-learn, para poder activar la GPU antes de que se cargue.
"""
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TIEMPOS_DIR = PROJECT_ROOT / "results" / "tiempos"

_BACKEND = None


def activar_backend():
    """Activa el backend indicado en EYP_BACKEND ('cpu' o 'gpu') y lo devuelve."""
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    pedido = os.environ.get("EYP_BACKEND", "cpu").strip().lower()
    if pedido == "gpu":
        if "sklearn" in sys.modules:
            raise RuntimeError("scikit-learn ya estaba importado: active la GPU antes de importarlo "
                               "(primera celda del notebook o inicio del script).")
        try:
            import cuml.accel
        except ImportError as e:
            raise RuntimeError("EYP_BACKEND=gpu requiere RAPIDS cuML (Linux o WSL2 con GPU NVIDIA). "
                               "Ver docs/tiempos_cpu_gpu.md.") from e
        cuml.accel.install()
        _BACKEND = "gpu"
    elif pedido == "cpu":
        _BACKEND = "cpu"
    else:
        raise ValueError(f"EYP_BACKEND debe ser 'cpu' o 'gpu', no {pedido!r}")
    return _BACKEND


def info_gpu():
    """Nombre y memoria de la GPU NVIDIA, si nvidia-smi esta disponible."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True, timeout=15).strip().splitlines()[0]
        nombre, memoria, driver = [x.strip() for x in out.split(",")]
        return {"gpu": nombre, "gpu_memoria": memoria, "gpu_driver": driver}
    except Exception:
        return {"gpu": None, "gpu_memoria": None, "gpu_driver": None}


def info_cpu():
    nombre = platform.processor() or None
    try:
        if sys.platform.startswith("linux"):
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    nombre = line.split(":", 1)[1].strip()
                    break
        elif sys.platform.startswith("win"):
            nombre = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                text=True, timeout=20).strip().splitlines()[0].strip() or nombre
    except Exception:
        pass
    return {"cpu": nombre, "nucleos_logicos": os.cpu_count(), "sistema": platform.platform()}


def _version(paquete):
    try:
        from importlib import metadata
        return metadata.version(paquete)
    except Exception:
        return None


class Cronometro:
    """Registra el tiempo de reloj (wall time) de secciones consecutivas de un notebook o script."""

    def __init__(self, nombre, backend=None):
        self.nombre = nombre
        self.backend = backend or activar_backend()
        self.registros = []
        self._actual = None
        self._t0 = None
        self._inicio = time.perf_counter()
        self.fecha = datetime.now().isoformat(timespec="seconds")

    def _cerrar(self):
        if self._actual is not None:
            seg = time.perf_counter() - self._t0
            self.registros.append({"orden": len(self.registros) + 1, "seccion": self._actual,
                                   "segundos": round(seg, 2)})
            print(f"[tiempo] {self._actual}: {seg:.1f} s", flush=True)
        self._actual = None

    def seccion(self, nombre):
        """Cierra la seccion en curso (si hay) y empieza a medir la siguiente."""
        self._cerrar()
        self._actual = nombre
        self._t0 = time.perf_counter()

    def registrar(self, nombre, segundos):
        """Agrega un tiempo medido por fuera (por ejemplo, el ajuste de un modelo dentro de un ciclo)."""
        self.registros.append({"orden": len(self.registros) + 1, "seccion": nombre,
                               "segundos": round(segundos, 2)})

    def guardar(self):
        """Cierra la seccion en curso y guarda la tabla en results/tiempos/."""
        import pandas as pd

        self._cerrar()
        total = time.perf_counter() - self._inicio
        meta = {"notebook": self.nombre, "backend": self.backend, "equipo": socket.gethostname(),
                "fecha": self.fecha, "python": platform.python_version(),
                "scikit_learn": _version("scikit-learn"), "cuml": _version("cuml") or _version("cuml-cu12"),
                **info_cpu(), **info_gpu()}
        tabla = pd.DataFrame(self.registros)
        tabla = pd.concat([tabla, pd.DataFrame([{"orden": len(tabla) + 1, "seccion": "TOTAL",
                                                 "segundos": round(total, 2)}])], ignore_index=True)
        for k, v in meta.items():
            tabla[k] = v
        TIEMPOS_DIR.mkdir(parents=True, exist_ok=True)
        equipo = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in meta["equipo"])
        ruta = TIEMPOS_DIR / f"{self.nombre}__{self.backend}__{equipo}.csv"
        tabla.to_csv(ruta, index=False)
        print(f"[tiempo] total {total / 60:.1f} min · guardado en {ruta}", flush=True)
        return tabla[["orden", "seccion", "segundos"]]
