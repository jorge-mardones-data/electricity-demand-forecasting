"""Punto de entrada unico del analisis (EYP3407, Caso 03: demanda electrica horaria a 24 h).

Ejecuta, en orden y con un solo comando desde la raiz del repositorio:

    1. Auditoria de la base original            (src/audit_data.py)
    2. Base analitica: pais-hora, respuesta,    (src/build_dataset.py)
       predictores y periodos
    3. Validacion temporal W1-W3 con los        (src/run_validation.py)
       hiperparametros fijados
       (con --busqueda-completa: notebook 03, que repite la busqueda; tarda varias horas)
    4. Sensibilidades S1 y S2                   (notebooks/04_sensitivity_analysis.ipynb)
    5. Reserva 2019, bootstrap y diagnostico    (notebooks/05_diagnostics_uncertainty.ipynb)
    6. Analisis complementarios                 (notebooks/06_complementary_analyses.ipynb)
    7. Tablas y figuras del informe             (src/make_report_assets.py)
    8. Compilacion del informe en PDF           (xelatex, dos pasadas)

Uso:
    python analisis.py                         # ejecucion completa
    python analisis.py --solo-informe          # solo pasos 7 y 8 (usa los resultados ya existentes)
    python analisis.py --pasos 1,2,7,8         # pasos elegidos
    python analisis.py --busqueda-completa     # repite la busqueda de hiperparametros (notebook 03)
    python analisis.py --sin-pdf               # omite la compilacion LaTeX

Salidas:
    results/                       resultados (CSV) y figuras
    results/run_info.json          entorno, tiempos por etapa y tiempo total (solo si se ejecutan los pasos 1 a 6)
    results/concordancia.csv       diferencia entre los resultados regenerados y los que habia antes de ejecutar
    results/ejecucion/             registros (.log) y copias ejecutadas de los notebooks (no se versionan)
    reports/informe/main.pdf       informe

Los notebooks originales no se modifican: se ejecutan y su copia ejecutada se guarda en results/ejecucion/.
"""
import argparse
import asyncio
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
NB_DIR = ROOT / "notebooks"
RESULTS = ROOT / "results"
EJEC = RESULTS / "ejecucion"
INFORME = ROOT / "reports" / "informe"
KERNEL_NAME = "analisis_eyp3407"

# Resultados que se comparan con los existentes antes de la ejecucion (concordancia con el informe),
# con las columnas que identifican cada fila.
CONCORDANCIA = {
    "validation_results.csv": ["model", "window", "country"],
    "global_comparison.csv": ["model"],
    "sensitivity_results.csv": ["window", "model", "country", "scenario", "analysis"],
    "test_results.csv": ["window", "model", "country"],
    "test_main_table.csv": ["country"],
    "test_bootstrap_ci.csv": ["Unnamed: 0"],
    "test_bootstrap_extra.csv": ["bloque_dias", "comparador", "nivel"],
    "bootstrap_block_length.csv": ["bloque_dias", "comparador"],
    "test_diagnostics.csv": ["country", "grupo", "nivel"],
}


# ------------------------------------------------------------------------------ utilidades
def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def run_script(script, log_name, args=()):
    """Ejecuta un script de Python como subproceso y guarda su salida en results/ejecucion/."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    with open(EJEC / log_name, "w", encoding="utf-8") as f:
        proc = subprocess.run([sys.executable, str(script), *args], cwd=ROOT, env=env,
                              stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"{script.name} termino con error (ver {EJEC / log_name})")


def prepare_kernel():
    """Registra un kernel que usa este mismo interprete, para que los notebooks usen este entorno."""
    kdir = EJEC / "jupyter" / "kernels" / KERNEL_NAME
    kdir.mkdir(parents=True, exist_ok=True)
    spec = {"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": KERNEL_NAME, "language": "python"}
    (kdir / "kernel.json").write_text(json.dumps(spec), encoding="utf-8")
    os.environ["JUPYTER_PATH"] = str(EJEC / "jupyter") + os.pathsep + os.environ.get("JUPYTER_PATH", "")
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def run_notebook(name):
    """Ejecuta un notebook de principio a fin; guarda la copia ejecutada en results/ejecucion/."""
    import nbformat
    from nbclient import NotebookClient
    nb = nbformat.read(NB_DIR / name, as_version=4)
    NotebookClient(nb, timeout=None, kernel_name=KERNEL_NAME,
                   resources={"metadata": {"path": str(NB_DIR)}}).execute()
    nbformat.write(nb, EJEC / name)


def compile_pdf():
    if shutil.which("xelatex") is None:
        raise RuntimeError("No se encontro xelatex. Instale TeX Live o MiKTeX y vuelva a ejecutar "
                           "con --pasos 8, o use --sin-pdf.")
    with open(EJEC / "xelatex.log", "w", encoding="utf-8") as f:
        for _ in range(2):  # dos pasadas para resolver referencias cruzadas
            proc = subprocess.run(["xelatex", "-interaction=nonstopmode", "main.tex"], cwd=INFORME,
                                  stdout=f, stderr=subprocess.STDOUT)
    if not (INFORME / "main.pdf").exists():
        raise RuntimeError(f"No se genero el PDF (ver {EJEC / 'xelatex.log'}).")
    log(f"Informe: {INFORME / 'main.pdf'}")


# ------------------------------------------------------------------------------ pasos
def step_audit():
    run_script(SRC / "audit_data.py", "01_audit_data.log")


def step_build():
    run_script(SRC / "build_dataset.py", "02_build_dataset.log")


def step_validation(full_search):
    if full_search:
        log("Busqueda completa de hiperparametros (notebook 03): puede tardar varias horas.")
        run_notebook("03_model_development.ipynb")
    else:
        run_script(SRC / "run_validation.py", "03_run_validation.log")


def step_sens():
    run_notebook("04_sensitivity_analysis.ipynb")


def step_final():
    run_notebook("05_diagnostics_uncertainty.ipynb")


def step_complementary():
    run_notebook("06_complementary_analyses.ipynb")


def step_assets():
    run_script(SRC / "make_report_assets.py", "07_make_report_assets.log")


def step_pdf():
    compile_pdf()


def build_steps(full_search):
    return [
        (1, "Auditoria de la base original", step_audit),
        (2, "Base analitica", step_build),
        (3, "Validacion temporal W1-W3" + (" con busqueda completa" if full_search else ""),
         lambda: step_validation(full_search)),
        (4, "Sensibilidades (notebook 04)", step_sens),
        (5, "Reserva 2019, bootstrap y diagnostico (notebook 05)", step_final),
        (6, "Analisis complementarios (notebook 06)", step_complementary),
        (7, "Tablas y figuras del informe", step_assets),
        (8, "Compilacion del informe (PDF)", step_pdf),
    ]


# ------------------------------------------------------------------------------ registro del entorno
def ram_gb():
    try:
        if sys.platform.startswith("win"):
            import ctypes

            class MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

            ms = MemStatus()
            ms.dwLength = ctypes.sizeof(ms)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            return round(ms.ullTotalPhys / 2 ** 30, 1)
        if sys.platform.startswith("linux"):
            for line in open("/proc/meminfo"):
                if line.startswith("MemTotal"):
                    return round(int(line.split()[1]) / 2 ** 20, 1)
        if sys.platform == "darwin":
            return round(int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 2 ** 30, 1)
    except Exception:
        pass
    return None


def cpu_name():
    try:
        if sys.platform.startswith("win"):
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                text=True, timeout=20).strip()
            if out:
                return out.splitlines()[0].strip()
        if sys.platform.startswith("linux"):
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        pass
    return platform.processor() or None


def package_versions():
    out = {}
    for p in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "nbformat", "nbclient", "ipykernel"]:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def xelatex_version():
    try:
        return subprocess.check_output(["xelatex", "--version"], text=True, timeout=20).splitlines()[0]
    except Exception:
        return None


def snapshot_results():
    """Lee, antes de ejecutar, los resultados existentes que luego se comparan con los regenerados."""
    snap = {}
    for name in CONCORDANCIA:
        path = RESULTS / name
        if path.exists():
            snap[name] = pd.read_csv(path)
    return snap


def compare_results(snap):
    """Compara los resultados regenerados con los existentes antes de la ejecucion."""
    rows = []
    for name, old in snap.items():
        path = RESULTS / name
        if not path.exists():
            rows.append({"archivo": name, "estado": "no regenerado"})
            continue
        new = pd.read_csv(path)
        keys = [c for c in CONCORDANCIA[name] if c in old.columns and c in new.columns]
        num = [c for c in old.columns if c in new.columns and c not in keys and
               pd.api.types.is_numeric_dtype(old[c]) and pd.api.types.is_numeric_dtype(new[c])]
        try:
            merged = old.merge(new, on=keys, suffixes=("_ant", "_nuevo")) if keys else \
                pd.concat([old.add_suffix("_ant"), new.add_suffix("_nuevo")], axis=1)
        except Exception as e:  # noqa: BLE001
            rows.append({"archivo": name, "estado": f"no comparable ({e})"})
            continue
        def vmax(s):
            v = s.max(skipna=True)
            return 0.0 if pd.isna(v) else float(v)

        # se ignoran los tiempos de ajuste, que cambian en cada ejecucion
        num = [c for c in num if "seconds" not in c and "segundos" not in c]
        dif, rel = 0.0, 0.0
        for c in num:
            a, b = merged[f"{c}_ant"], merged[f"{c}_nuevo"]
            d = (a - b).abs()
            dif = max(dif, vmax(d))
            # diferencia relativa a la escala de la columna (evita dividir por valores cercanos a cero)
            scale = max(vmax(a.abs()), 1e-12)
            rel = max(rel, vmax(d) / scale)
        same_rows = len(merged) == len(old) == len(new)
        rows.append({"archivo": name, "filas_anteriores": len(old), "filas_nuevas": len(new),
                     "dif_abs_max": dif, "dif_rel_max_escala": rel,
                     "estado": "coincide (< 1e-6 de la escala)" if same_rows and rel < 1e-6 else
                               ("difiere levemente (< 1 % de la escala)" if same_rows and rel < 0.01 else "difiere")})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "concordancia.csv", index=False)
    return out


# ------------------------------------------------------------------------------ principal
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pasos", help="lista de pasos a ejecutar, por ejemplo 1,2,7,8 (por defecto todos)")
    ap.add_argument("--solo-informe", action="store_true", help="solo tablas, figuras y PDF (pasos 7 y 8)")
    ap.add_argument("--busqueda-completa", action="store_true",
                    help="repite la busqueda de hiperparametros ejecutando el notebook 03 (varias horas)")
    ap.add_argument("--sin-pdf", action="store_true", help="omite la compilacion del PDF (paso 8)")
    args = ap.parse_args()

    steps = build_steps(args.busqueda_completa)
    if args.solo_informe:
        chosen = {7, 8}
    elif args.pasos:
        chosen = {int(x) for x in args.pasos.split(",")}
    else:
        chosen = {s[0] for s in steps}
    if args.sin_pdf:
        chosen.discard(8)
    if chosen & {7, 8} and not (INFORME / "main.tex").exists():
        print("Aviso: reports/informe/ no esta en este repositorio todavia; se omiten los pasos 7 y 8.", flush=True)
        chosen -= {7, 8}

    EJEC.mkdir(parents=True, exist_ok=True)
    prepare_kernel()
    snap = snapshot_results()

    info = {"inicio": datetime.now().isoformat(timespec="seconds"),
            "comando": " ".join(sys.argv),
            "sistema": {"plataforma": platform.platform(), "procesador": cpu_name(),
                        "nucleos_logicos": os.cpu_count(), "memoria_gb": ram_gb()},
            "python": sys.version.split()[0], "paquetes": package_versions(), "xelatex": xelatex_version(),
            "etapas": []}
    t_total = time.perf_counter()
    status = "completo"
    try:
        for num_, name, fn in steps:
            if num_ not in chosen:
                continue
            log(f"Paso {num_}: {name}")
            t0 = time.perf_counter()
            fn()
            seg = time.perf_counter() - t0
            info["etapas"].append({"paso": num_, "nombre": name, "segundos": round(seg, 1)})
            log(f"  listo en {seg / 60:.1f} min")
    except Exception as e:  # noqa: BLE001
        status = f"con error: {e}"
        log(f"ERROR: {e}")
        raise
    finally:
        info["estado"] = status
        info["segundos_totales"] = round(time.perf_counter() - t_total, 1)
        info["fin"] = datetime.now().isoformat(timespec="seconds")
        # el registro de la ejecucion completa queda en results/run_info.json; las parciales no lo pisan
        full = set(range(1, 7)) <= chosen
        target = RESULTS / "run_info.json" if full else EJEC / "run_info_parcial.json"
        target.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"Registro guardado en {target} (total {info['segundos_totales'] / 60:.1f} min)")

    if chosen & {3, 4, 5, 6}:
        comp = compare_results(snap)
        log("Concordancia con los resultados previos (results/concordancia.csv):")
        print(comp.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
