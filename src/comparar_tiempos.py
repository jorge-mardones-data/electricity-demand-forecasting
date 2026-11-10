"""Compara los tiempos por seccion de una corrida en CPU con una en GPU.

Lee todos los archivos results/tiempos/<notebook>__<backend>__<equipo>.csv y genera:
    results/tiempos/comparacion_tiempos.csv     una fila por notebook y seccion, con la aceleracion
    results/tiempos/resumen_por_notebook.csv    tiempo total de cada notebook en cada backend
    results/figures/tiempos_cpu_vs_gpu.png      grafico de las secciones mas lentas

Uso (desde la raiz del repositorio):
    python src/comparar_tiempos.py
    python src/comparar_tiempos.py --equipo-cpu MI-PC --equipo-gpu MI-PC   # si hay varios equipos
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TIEMPOS = ROOT / "results" / "tiempos"
FIGURES = ROOT / "results" / "figures"


def cargar(equipo_cpu=None, equipo_gpu=None):
    archivos = sorted(TIEMPOS.glob("*__*__*.csv"))
    if not archivos:
        raise SystemExit(f"No hay archivos de tiempos en {TIEMPOS}")
    df = pd.concat([pd.read_csv(f) for f in archivos], ignore_index=True)
    if equipo_cpu:
        df = df[(df.backend != "cpu") | (df.equipo == equipo_cpu)]
    if equipo_gpu:
        df = df[(df.backend != "gpu") | (df.equipo == equipo_gpu)]
    # si hay varias corridas del mismo notebook y backend, se usa la mas reciente
    ultima = df.groupby(["notebook", "backend"])["fecha"].transform("max")
    return df[df.fecha == ultima]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--equipo-cpu")
    ap.add_argument("--equipo-gpu")
    ap.add_argument("--top", type=int, default=15, help="secciones a mostrar en el grafico")
    args = ap.parse_args()

    df = cargar(args.equipo_cpu, args.equipo_gpu)
    faltan = {"cpu", "gpu"} - set(df.backend)
    if faltan:
        raise SystemExit(f"Falta la corrida en: {', '.join(sorted(faltan))}")

    comp = (df.pivot_table(index=["notebook", "orden", "seccion"], columns="backend",
                           values="segundos", aggfunc="first")
            .reset_index().rename(columns={"cpu": "seg_cpu", "gpu": "seg_gpu"}))
    comp["aceleracion_x"] = comp.seg_cpu / comp.seg_gpu
    comp["ahorro_seg"] = comp.seg_cpu - comp.seg_gpu
    comp = comp.sort_values(["notebook", "orden"])
    comp.to_csv(TIEMPOS / "comparacion_tiempos.csv", index=False)

    resumen = comp[comp.seccion == "TOTAL"][["notebook", "seg_cpu", "seg_gpu", "aceleracion_x", "ahorro_seg"]]
    resumen.to_csv(TIEMPOS / "resumen_por_notebook.csv", index=False)

    equipos = df.groupby("backend")[["equipo", "cpu", "gpu"]].first()
    print("Equipos:\n", equipos.to_string(), "\n")
    print("Tiempo total por notebook (segundos):")
    print(resumen.round(2).to_string(index=False))

    secc = comp[comp.seccion != "TOTAL"].dropna(subset=["seg_cpu", "seg_gpu"])
    secc = secc.nlargest(args.top, "seg_cpu").iloc[::-1]
    etiquetas = [f"{n[:2]} · {s[:45]}" for n, s in zip(secc.notebook, secc.seccion)]
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(secc) + 1.5))
    y = range(len(secc))
    ax.barh([i + 0.2 for i in y], secc.seg_cpu, height=0.4, label=f"CPU · {equipos.loc['cpu', 'cpu']}")
    ax.barh([i - 0.2 for i in y], secc.seg_gpu, height=0.4, label=f"GPU · {equipos.loc['gpu', 'gpu']}")
    for i, (c, g) in enumerate(zip(secc.seg_cpu, secc.seg_gpu)):
        ax.text(max(c, g) * 1.01, i, f"{c / g:.1f}x", va="center", fontsize=8)
    ax.set_yticks(list(y))
    ax.set_yticklabels(etiquetas, fontsize=8)
    ax.set_xlabel("Tiempo de reloj (segundos)")
    ax.set_title(f"Secciones más lentas en CPU y su tiempo en GPU (aceleración = CPU / GPU)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "tiempos_cpu_vs_gpu.png", dpi=150)
    print(f"\nGuardado: {TIEMPOS / 'comparacion_tiempos.csv'}, {TIEMPOS / 'resumen_por_notebook.csv'}, "
          f"{FIGURES / 'tiempos_cpu_vs_gpu.png'}")


if __name__ == "__main__":
    main()
