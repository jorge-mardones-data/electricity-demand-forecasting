"""Experimento: forzar los modelos a la GPU (RAPIDS cuML) y compararlos con scikit-learn en CPU.

A diferencia de `analisis.py --gpu` (cuml.accel, que devuelve a la CPU lo que no puede acelerar), este
script llama directamente a los estimadores de cuML, con datos densos en float32, para que el ajuste
ocurra SI O SI en la GPU. Solo usa el periodo de desarrollo (ventanas W1-W3); no toca la reserva 2019
ni los resultados oficiales.

Comparaciones (mismas filas, mismas variables y mismos hiperparametros en lo posible):
    Ridge                    sklearn (CPU)  vs. cuML (GPU)
    Random Forest dummy      sklearn oficial (max_depth=None) | sklearn max_depth=D | cuML max_depth=D
    Random Forest ciclico    idem
cuML exige una profundidad maxima finita, por eso se agrega sklearn con la misma profundidad D:
esa es la comparacion justa de velocidad. El arbol de decision podado no se incluye: cuML no tiene un
arbol unico con poda por costo-complejidad.

Uso (desde la raiz del repositorio, en Linux/WSL2 con cuML instalado):
    python src/benchmark_gpu.py                       # profundidades 16 y 32
    python src/benchmark_gpu.py --profundidades 16,24,32
    python src/benchmark_gpu.py --rapido              # prueba corta: 20 arboles
    python src/benchmark_gpu.py --sin-gpu             # solo la parte CPU (para probar el script)

Salidas:
    results/tiempos/benchmark_gpu__<equipo>.csv          una fila por modelo, motor y ventana
    results/tiempos/benchmark_gpu_resumen__<equipo>.csv  tiempo total, aceleracion y NMAE medio
    results/figures/benchmark_gpu.png                    tiempos y NMAE
"""
import argparse
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import timing  # noqa: E402
from modeling import (COUNTRIES, CYCLIC_FEATURES, NUMERIC_FEATURES, RF_PARAMS, RIDGE_ALPHA,  # noqa: E402
                      TARGET, load_data, make_partitions)

ROOT = Path(__file__).resolve().parent.parent
TIEMPOS = ROOT / "results" / "tiempos"
FIGURES = ROOT / "results" / "figures"

# Categorias fijas (no se estiman con datos: no hay fuga de informacion)
CATEGORIES = {
    "country": COUNTRIES,
    "target_hour": list(range(24)),
    "target_day": list(range(7)),
    "target_month": list(range(1, 13)),
}


# ------------------------------------------------------------------------------ datos densos
def dense_matrix(df, calendar):
    """Matriz densa float32: numericas + dummies (todas las categorias, como OneHotEncoder)."""
    if calendar == "dummy":
        num, cats = NUMERIC_FEATURES, ["country", "target_hour", "target_day", "target_month"]
    else:
        num, cats = NUMERIC_FEATURES + CYCLIC_FEATURES, ["country"]
    blocks = [df[num].to_numpy(dtype=np.float32)]
    for c in cats:
        values = df[c].to_numpy()
        blocks.append(np.stack([(values == k) for k in CATEGORIES[c]], axis=1).astype(np.float32))
    return np.ascontiguousarray(np.hstack(blocks)), len(num)


def nmae(train_df, valid_df, pred):
    out = {}
    for c in COUNTRIES:
        m = (valid_df["country"] == c).to_numpy()
        mae = np.mean(np.abs(valid_df[TARGET].to_numpy()[m] - pred[m]))
        out[f"NMAE_{c}"] = mae / train_df.loc[train_df["country"] == c, TARGET].mean()
    out["NMAE_macro"] = np.mean([out[f"NMAE_{c}"] for c in COUNTRIES])
    return out


def to_numpy(x):
    if hasattr(x, "get"):          # cupy
        x = x.get()
    if hasattr(x, "to_numpy"):     # cudf / pandas
        x = x.to_numpy()
    return np.asarray(x, dtype=np.float64).ravel()


# ------------------------------------------------------------------------------ modelos
def sk_ridge():
    from sklearn.linear_model import Ridge
    return Ridge(alpha=RIDGE_ALPHA)


def cu_ridge():
    from cuml.linear_model import Ridge
    return Ridge(alpha=RIDGE_ALPHA)


def sk_rf(depth, n_estimators):
    from sklearn.ensemble import RandomForestRegressor
    p = {**RF_PARAMS, "n_estimators": n_estimators, "max_depth": depth}
    return RandomForestRegressor(n_jobs=-1, **p)


def cu_rf(depth, n_estimators):
    from cuml.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=n_estimators, max_depth=depth,
                                 min_samples_leaf=RF_PARAMS["min_samples_leaf"],
                                 max_features=RF_PARAMS["max_features"], bootstrap=True,
                                 n_bins=256, random_state=RF_PARAMS["random_state"])


def warm_up_gpu():
    """Inicializa CUDA y cuML con un ajuste minimo, para no cargar ese costo a la primera ventana."""
    t0 = time.perf_counter()
    X = np.random.default_rng(0).random((2000, 10), dtype=np.float32)
    y = X.sum(axis=1)
    cu_ridge().fit(X, y).predict(X)
    cu_rf(8, 5).fit(X, y).predict(X)
    return time.perf_counter() - t0


# ------------------------------------------------------------------------------ principal
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profundidades", default="16,32", help="max_depth para cuML y sklearn emparejado")
    ap.add_argument("--rapido", action="store_true", help="20 arboles en vez de 200 (prueba corta)")
    ap.add_argument("--sin-gpu", action="store_true", help="omite cuML (para probar el script en CPU)")
    args = ap.parse_args()
    depths = [int(d) for d in args.profundidades.split(",")]
    n_est = 20 if args.rapido else RF_PARAMS["n_estimators"]
    use_gpu = not args.sin_gpu

    gpu_info = timing.info_gpu()
    cpu_info = timing.info_cpu()
    print(f"CPU: {cpu_info['cpu']} ({cpu_info['nucleos_logicos']} hilos) | GPU: {gpu_info['gpu']}", flush=True)

    warm = None
    if use_gpu:
        try:
            import cuml
        except ImportError:
            sys.exit("Este script necesita RAPIDS cuML. Use --sin-gpu para probar solo la parte CPU.")
        warm = warm_up_gpu()
        print(f"cuML {cuml.__version__} · inicializacion de la GPU: {warm:.1f} s (no se suma a los modelos)",
              flush=True)

    # (nombre, calendario, motor, fabrica)
    configs = [("Ridge", "dummy", "CPU sklearn", sk_ridge)]
    if use_gpu:
        configs.append(("Ridge", "dummy", "GPU cuML", cu_ridge))
    for cal in ["dummy", "cyclic"]:
        configs.append((f"RF {cal} · oficial (sin limite)", cal, "CPU sklearn", lambda: sk_rf(None, n_est)))
        for d in depths:
            configs.append((f"RF {cal} · max_depth={d}", cal, "CPU sklearn", lambda d=d: sk_rf(d, n_est)))
            if use_gpu:
                configs.append((f"RF {cal} · max_depth={d}", cal, "GPU cuML", lambda d=d: cu_rf(d, n_est)))

    df = load_data()
    parts = make_partitions(df)
    rows = []
    for window, p in parts.items():
        tr, va = p["train"], p["validation"]
        y_tr = tr[TARGET].to_numpy(dtype=np.float32)
        mats = {cal: (dense_matrix(tr, cal), dense_matrix(va, cal)) for cal in ["dummy", "cyclic"]}
        for model, cal, engine, factory in configs:
            (X_tr, n_num), (X_va, _) = mats[cal]
            if model == "Ridge":   # estandarizacion de las numericas con medias y desviaciones de entrenamiento
                mu, sd = X_tr[:, :n_num].mean(0), X_tr[:, :n_num].std(0)
                X_tr, X_va = X_tr.copy(), X_va.copy()
                X_tr[:, :n_num] = (X_tr[:, :n_num] - mu) / sd
                X_va[:, :n_num] = (X_va[:, :n_num] - mu) / sd
            est = factory()
            t0 = time.perf_counter()
            est.fit(X_tr, y_tr)
            t1 = time.perf_counter()
            pred = to_numpy(est.predict(X_va))
            t2 = time.perf_counter()
            row = {"ventana": window, "modelo": model, "calendario": cal, "motor": engine,
                   "n_train": len(tr), "n_valid": len(va), "n_columnas": X_tr.shape[1],
                   "seg_ajuste": round(t1 - t0, 3), "seg_prediccion": round(t2 - t1, 3),
                   "seg_total": round(t2 - t0, 3), **nmae(tr, va, pred)}
            rows.append(row)
            print(f"{window} | {model:32s} | {engine:11s} | {row['seg_total']:7.1f} s | "
                  f"NMAE {row['NMAE_macro']:.5f}", flush=True)

    res = pd.DataFrame(rows)
    meta = {"equipo": socket.gethostname(), "arboles": n_est, "seg_inicializacion_gpu": warm,
            **cpu_info, **gpu_info}
    for k, v in meta.items():
        res[k] = v
    TIEMPOS.mkdir(parents=True, exist_ok=True)
    equipo = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in meta["equipo"])
    res.to_csv(TIEMPOS / f"benchmark_gpu__{equipo}.csv", index=False)

    resumen = (res.groupby(["modelo", "motor"], sort=False)
               .agg(seg_total_W1_W3=("seg_total", "sum"), NMAE_macro_medio=("NMAE_macro", "mean"))
               .reset_index())
    cpu_ref = (resumen[resumen.motor == "CPU sklearn"].set_index("modelo")["seg_total_W1_W3"])
    resumen["aceleracion_vs_CPU_mismo_modelo"] = (resumen.modelo.map(cpu_ref) / resumen.seg_total_W1_W3).round(2)
    resumen.to_csv(TIEMPOS / f"benchmark_gpu_resumen__{equipo}.csv", index=False)
    print("\nResumen (suma de W1-W3):")
    print(resumen.round(5).to_string(index=False))

    # ---------------------------------------------------------------- figura
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = resumen.iloc[::-1].reset_index(drop=True)
    colors = ["tab:orange" if m.startswith("GPU") else "tab:blue" for m in r.motor]
    labels = [f"{m} · {e}" for m, e in zip(r.modelo, r.motor)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.42 * len(r) + 1.8), sharey=True)
    axes[0].barh(labels, r.seg_total_W1_W3, color=colors)
    for i, (s, a) in enumerate(zip(r.seg_total_W1_W3, r.aceleracion_vs_CPU_mismo_modelo)):
        txt = f"{s:.1f} s" + (f"  ({a:.1f}x)" if r.motor[i].startswith("GPU") else "")
        axes[0].text(s, i, " " + txt, va="center", fontsize=8)
    axes[0].set_xlabel("Ajuste + predicción, suma de W1–W3 (segundos)")
    axes[0].set_title("Tiempo")
    axes[1].barh(labels, r.NMAE_macro_medio, color=colors)
    axes[1].set_xlim(r.NMAE_macro_medio.min() * 0.9, r.NMAE_macro_medio.max() * 1.03)
    axes[1].set_xlabel("NMAE macro medio W1–W3 (adimensional)")
    axes[1].set_title("Error")
    for ax in axes:
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle(f"CPU ({cpu_info['cpu']}) vs. GPU ({gpu_info['gpu']}) · {n_est} árboles · "
                 "azul = CPU, naranjo = GPU", fontsize=10)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "benchmark_gpu.png", dpi=150)
    print(f"\nGuardado en {TIEMPOS} y {FIGURES / 'benchmark_gpu.png'}")


if __name__ == "__main__":
    main()
