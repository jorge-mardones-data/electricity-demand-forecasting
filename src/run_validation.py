"""Validacion temporal (ventanas W1-W3) con los hiperparametros ya seleccionados.

Reproduce results/validation_results.csv y results/global_comparison.csv SIN repetir la busqueda de
hiperparametros del notebook 03 (que tarda varias horas). Los hiperparametros fijados estan en
src/modeling.py (RIDGE_ALPHA, TREE_CCP_ALPHA, RF_PARAMS) y en results/selected_configurations.csv.

Uso (desde la raiz del repositorio):
    python src/run_validation.py
    python src/run_validation.py --salida carpeta_de_prueba --sin-rf     # prueba rapida, sin Random Forest
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import timing  # noqa: E402

timing.activar_backend()  # EYP_BACKEND=cpu|gpu; debe ir antes de importar scikit-learn
from modeling import (RESULTS_DIR, evaluate_predictions, load_data, make_partitions,  # noqa: E402
                      make_rf, make_ridge, make_tree)


def run(out_dir=RESULTS_DIR, with_rf=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    crono = timing.Cronometro("03_run_validation")
    crono.seccion("Carga y particiones")
    df = load_data()
    parts = make_partitions(df)
    crono.seccion("Ciclo W1-W3 completo y exportacion (incluye los ajustes de abajo)")

    procedures = [("Ridge", make_ridge), ("Decision Tree", make_tree)]
    if with_rf:
        procedures.append(("Random Forest", make_rf))

    frames = []
    for window, p in parts.items():
        tr, va = p["train"], p["validation"]
        frames.append(evaluate_predictions(tr, va, va["demand_mw"], "Persistence", window))
        frames.append(evaluate_predictions(tr, va, va["lag_168h"], "Weekly Seasonal", window))
        for name, factory in procedures:
            model, feats = factory()
            t0 = time.perf_counter()
            model.fit(tr[feats], tr["target_demand_mw"])
            pred = model.predict(va[feats])
            seg = time.perf_counter() - t0
            print(f"{window} {name}: {seg:.1f} s", flush=True)
            crono.registrar(f"{name} · {window} (ajuste + prediccion)", seg)
            frames.append(evaluate_predictions(tr, va, pred, name, window))

    validation = pd.concat(frames, ignore_index=True)[["model", "window", "country", "n", "MAE", "RMSE", "NMAE"]]
    validation = validation.sort_values(["model", "window", "country"]).reset_index(drop=True)
    validation.to_csv(out_dir / "validation_results.csv", index=False)

    macro = validation[validation["country"] == "Macro"]
    glob = macro.pivot(index="model", columns="window", values="NMAE").reindex(columns=["W1", "W2", "W3"])
    glob["Mean NMAE"] = glob.mean(axis=1)
    glob["SD NMAE"] = glob[["W1", "W2", "W3"]].std(axis=1, ddof=1)
    glob = glob.sort_values("Mean NMAE")
    glob.to_csv(out_dir / "global_comparison.csv", index=True)
    print(glob.round(5))
    crono.guardar()
    return validation, glob


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--salida", default=str(RESULTS_DIR), help="carpeta de salida (por defecto results/)")
    ap.add_argument("--sin-rf", action="store_true", help="omite el Random Forest (solo para pruebas rapidas)")
    args = ap.parse_args()
    run(args.salida, with_rf=not args.sin_rf)
