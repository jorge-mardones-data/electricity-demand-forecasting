"""Genera las tablas (.tex) y figuras (.pdf) del informe a partir de results/ y data/processed/.

Salidas:
    reports/informe/generated/*.tex   tablas incluidas con \\input
    reports/informe/img/*.pdf         figuras

Uso (desde la raíz del repositorio):
    python src/make_report_assets.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "processed" / "demanda_base_analitica.csv"
OUT = ROOT / "reports" / "informe"
GEN = OUT / "generated"
IMG = OUT / "img"
GEN.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

MODELS = ["Persistence", "Weekly Seasonal", "Ridge", "Decision Tree", "Random Forest"]
LABELS = {
    "Persistence": "Persistencia",
    "Weekly Seasonal": "Estacional semanal",
    "Ridge": "M1: Ridge",
    "Decision Tree": "M2: Árbol podado",
    "Random Forest": "M3: Random Forest (dummies)",
}
# Paleta apta para daltonismo (Okabe-Ito); referencias en tonos neutros.
COLORS = {
    "Persistence": "#999999",
    "Weekly Seasonal": "#555555",
    "Ridge": "#0072B2",
    "Decision Tree": "#E69F00",
    "Random Forest": "#009E73",
}
PREDICTORS = [
    "country", "demand_mw", "lag_48h", "lag_72h", "lag_168h",
    "rolling_mean_24h", "rolling_mean_168h",
    "target_hour", "target_day", "target_month",
]
TARGET = "target_demand_mw"
REFERENCE = "Weekly Seasonal"


def num(x, d=4):
    """Número con coma decimal."""
    return f"{x:.{d}f}".replace(".", "{,}")


def integer(n):
    """Entero con espacio fino como separador de miles."""
    return f"{int(n):,}".replace(",", "\\,")


def pct(x, d=1):
    sign = "+" if x > 0 else ""
    return sign + f"{x:.{d}f}".replace(".", "{,}") + "\\%"


def write(name, text):
    (GEN / name).write_text(text, encoding="utf-8")
    print("escrito", GEN / name)


# ---------------------------------------------------------------- datos
val = pd.read_csv(RESULTS / "validation_results.csv")
glob = pd.read_csv(RESULTS / "global_comparison.csv").set_index("model")
sens = pd.read_csv(RESULTS / "sensitivity_results.csv", encoding="utf-8")
# NMAE macro del Random Forest con calendario cíclico en W1--W3 (sensibilidad S1)
cyc_w = (sens[(sens["analysis"] == "A_calendar") & (sens["model"] == "Random Forest")
              & (sens["country"] == "Macro") & (sens["scenario"] != "Dummy")]
         .sort_values("window")["NMAE"].to_numpy())
assert len(cyc_w) == 3
base = pd.read_csv(DATA, parse_dates=["utc_timestamp", "target_timestamp"])

# ------------------------------------------------- T1: construcción de la muestra
base["complete"] = base[PREDICTORS + [TARGET]].notna().all(axis=1)
base["periodo"] = base["period"].fillna("excluido")
rows = []
for period, plabel in [("development", "Periodo de desarrollo (2015--2018)"),
                       ("holdout", "Reserva (2019)"),
                       ("excluido", "Excluidos (31-dic-2018)")]:
    for c in ["DE", "ES", "FR"]:
        s = base[(base["periodo"] == period) & (base["country"] == c)]
        rows.append((plabel, c, len(s), s[TARGET].notna().sum(), s["complete"].sum()))
lines = []
last = None
for plabel, c, n, nt, nc in rows:
    first = plabel if plabel != last else ""
    last = plabel
    ncs = "--" if plabel.startswith("Excluidos") else integer(nc)
    lines.append(f"{first} & {c} & {integer(n)} & {integer(nt)} & {ncs} \\\\")
    if c == "FR":
        lines.append("\\midrule")
lines = lines[:-1]
write("tabla_muestra.tex", "\n".join([
    "\\begin{tabular}{llrrr}", "\\toprule",
    "Periodo (según año del origen) & País & Orígenes & Con respuesta & Casos completos \\\\",
    "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))

# ------------------------------------------------- T2: descripción de la respuesta (desarrollo)
dev = base[base["period"] == "development"]
lines = []
for c in ["DE", "ES", "FR"]:
    y = dev.loc[dev["country"] == c, TARGET].dropna()
    lines.append(f"{c} & {integer(len(y))} & {integer(round(y.mean()))} & "
                 f"{integer(round(y.std()))} & {integer(y.min())} & "
                 f"{integer(y.quantile(.5))} & {integer(y.max())} \\\\")
write("tabla_respuesta.tex", "\n".join([
    "\\begin{tabular}{lrrrrrr}", "\\toprule",
    "País & $n$ & Media & Desv. est. & Mínimo & Mediana & Máximo \\\\",
    "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))

# ------------------------------------------------- T3: comparación global (desarrollo)
ref_mean = glob.loc[REFERENCE, "Mean NMAE"]
lines = []
for m in MODELS:
    r = glob.loc[m]
    delta = 100 * (r["Mean NMAE"] / ref_mean - 1)
    d = "--" if m == REFERENCE else pct(delta)
    name = LABELS[m]
    lines.append(f"{name} & {num(r['W1'])} & {num(r['W2'])} & {num(r['W3'])} & "
                 f"{num(r['Mean NMAE'])} & {num(r['SD NMAE'], 4)} & {d} \\\\")
    if m == "Weekly Seasonal":
        lines.append("\\midrule")
lines.append(f"\\textbf{{M3$^\\ast$: Random Forest (cíclico)}} & {num(cyc_w[0])} & {num(cyc_w[1])} & "
             f"{num(cyc_w[2])} & {num(cyc_w.mean())} & {num(cyc_w.std(ddof=1), 4)} & "
             f"{pct(100 * (cyc_w.mean() / ref_mean - 1))} \\\\")
write("tabla_global.tex", "\n".join([
    "\\begin{tabular}{lrrrrrr}", "\\toprule",
    "Procedimiento & W1 & W2 & W3 & Media & Desv. est. entre ventanas & Cambio vs. estacional \\\\",
    "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))

# ------------------------------------------------- T4: MAE, RMSE y NMAE por país (media de ventanas)
c = val[val["country"] != "Macro"]
agg = c.groupby(["model", "country"])[["MAE", "RMSE", "NMAE"]].mean()
lines = []
for m in MODELS:
    cells = []
    for ctry in ["DE", "ES", "FR"]:
        r = agg.loc[(m, ctry)]
        cells.append(f"{integer(round(r['MAE']))} & {integer(round(r['RMSE']))} & {num(r['NMAE'], 4)}")
    name = LABELS[m]
    lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    if m == "Weekly Seasonal":
        lines.append("\\midrule")
write("tabla_pais.tex", "\n".join([
    "\\begin{tabular}{lrrrrrrrrr}", "\\toprule",
    " & \\multicolumn{3}{c}{Alemania (DE)} & \\multicolumn{3}{c}{España (ES)} & \\multicolumn{3}{c}{Francia (FR)} \\\\",
    "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
    "Procedimiento & MAE & RMSE & NMAE & MAE & RMSE & NMAE & MAE & RMSE & NMAE \\\\",
    "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))

# ------------------------------------------------- T5: diferencia M3 - referencia por ventana
mac = val[val["country"] == "Macro"].pivot(index="window", columns="model", values="NMAE")
if mac["Random Forest"].isna().any():
    mac["Random Forest"] = glob.loc["Random Forest", ["W1", "W2", "W3"]].values
lines = []
for w in ["W1", "W2", "W3"]:
    rf, ref = mac.loc[w, "Random Forest"], mac.loc[w, REFERENCE]
    lines.append(f"{w} & {num(ref)} & {num(rf)} & {num(rf - ref)} & {pct(100 * (rf / ref - 1))} \\\\")
write("tabla_diferencia.tex", "\n".join([
    "\\begin{tabular}{lrrrr}", "\\toprule",
    "Ventana & Estacional semanal & M3: Random Forest & Diferencia & Cambio relativo \\\\",
    "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))


# ------------------------------------------------- T6: detalle completo por modelo, ventana y país (anexo)
lines = []
for m in MODELS:
    for wd in ["W1", "W2", "W3"]:
        cells = []
        for ctry in ["DE", "ES", "FR"]:
            r = val[(val["model"] == m) & (val["window"] == wd) & (val["country"] == ctry)].iloc[0]
            cells.append(f"{integer(round(r['MAE']))} & {integer(round(r['RMSE']))} & {num(r['NMAE'], 4)}")
        macro = glob.loc[m, wd]
        name = LABELS[m] if wd == "W1" else ""
        lines.append(f"{name} & {wd} & " + " & ".join(cells) + f" & {num(macro, 4)} \\\\")
    if m != MODELS[-1]:
        lines.append("\\midrule")
write("tabla_detalle.tex", "\n".join([
    "\\begin{tabular}{llrrrrrrrrrr}", "\\toprule",
    " & & \\multicolumn{3}{c}{DE} & \\multicolumn{3}{c}{ES} & \\multicolumn{3}{c}{FR} & Macro \\\\",
    "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-11}",
    "Procedimiento & Ventana & MAE & RMSE & NMAE & MAE & RMSE & NMAE & MAE & RMSE & NMAE & NMAE \\\\",
    "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))

# ------------------------------------------------- T7: denominadores de evaluación (n por país y ventana)
nn = val[(val["model"] == "Ridge") & (val["country"] != "Macro")].pivot(index="window", columns="country", values="n")
lines = [f"{wd} & {integer(nn.loc[wd, 'DE'])} & {integer(nn.loc[wd, 'ES'])} & {integer(nn.loc[wd, 'FR'])} & "
         f"{integer(nn.loc[wd].sum())} \\\\" for wd in ["W1", "W2", "W3"]]
write("tabla_n.tex", "\n".join([
    "\\begin{tabular}{lrrrr}", "\\toprule",
    "Ventana & DE & ES & FR & Total \\\\", "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))

# ---------------------------------------------------------------- Figuras
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# F1: NMAE macro por ventana
fig, ax = plt.subplots(figsize=(6.2, 3.2))
for m in MODELS:
    ys = [glob.loc[m, w] for w in ["W1", "W2", "W3"]]
    ax.plot(["W1", "W2", "W3"], ys, marker="o", color=COLORS[m], label=LABELS[m],
            lw=2 if m == "Random Forest" else 1.5)
ax.plot(["W1", "W2", "W3"], cyc_w, marker="s", ls="--", color="#D55E00", lw=2,
        label="M3*: Random Forest (cíclico)")
ax.set_xlabel("Ventana de validación (W1: 2016, W2: 2017, W3: 2018)")
ax.set_ylabel("NMAE macro (adimensional)")
ax.set_ylim(0, 0.075)
ax.grid(axis="y", alpha=0.3)
ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.22))
fig.tight_layout()
fig.savefig(IMG / "fig_nmae_ventanas.pdf")
plt.close(fig)

# F2: NMAE por país (media de ventanas)
fig, ax = plt.subplots(figsize=(6.2, 3.2))
w = 0.16
x = np.arange(3)
for i, m in enumerate(MODELS):
    ys = [agg.loc[(m, ctry), "NMAE"] for ctry in ["DE", "ES", "FR"]]
    ax.bar(x + (i - 2) * w, ys, w, color=COLORS[m], label=LABELS[m])
ax.set_xticks(x)
ax.set_xticklabels(["Alemania (DE)", "España (ES)", "Francia (FR)"])
ax.set_ylabel("NMAE (media de W1--W3)")
ax.grid(axis="y", alpha=0.3)
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.25))
fig.tight_layout()
fig.savefig(IMG / "fig_nmae_paises.pdf")
plt.close(fig)

# ================================================================ Sensibilidades, 2019, incertidumbre y diagnóstico
C = ["DE", "ES", "FR"]
CCOL = {"DE": "#0072B2", "ES": "#D55E00", "FR": "#009E73"}


def snum(x, d=4):
    """Número con coma decimal y signo menos tipográfico."""
    return ("$-$" if x < 0 else "") + f"{abs(x):.{d}f}".replace(".", "{,}")


def sint(x):
    v = round(abs(x))
    return ("$-$" if x < 0 and v != 0 else "") + integer(v)


def pct0(x, d=1):
    return ("$-$" if x < 0 else "") + f"{abs(x):.{d}f}".replace(".", "{,}") + "\\%"


TEST_MODELS = ["Persistence", "Weekly Seasonal", "Ridge", "Decision Tree",
               "Random Forest (dummy)", "Random Forest"]
TLABEL = {**LABELS,
          "Random Forest (dummy)": "M3: Random Forest (dummies)",
          "Random Forest": "M3$^\\ast$: Random Forest (cíclico)"}

# ------------------------------------------------- T8: sensibilidades S1 (calendario) y S2 (historia)
sm = sens[sens["country"] == "Macro"]


base_a = {m: sm[(sm["analysis"] == "A_calendar") & (sm["model"] == m) & (sm["scenario"] == "Dummy")]
          .sort_values("window")["NMAE"].to_numpy() for m in ["Random Forest", "Ridge"]}
alt_a = {m: sm[(sm["analysis"] == "A_calendar") & (sm["model"] == m) & (sm["scenario"] != "Dummy")]
         .sort_values("window")["NMAE"].to_numpy() for m in ["Random Forest", "Ridge"]}
alt_b = {m: sm[(sm["analysis"] == "B_history") & (sm["model"] == m)]
         .sort_values("window")["NMAE"].to_numpy() for m in ["Random Forest", "Ridge"]}
MN = {"Random Forest": "Random Forest", "Ridge": "Ridge"}
lines = []
for m in ["Random Forest", "Ridge"]:
    b, a = base_a[m], alt_a[m]
    lines.append(f"S1: calendario & {MN[m]}, dummies (base) & {num(b[0])} & {num(b[1])} & {num(b[2])} & "
                 f"{num(b.mean())} & -- \\\\")
    lines.append(f" & {MN[m]}, cíclico & {num(a[0])} & {num(a[1])} & {num(a[2])} & "
                 f"{num(a.mean())} & {pct(100 * (a.mean() / b.mean() - 1))} \\\\")
lines.append("\\midrule")
for m in ["Random Forest", "Ridge"]:
    b, a = base_a[m], alt_b[m]
    lines.append(f"S2: historia$^\\dagger$ & {MN[m]}, ventana creciente (base) & {num(b[0])} & {num(b[1])} & "
                 f"{num(b[2])} & {num(b[1:].mean())} & -- \\\\")
    lines.append(f" & {MN[m]}, último año & {num(a[0])} & {num(a[1])} & {num(a[2])} & "
                 f"{num(a[1:].mean())} & {pct(100 * (a[1:].mean() / b[1:].mean() - 1))} \\\\")
write("tabla_sens.tex", "\n".join([
    "\\begin{tabular}{llrrrrr}", "\\toprule",
    "Análisis & Escenario & W1 & W2 & W3 & Promedio & Cambio vs. base \\\\", "\\midrule",
    *lines, "\\bottomrule", "\\end{tabular}"]))

# ------------------------------------------------- Anexo: sensibilidades S1 y S2 por país
sc = sens[sens["country"].isin(C)]


def sens_country(analysis, model, scenario, windows):
    d = sc[(sc["analysis"] == analysis) & (sc["model"] == model) & (sc["scenario"] == scenario)
           & (sc["window"].isin(windows))]
    return d.groupby("country")["NMAE"].mean().reindex(C)


def cell(v, b=None):
    return num(v) if b is None else f"{num(v)} ({pct(100 * (v / b - 1))})"


lines = []
for analysis, label, scen_base, scen_alt, alt_label, wins in [
        ("A_calendar", "S1: calendario", "Dummy", "Cíclico", "cíclico", ["W1", "W2", "W3"]),
        ("B_history", "S2: historia$^\\dagger$", "Dummy", "Último año", "último año", ["W2", "W3"])]:
    base_label = "dummies (base)" if analysis == "A_calendar" else "ventana creciente (base)"
    for m in ["Random Forest", "Ridge"]:
        b = sens_country("A_calendar", m, scen_base, wins)
        a = sens_country(analysis, m, scen_alt, wins)
        lines.append(f"{label} & {m}, {base_label} & " + " & ".join(cell(b[c]) for c in C) + " \\\\")
        lines.append(f" & {m}, {alt_label} & " + " & ".join(cell(a[c], b[c]) for c in C) + " \\\\")
    if analysis == "A_calendar":
        lines.append("\\midrule")
write("tabla_sens_pais.tex", "\n".join([
    "\\begin{tabular}{llrrr}", "\\toprule",
    "Análisis & Escenario & DE & ES & FR \\\\", "\\midrule",
    *lines, "\\bottomrule", "\\end{tabular}"]))

# ------------------------------------------------- Datos de la reserva 2019
test = pd.read_csv(RESULTS / "test_results.csv")
tm = test.set_index(["model", "country"])
pred = pd.read_csv(RESULTS / "test_predictions.csv", parse_dates=["utc_timestamp"])
diag = pd.read_csv(RESULTS / "test_diagnostics.csv")
hourly = pd.read_csv(RESULTS / "test_error_by_hour.csv")
worst = pd.read_csv(RESULTS / "test_worst_days.csv")
ftimes = pd.read_csv(RESULTS / "final_fit_times.csv", index_col=0).iloc[:, 0]

dev_macro = {"Persistence": glob.loc["Persistence", "Mean NMAE"],
             "Weekly Seasonal": glob.loc["Weekly Seasonal", "Mean NMAE"],
             "Ridge": glob.loc["Ridge", "Mean NMAE"],
             "Decision Tree": glob.loc["Decision Tree", "Mean NMAE"],
             "Random Forest (dummy)": glob.loc["Random Forest", "Mean NMAE"],
             "Random Forest": cyc_w.mean()}
ref_t = tm.loc[("Weekly Seasonal", "Macro"), "NMAE"]

# ------------------------------------------------- T9: comparación en la reserva 2019
lines = []
for m in TEST_MODELS:
    cells = " & ".join(num(tm.loc[(m, c), "NMAE"]) for c in C)
    macro = tm.loc[(m, "Macro"), "NMAE"]
    d = "--" if m == REFERENCE else pct(100 * (macro / ref_t - 1))
    name = TLABEL[m]
    mac = num(macro)
    if m == "Random Forest":
        name, mac = "\\textbf{" + name + "}", "\\textbf{" + mac + "}"
    lines.append(f"{name} & {cells} & {mac} & {num(dev_macro[m])} & {d} \\\\")
    if m == "Weekly Seasonal":
        lines.append("\\midrule")
write("tabla_test.tex", "\n".join([
    "\\begin{tabular}{lrrrrrr}", "\\toprule",
    "Procedimiento & NMAE DE & NMAE ES & NMAE FR & NMAE macro 2019 & NMAE macro del periodo de desarrollo (2015--2018) & Cambio vs. estacional (2019) \\\\",
    "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))

# ------------------------------------------------- T10: Random Forest cíclico vs. referencia por país (2019)
lines = []
for c in C:
    rr, rf = tm.loc[("Weekly Seasonal", c)], tm.loc[("Random Forest", c)]
    lines.append(f"{c} & {integer(rf['n'])} & {integer(round(rr['MAE']))} & {integer(round(rf['MAE']))} & "
                 f"{integer(round(rr['RMSE']))} & {integer(round(rf['RMSE']))} & {num(rr['NMAE'])} & "
                 f"{num(rf['NMAE'])} & {pct0(100 * (rf['NMAE'] / rr['NMAE'] - 1))} \\\\")
rr, rf = tm.loc[("Weekly Seasonal", "Macro")], tm.loc[("Random Forest", "Macro")]
lines.append("\\midrule")
lines.append(f"Macro & {integer(rf['n'])} & -- & -- & -- & -- & {num(rr['NMAE'])} & {num(rf['NMAE'])} & "
             f"{pct0(100 * (rf['NMAE'] / rr['NMAE'] - 1))} \\\\")
write("tabla_test_rf.tex", "\n".join([
    "\\begin{tabular}{lrrrrrrrr}", "\\toprule",
    " & & \\multicolumn{2}{c}{MAE (MW)} & \\multicolumn{2}{c}{RMSE (MW)} & \\multicolumn{2}{c}{NMAE} & \\\\",
    "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
    "País & $n$ & Estac. & RF & Estac. & RF & Estac. & RF & Cambio \\\\",
    "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))

# ------------------------------------------------- T11: detalle de la reserva por procedimiento y país (anexo)
lines = []
for m in TEST_MODELS:
    cells = []
    for c in C:
        r = tm.loc[(m, c)]
        cells.append(f"{integer(round(r['MAE']))} & {integer(round(r['RMSE']))} & {num(r['NMAE'])}")
    lines.append(f"{TLABEL[m]} & " + " & ".join(cells) + f" & {num(tm.loc[(m, 'Macro'), 'NMAE'])} \\\\")
    if m != TEST_MODELS[-1]:
        lines.append("\\midrule")
write("tabla_test_detalle.tex", "\n".join([
    "\\begin{tabular}{lrrrrrrrrrr}", "\\toprule",
    " & \\multicolumn{3}{c}{DE} & \\multicolumn{3}{c}{ES} & \\multicolumn{3}{c}{FR} & Macro \\\\",
    "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
    "Procedimiento & MAE & RMSE & NMAE & MAE & RMSE & NMAE & MAE & RMSE & NMAE & NMAE \\\\",
    "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))

# ------------------------------------------------- Tiempos de ajuste final
tn = {"Ridge": "M1: Ridge", "Decision Tree": "M2: Árbol podado",
      "Random Forest (dummy)": "M3: Random Forest (dummies)",
      "Random Forest": "M3$^\\ast$: Random Forest (cíclico)"}
lines = [f"{tn[k]} & {num(ftimes[k], 1)} \\\\" for k in tn]
write("tabla_tiempos.tex", "\n".join([
    "\\begin{tabular}{lr}", "\\toprule", "Procedimiento & Ajuste final (s) \\\\", "\\midrule",
    *lines, "\\bottomrule", "\\end{tabular}"]))

# ------------------------------------------------- Bootstrap por bloques semanales (predicciones fijas)
ft = base[(base["period"] == "development") & base["complete"]
          & (base["target_timestamp"] < pd.Timestamp("2019-01-01", tz="UTC"))]
mu = ft.groupby("country")[TARGET].mean()[C].to_numpy()
pred["block"] = (pred["utc_timestamp"] - pd.Timestamp("2019-01-01", tz="UTC")).dt.days // 7
nb = int(pred["block"].max()) + 1
PCOL = {"ref": "pred_weekly", "rf": "pred_rf", "dummy": "pred_rf_dummy",
        "ridge": "pred_ridge", "tree": "pred_tree"}


def block_sums(col):
    ae = (pred[TARGET] - pred[col]).abs()
    s = ae.groupby([pred["block"], pred["country"]]).sum().unstack()[C]
    return s.reindex(range(nb)).fillna(0).to_numpy()


NB = pred.groupby(["block", "country"]).size().unstack()[C].reindex(range(nb)).fillna(0).to_numpy()
AE = {k: block_sums(v) for k, v in PCOL.items()}


def stats(a, b, idx):
    n = NB[idx].sum(0)
    na, nbb = AE[a][idx].sum(0) / n / mu, AE[b][idx].sum(0) / n / mu
    return np.r_[na.mean() - nbb.mean(), na - nbb, 1 - na.mean() / nbb.mean(), 1 - na / nbb]


B = 2000
rng = np.random.default_rng(42)
draws = [rng.integers(0, nb, nb) for _ in range(B)]
COMP = {"ref": ("rf", "ref"), "ridge": ("rf", "ridge"), "tree": ("rf", "tree"), "dummy": ("rf", "dummy")}
boot_res = {}
for key, (a, b) in COMP.items():
    obs = stats(a, b, np.arange(nb))
    bs = np.array([stats(a, b, d) for d in draws])
    boot_res[key] = (obs, np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0),
                     (bs[:, :4] >= 0).mean(0), bs)

# verificación contra lo exportado por el notebook 05
chk = pd.read_csv(RESULTS / "test_bootstrap_ci.csv", index_col=0)
o, lo, hi, _, _ = boot_res["ref"]
mine = np.r_[o[:5]]
assert np.allclose(mine, chk["observado"].to_numpy(), atol=1e-9), "bootstrap: observado difiere del notebook"
assert np.allclose(np.r_[lo[:5]], chk["IC95 inf"].to_numpy(), atol=1e-6), "bootstrap: IC inf difiere del notebook"
assert np.allclose(np.r_[hi[:5]], chk["IC95 sup"].to_numpy(), atol=1e-6), "bootstrap: IC sup difiere del notebook"

rows = [("rf-ref-macro", "ref", 0, 4, "Macro vs. estacional semanal"),
        ("", "ref", 1, 5, "\\hspace{1em}DE"), ("", "ref", 2, 6, "\\hspace{1em}ES"),
        ("", "ref", 3, 7, "\\hspace{1em}FR"),
        ("", "ridge", 0, 4, "Macro vs. M1: Ridge"),
        ("", "tree", 0, 4, "Macro vs. M2: Árbol podado"),
        ("", "dummy", 0, 4, "Macro vs. M3: RF (dummies)")]
lines = []
for _, key, i, j, lab in rows:
    obs, lo, hi, pz, _ = boot_res[key]
    p = pz[i]
    ptxt = "$<0{,}001$" if p == 0 else num(p, 3)
    lines.append(f"{lab} & {snum(obs[i])} & [{snum(lo[i])}; {snum(hi[i])}] & {ptxt} & "
                 f"{pct0(100 * obs[j])} & [{pct0(100 * lo[j])}; {pct0(100 * hi[j])}] \\\\")
    if lab.endswith("FR"):
        lines.append("\\midrule")
write("tabla_boot.tex", "\n".join([
    "\\begin{tabular}{lrcrrc}", "\\toprule",
    "Comparación (M3$^\\ast$ menos el comparador) & $\\Delta$ NMAE & IC 95\\,\\% de $\\Delta$ & $P(\\Delta\\geq 0)$ & "
    "Mejora relativa & IC 95\\,\\% de la mejora \\\\",
    "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))

# ------------------------------------------------- T12: diagnóstico por subgrupo (cuerpo) y detalle (anexo)
GL = [("day_type", ["Laborable", "Fin de semana"]),
      ("season", ["Invierno", "Primavera", "Verano", "Otoño"]),
      ("high_demand", ["Alta (>p90)", "Resto"])]
d_idx = diag.set_index(["country", "grupo", "nivel"])
lines = []
for gi, (g, levels) in enumerate(GL):
    for lv in levels:
        cells = []
        for c in C:
            r = d_idx.loc[(c, g, lv)]
            cells.append(f"{integer(r['n'])} & {integer(round(r['MAE_rf']))} & {pct0(r['Mejora %'], 0)}")
        lines.append(f"{lv.replace('>', '$>$ ')} & " + " & ".join(cells) + " \\\\")
    if gi < len(GL) - 1:
        lines.append("\\midrule")
write("tabla_diag.tex", "\n".join([
    "\\begin{tabular}{lrrrrrrrrr}", "\\toprule",
    " & \\multicolumn{3}{c}{DE} & \\multicolumn{3}{c}{ES} & \\multicolumn{3}{c}{FR} \\\\",
    "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
    "Subgrupo & $n$ & MAE RF & Mejora & $n$ & MAE RF & Mejora & $n$ & MAE RF & Mejora \\\\",
    "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))

lines = []
for c in C:
    for gi, (g, levels) in enumerate(GL):
        for lv in levels:
            r = d_idx.loc[(c, g, lv)]
            first = c if (g == "day_type" and lv == "Laborable") else ""
            lines.append(f"{first} & {lv.replace('>', '$>$ ')} & {integer(r['n'])} & {integer(round(r['MAE_rf']))} & "
                         f"{integer(round(r['MAE_ref']))} & {pct0(r['Mejora %'])} & {sint(r['bias_rf'])} \\\\")
    if c != "FR":
        lines.append("\\midrule")
write("tabla_diag_detalle.tex", "\n".join([
    "\\begin{tabular}{llrrrrr}", "\\toprule",
    "País & Subgrupo & $n$ & MAE RF (MW) & MAE estac. (MW) & Mejora & Sesgo RF (MW) \\\\",
    "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))

# ------------------------------------------------- T13: peores días (tres por país)
HOL = {("DE", "2019-10-03"): "Día de la Unidad Alemana", ("DE", "2019-05-01"): "Día del Trabajo",
       ("DE", "2019-04-19"): "Viernes Santo", ("ES", "2019-12-25"): "Navidad",
       ("ES", "2019-05-01"): "Día del Trabajo", ("ES", "2019-12-06"): "Día de la Constitución",
       ("FR", "2019-04-22"): "Lunes de Pascua", ("FR", "2019-05-01"): "Día del Trabajo",
       ("FR", "2019-11-11"): "Armisticio"}
lines = []
for c in C:
    w = worst[worst["country"] == c].head(3)
    for k, (_, r) in enumerate(w.iterrows()):
        ev = HOL.get((c, r["target_date"]), "--")
        y, mth, dd = r["target_date"].split("-")
        lines.append(f"{c if k == 0 else ''} & {dd}-{mth}-{y} & {ev} & {integer(round(r['MAE_rf']))} & "
                     f"{integer(round(r['MAE_ref']))} & {sint(r['bias_rf'])} \\\\")
    if c != "FR":
        lines.append("\\midrule")
write("tabla_peores.tex", "\n".join([
    "\\begin{tabular}{llp{4.2cm}rrr}", "\\toprule",
    "País & Fecha objetivo & Feriado nacional & MAE RF (MW) & MAE estac. (MW) & Sesgo RF (MW) \\\\",
    "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))

# ---------------------------------------------------------------- Figuras de la reserva
# F3: diagnóstico (error por hora y error diario)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
ax = axes[0]
for c in C:
    h = hourly[hourly["country"] == c]
    mc = mu[C.index(c)]
    ax.plot(h["target_hour"], h["ae_ref"] / mc, ls="--", color=CCOL[c], lw=1.2)
    ax.plot(h["target_hour"], h["ae_rf"] / mc, color=CCOL[c], lw=2, label=c)
ax.set_xlabel("Hora local del instante objetivo")
ax.set_ylabel("MAE / demanda media (NMAE)")
ax.set_xticks(range(0, 24, 4))
ax.grid(alpha=0.3)
ax.set_title("(a) Error por hora del día", fontsize=9)
ax.plot([], [], color="k", lw=2, label="RF (cíclico)")
ax.plot([], [], color="k", ls="--", lw=1.2, label="Estacional")
ax.legend(frameon=False, fontsize=7, ncol=2)
ax = axes[1]
pred["_ae"] = (pred[TARGET] - pred["pred_rf"]).abs()
pred["_date"] = pd.to_datetime(pred["target_local_timestamp"].str[:10])
dly = pred.groupby(["country", "_date"])["_ae"].mean().reset_index()
for c in C:
    s = dly[dly["country"] == c]
    ax.plot(s["_date"], s["_ae"] / mu[C.index(c)], color=CCOL[c], lw=0.9, label=c)
ax.set_ylabel("NMAE diario del RF (cíclico)")
ax.set_xlabel("Día objetivo (2019)")
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=2))
ax.grid(alpha=0.3)
ax.set_title("(b) Error diario a lo largo de 2019", fontsize=9)
fig.tight_layout()
fig.savefig(IMG / "fig_diagnostico.pdf")
plt.close(fig)

# F4: distribución bootstrap de la diferencia macro (RF cíclico - estacional)
obs, lo, hi, _, bs = boot_res["ref"]
fig, ax = plt.subplots(figsize=(5.6, 2.8))
ax.hist(bs[:, 0], bins=40, color="#8fbfae", edgecolor="white")
ax.axvline(obs[0], color="k", lw=1.5, label="Observado")
for q in (lo[0], hi[0]):
    ax.axvline(q, color="k", ls="--", lw=1)
ax.axvline(0, color="#D55E00", lw=1.5, label="Sin diferencia")
ax.set_xlabel("$\\Delta$ NMAE macro (RF cíclico $-$ estacional semanal)")
ax.set_ylabel("Réplicas (de 2000)")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(IMG / "fig_bootstrap.pdf")
plt.close(fig)

# ------------------------------------------------- T14: sensibilidad del bootstrap al largo del bloque (anexo; notebook 06)
bl = pd.read_csv(RESULTS / "bootstrap_block_length.csv")
lines = []
for L in sorted(bl["bloque_dias"].unique()):
    s = bl[bl["bloque_dias"] == L].set_index("comparador")
    cells = [f"[{pct0(100 * s.loc[k, 'mejora_inf'])}; {pct0(100 * s.loc[k, 'mejora_sup'])}]"
             for k in ["Estacional semanal", "Ridge", "Árbol podado", "RF dummies"]]
    lines.append(f"{int(L)} & {int(s['n_bloques'].iloc[0])} & " + " & ".join(cells) + " \\\\")
obs = bl[bl["bloque_dias"] == 7].set_index("comparador")["mejora"]
lines.append("\\midrule")
lines.append("Observado & -- & " + " & ".join(pct0(100 * obs[k]) for k in
             ["Estacional semanal", "Ridge", "Árbol podado", "RF dummies"]) + " \\\\")
write("tabla_boot_bloque.tex", "\n".join([
    "\\begin{tabular}{rrcccc}", "\\toprule",
    "Bloque (días) & Bloques & vs. estacional & vs. Ridge & vs. árbol podado & vs. RF (dummies) \\\\",
    "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))

# ================================================================ Cifras exportadas por el notebook 06
def fa(x):
    """Valor de hiperparámetro de poda o regularización."""
    if x == 0:
        return "0"
    if 0.1 <= x < 1000:
        return num(x, 2)
    if 1000 <= x < 1e5:
        return integer(round(x))
    m, e = f"{x:.1e}".split("e")
    return f"${m.replace('.', '{,}')}\\times10^{{{int(e)}}}$"


def two_columns(rows, header, ncol_fmt):
    """Tabla en dos bloques de columnas lado a lado (rows: lista de tuplas de celdas)."""
    half = (len(rows) + 1) // 2
    left, right = rows[:half], rows[half:]
    right = right + [("",) * len(rows[0])] * (len(left) - len(right))
    lines = [" & ".join(l) + " & " + " & ".join(r) + " \\\\" for l, r in zip(left, right)]
    return "\n".join([f"\\begin{{tabular}}{{{ncol_fmt}}}", "\\toprule", header, "\\midrule", *lines,
                      "\\bottomrule", "\\end{tabular}"])


# ------------------------------------------------- Ventanas (diseño)
ws = pd.read_csv(RESULTS / "window_sizes.csv").set_index("window")
span = {"W1": ("2015", "2016"), "W2": ("2015--2016", "2017"), "W3": ("2015--2017", "2018")}
lines = [f"{w} & {span[w][0]} & {span[w][1]} & {integer(ws.loc[w, 'n_train'])} & {integer(ws.loc[w, 'n_val'])} \\\\"
         for w in ["W1", "W2", "W3"]]
write("tabla_ventanas.tex", "\n".join([
    "\\begin{tabular}{lllrr}", "\\toprule",
    "Ventana & Entrenamiento (año de $t$) & Validación (año de $t$) & $n$ entrenamiento & $n$ validación \\\\",
    "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))

# ------------------------------------------------- Búsquedas (anexo)
rs = pd.read_csv(RESULTS / "ridge_search.csv")
rows = [(fa(r.alpha), num(r.mean_NMAE, 5), num(r.std_NMAE, 5)) for r in rs.itertuples()]
write("tabla_ridge.tex", two_columns(rows, "$\\alpha$ & Media & Desv. est. & $\\alpha$ & Media & Desv. est. \\\\",
                                     "rrr@{\\hspace{1cm}}rrr"))
ts = pd.read_csv(RESULTS / "tree_search.csv")
rows = [(fa(r.ccp_alpha), num(r.mean_NMAE, 5), num(r.std_NMAE, 5)) for r in ts.itertuples()]
write("tabla_arbol.tex", two_columns(rows, "\\texttt{ccp\\_alpha} & Media & Desv. est. & \\texttt{ccp\\_alpha} & Media & Desv. est. \\\\",
                                     "rrr@{\\hspace{1cm}}rrr"))
rf_s = pd.read_csv(RESULTS / "rf_search.csv")
lines = []
for r in rf_s.itertuples():
    md_ = "ninguna" if pd.isna(r.max_depth) else str(int(r.max_depth))
    lines.append(f"{md_} & {int(r.min_samples_leaf)} & {num(r.max_features, 1)} & {num(r.mean_nmae, 5)} & "
                 f"{num(r.std_nmae, 5)} & {num(r.min_nmae, 5)} & {num(r.max_nmae, 5)} \\\\")
write("tabla_rf_busqueda.tex", "\n".join([
    "\\begin{tabular}{rrrrrrr}", "\\toprule",
    "\\texttt{max\\_depth} & \\texttt{min\\_samples\\_leaf} & \\texttt{max\\_features} & Media & Desv. est. & Mínimo & Máximo \\\\",
    "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))

# ------------------------------------------------- Importancia y complejidad del modelo final
imp = pd.read_csv(RESULTS / "rf_importance_final.csv")
rows = [(f"\\texttt{{{r.predictor.replace('_', chr(92) + '_')}}}", num(r.importancia, 4)) for r in imp.itertuples()]
write("tabla_importancia.tex", two_columns(rows, "Predictor & Importancia & Predictor & Importancia \\\\",
                                           "lr@{\\hspace{1.2cm}}lr"))
cx = pd.read_csv(RESULTS / "final_model_complexity.csv")
lines = [f"{r.modelo} & {integer(r.arboles)} & {num(r.profundidad_media, 1)} & {integer(r.profundidad_max)} & "
         f"{integer(r.nodos_totales)} & {integer(r.hojas_totales)} \\\\" for r in cx.itertuples()]
write("tabla_complejidad.tex", "\n".join([
    "\\begin{tabular}{lrrrrr}", "\\toprule",
    "Modelo final & Árboles & Profundidad media & Profundidad máxima & Nodos & Hojas \\\\",
    "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))

# ------------------------------------------------- Correlación entre predictores (anexo)
cr = pd.read_csv(RESULTS / "predictor_correlation.csv")
VARS = ["demand_mw", "lag_48h", "lag_72h", "lag_168h", "rolling_mean_24h", "rolling_mean_168h", "target_demand_mw"]
VL = {v: f"\\texttt{{{v.replace('_', chr(92) + '_')}}}" for v in VARS}
cm = {(r.var_a, r.var_b): r.promedio for r in cr.itertuples()}
lines = []
for i, a in enumerate(VARS[1:], start=1):
    cells = [num(cm[(VARS[j], a)], 2) for j in range(i)]
    cells += [""] * (len(VARS) - 1 - i)
    lines.append(f"{VL[a]} & " + " & ".join(cells) + " \\\\")
write("tabla_correlacion.tex", "\n".join([
    "\\begin{tabular}{l" + "r" * (len(VARS) - 1) + "}", "\\toprule",
    " & " + " & ".join(VL[v] for v in VARS[:-1]) + " \\\\", "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))

# ------------------------------------------------- Ausentes: bloques y efecto de la exclusión
mb = pd.read_csv(RESULTS / "missing_blocks.csv", parse_dates=["inicio_utc"])
lines = []
for c in C:
    s = mb[mb["country"] == c]
    if s.empty:
        lines.append(f"{c} & sin ausentes & -- & 0 \\\\")
    for k, r in enumerate(s.itertuples()):
        t = r.inicio_utc
        lines.append(f"{c if k == 0 else ''} & {t:%d-%m-%Y} & {t:%H:%M} & {int(r.horas)} \\\\")
    if c != C[-1]:
        lines.append("\\midrule")
write("tabla_ausentes_bloques.tex", "\n".join([
    "\\begin{tabular}{llrr}", "\\toprule", "País & Fecha de inicio (UTC) & Hora & Horas consecutivas \\\\",
    "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))

me = pd.read_csv(RESULTS / "missing_effect.csv")
me_i = me.set_index(["periodo", "pais", "modelo"])
lines = []
for per in ["W1", "W2", "W3", "2019"]:
    for c in C + ["Macro"]:
        base_r = me_i.loc[(per, c, "Estacional semanal")]
        cells = []
        for mname in ["Estacional semanal", "Ridge", "RF cíclico"]:
            if (per, c, mname) in me_i.index:
                r = me_i.loc[(per, c, mname)]
                cells.append(f"{num(r['nmae_propio'])} & {num(r['nmae_comun'])}")
            else:
                cells.append("-- & --")
        lines.append(f"{per if c == 'DE' else ''} & {c} & {integer(base_r['n_propio'])} & "
                     f"{integer(base_r['n_comun'])} & " + " & ".join(cells) + " \\\\")
    if per != "2019":
        lines.append("\\midrule")
write("tabla_ausentes_efecto.tex", "\n".join([
    "\\begin{tabular}{llrrrrrrrr}", "\\toprule",
    " & & \\multicolumn{2}{c}{$n$} & \\multicolumn{2}{c}{Estacional} & \\multicolumn{2}{c}{Ridge} & \\multicolumn{2}{c}{RF cíclico} \\\\",
    "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}",
    "Periodo & País & Propio & Común & Propio & Común & Propio & Común & Propio & Común \\\\",
    "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))

# ------------------------------------------------- F5: descripción de la demanda (desarrollo)
dd = base[(base["period"] == "development")].dropna(subset=[TARGET])
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
for ax, col, ticks, labs, xl in [
        (axes[0], "target_hour", range(0, 24, 4), None, "Hora local del instante objetivo"),
        (axes[1], "target_day", range(7), list("LMXJVSD"), "Día de la semana"),
        (axes[2], "target_month", range(1, 13, 2), None, "Mes")]:
    for c in C:
        s = dd[dd["country"] == c]
        prof = s.groupby(col)[TARGET].mean() / s[TARGET].mean()
        ax.plot(prof.index, prof.values, marker="o", ms=2.5, lw=1.5, color=CCOL[c], label=c)
    ax.set_xticks(list(ticks))
    if labs:
        ax.set_xticklabels(labs)
    ax.set_xlabel(xl)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Demanda media / media del país")
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(IMG / "fig_descriptiva.pdf")
plt.close(fig)
print("figuras escritas en", IMG)
print("n_blocks", nb, "mu", mu.round(1))
