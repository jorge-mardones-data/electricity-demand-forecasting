"""Genera las figuras (.pdf/.png) y tablas (.tex) para la presentación en Beamer,
a partir de los mismos results/ y data/processed/ que usa make_report_assets.py.

No modifica datos, scripts ni resultados originales del proyecto; solo lee de
results/ y data/processed/ y escribe en reports/presentacion/.

Salidas:
    reports/presentacion/generated/*.tex            tablas listas para \\input en Beamer
    reports/presentacion/generated/respaldo/*.tex    tablas técnicas de respaldo (anexo)
    reports/presentacion/img/*.pdf y *.png           figuras (vectorial + alta resolución)

Uso (desde la raíz del repositorio):
    python src/make_presentation_assets.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "processed" / "demanda_base_analitica.csv"
OUT = ROOT / "reports" / "presentacion"
GEN = OUT / "generated"
BACKUP = GEN / "respaldo"
IMG = OUT / "img"
for d in (OUT, OUT / "content", GEN, BACKUP, IMG):
    d.mkdir(parents=True, exist_ok=True)

MODELS = ["Persistence", "Weekly Seasonal", "Ridge", "Decision Tree", "Random Forest"]
LABELS = {
    "Persistence": "Persistencia",
    "Weekly Seasonal": "Estacional semanal",
    "Ridge": "M1: Ridge",
    "Decision Tree": "M2: Árbol podado",
    "Random Forest": "M3: Random Forest (dummies)",
}
# Misma paleta Okabe-Ito (apta para daltonismo) usada en el informe, por consistencia.
COLORS = {
    "Persistence": "#999999",
    "Weekly Seasonal": "#555555",
    "Ridge": "#0072B2",
    "Decision Tree": "#E69F00",
    "Random Forest": "#009E73",
}
CYCLIC_COLOR = "#D55E00"  # M3*: Random Forest cíclico (versión final)
PREDICTORS = [
    "country", "demand_mw", "lag_48h", "lag_72h", "lag_168h",
    "rolling_mean_24h", "rolling_mean_168h",
    "target_hour", "target_day", "target_month",
]
TARGET = "target_demand_mw"
REFERENCE = "Weekly Seasonal"
C = ["DE", "ES", "FR"]
CCOL = {"DE": "#0072B2", "ES": "#D55E00", "FR": "#009E73"}
COUNTRY_NAME = {"DE": "Alemania (DE)", "ES": "España (ES)", "FR": "Francia (FR)"}

TEST_MODELS = ["Persistence", "Weekly Seasonal", "Ridge", "Decision Tree",
               "Random Forest (dummy)", "Random Forest"]
TLABEL = {**LABELS,
          "Random Forest (dummy)": "M3: RF (dummies)",
          "Random Forest": "M3$^\\ast$: RF (cíclico)"}
TLABEL_FIG = {**LABELS,
              "Random Forest (dummy)": "M3: RF (dummies)",
              "Random Forest": "M3*: RF (cíclico)"}  # texto plano, sin mathtext, para figuras
TCOLORS = {**COLORS, "Random Forest (dummy)": COLORS["Random Forest"], "Random Forest": CYCLIC_COLOR}

HOL = {("DE", "2019-10-03"): "Día de la Unidad Alemana", ("DE", "2019-05-01"): "Día del Trabajo",
       ("DE", "2019-04-19"): "Viernes Santo", ("ES", "2019-12-25"): "Navidad",
       ("ES", "2019-05-01"): "Día del Trabajo", ("ES", "2019-12-06"): "Día de la Constitución",
       ("FR", "2019-04-22"): "Lunes de Pascua", ("FR", "2019-05-01"): "Día del Trabajo",
       ("FR", "2019-11-11"): "Armisticio"}


# ---------------------------------------------------------------- utilidades (texto de tablas .tex)
def num(x, d=4):
    """Número con coma decimal, para celdas de tabla LaTeX."""
    return f"{x:.{d}f}".replace(".", "{,}")


def integer(n):
    """Entero con espacio fino como separador de miles, para celdas LaTeX."""
    return f"{int(n):,}".replace(",", "\\,")


def pct(x, d=1):
    sign = "+" if x > 0 else ""
    return sign + f"{x:.{d}f}".replace(".", "{,}") + "\\%"


def pct0(x, d=1):
    """Porcentaje sin signo forzado; negativo con signo menos tipográfico."""
    return ("$-$" if x < 0 else "") + f"{abs(x):.{d}f}".replace(".", "{,}") + "\\%"


def snum(x, d=4):
    return ("$-$" if x < 0 else "") + f"{abs(x):.{d}f}".replace(".", "{,}")


def sint(x):
    v = round(abs(x))
    return ("$-$" if x < 0 and v != 0 else "") + integer(v)


def fa(x):
    """Valor de hiperparámetro de poda o regularización (notación científica si corresponde)."""
    if x == 0:
        return "0"
    if 0.1 <= x < 1000:
        return num(x, 2)
    if 1000 <= x < 1e5:
        return integer(round(x))
    m, e = f"{x:.1e}".split("e")
    return f"${m.replace('.', '{,}')}\\times10^{{{int(e)}}}$"


def two_columns(rows, header, ncol_fmt):
    half = (len(rows) + 1) // 2
    left, right = rows[:half], rows[half:]
    right = right + [("",) * len(rows[0])] * (len(left) - len(right))
    lines = [" & ".join(l) + " & " + " & ".join(r) + " \\\\" for l, r in zip(left, right)]
    return "\n".join([f"\\begin{{tabular}}{{{ncol_fmt}}}", "\\toprule", header, "\\midrule", *lines,
                      "\\bottomrule", "\\end{tabular}"])


def write_tex(folder, name, text):
    (folder / name).write_text(text, encoding="utf-8")
    print("tabla  ", folder / name)


def fint(n):
    """Entero con espacio fino para texto embebido en figuras (no LaTeX)."""
    return f"{int(round(n)):,}".replace(",", "\u2009")


def save_fig(fig, name):
    fig.savefig(IMG / f"{name}.pdf")
    fig.savefig(IMG / f"{name}.png", dpi=300)
    plt.close(fig)
    print("figura ", name)


# ---------------------------------------------------------------- estilo: sobrio, grande, 16:9
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

# ================================================================ datos comunes
glob_tbl = pd.read_csv(RESULTS / "global_comparison.csv").set_index("model")
sens = pd.read_csv(RESULTS / "sensitivity_results.csv", encoding="utf-8")
cyc_w = (sens[(sens["analysis"] == "A_calendar") & (sens["model"] == "Random Forest")
              & (sens["country"] == "Macro") & (sens["scenario"] != "Dummy")]
         .sort_values("window")["NMAE"].to_numpy())
assert len(cyc_w) == 3

base = pd.read_csv(DATA, parse_dates=["utc_timestamp", "target_timestamp"])
base["complete"] = base[PREDICTORS + [TARGET]].notna().all(axis=1)

test = pd.read_csv(RESULTS / "test_results.csv")
tm = test.set_index(["model", "country"])
pred = pd.read_csv(RESULTS / "test_predictions.csv", parse_dates=["utc_timestamp"])
hourly = pd.read_csv(RESULTS / "test_error_by_hour.csv")
diag = pd.read_csv(RESULTS / "test_diagnostics.csv")
worst = pd.read_csv(RESULTS / "test_worst_days.csv")
window_sizes = pd.read_csv(RESULTS / "window_sizes.csv").set_index("window")

ft = base[(base["period"] == "development") & base["complete"]
          & (base["target_timestamp"] < pd.Timestamp("2019-01-01", tz="UTC"))]
mu = ft.groupby("country")[TARGET].mean()[C].to_numpy()

# ================================================================ FIGURA 1: flujo de construcción de la muestra
periodo = base["period"].fillna("excluido")
n_original_hours = len(base) // 3
dev_n = int((periodo == "development").sum())
dev_c = int(base.loc[periodo == "development", "complete"].sum())
hold_n = int((periodo == "holdout").sum())
hold_c = int(base.loc[periodo == "holdout", "complete"].sum())
excl_n = int((periodo == "excluido").sum())


def box(ax, x, y, w, h, text, fc="#f2f2f2", ec="#333333", fontsize=15, fontweight="normal"):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                          linewidth=1.5, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, fontweight=fontweight)


def arrow(ax, xy1, xy2, color="#666666"):
    ax.annotate("", xy=xy2, xytext=xy1,
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color=color, shrinkA=0, shrinkB=0))


fig, ax = plt.subplots(figsize=(9.6, 5.0))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")

box(ax, 1.7, 4.5, 6.6, 1.1, f"Datos originales\n{fint(n_original_hours)} horas $\\times$ 3 países",
    fc="#dbe7f3", ec="#0072B2", fontweight="bold")
arrow(ax, (5.0, 4.5), (5.0, 3.85))
box(ax, 3.4, 3.05, 3.2, 0.7, f"Excluidas: {fint(excl_n)} filas\n(31-dic-2018)", fc="#f2f2f2", ec="#999999", fontsize=12)
arrow(ax, (3.7, 3.05), (2.0, 2.55))
arrow(ax, (6.3, 3.05), (8.0, 2.55))

box(ax, 0.4, 1.35, 4.2, 1.2,
    f"Periodo de desarrollo\n2015\u20132018\n{fint(dev_n)} filas · {fint(dev_c)} completos",
    fc="#dbe7f3", ec="#0072B2")
box(ax, 5.4, 1.35, 4.2, 1.2,
    f"Reserva\n2019\n{fint(hold_n)} filas · {fint(hold_c)} completos",
    fc="#e8e8e8", ec="#999999")
ax.text(5.0, 0.55, "Entrenamiento, validación y selección                    Evaluación final (una sola vez)",
        ha="center", va="center", fontsize=12, color="#555555")
fig.tight_layout()
save_fig(fig, "flujo_muestra")

# ================================================================ FIGURA 2: validación temporal
COL_TRAIN, COL_VAL, COL_RES = "#0072B2", "#E69F00", "#999999"
specs2 = [
    ("W1", 1, 1),
    ("W2", 2, 1),
    ("W3", 3, 1),
    ("Evaluación\nfinal", 4, 1),
]
fig, ax = plt.subplots(figsize=(9.6, 4.6))
bar_h = 0.58
n_rows = len(specs2)
for i, (label, train_w, other_w) in enumerate(specs2):
    y = n_rows - i
    other_color = COL_RES if label.startswith("Evaluación") else COL_VAL
    ax.broken_barh([(0, train_w)], (y - bar_h / 2, bar_h), facecolors=COL_TRAIN, edgecolor="white")
    ax.broken_barh([(train_w, other_w)], (y - bar_h / 2, bar_h), facecolors=other_color, edgecolor="white")
    ax.text(-0.15, y, label, ha="right", va="center", fontsize=15)
ax.set_xlim(0, 5)
ax.set_ylim(0.3, n_rows + 0.8)
ax.set_xticks(range(0, 6))
ax.set_xticklabels(["2015", "2016", "2017", "2018", "2019", ""])
ax.set_yticks([])
for spine in ("left", "top", "right"):
    ax.spines[spine].set_visible(False)
handles = [mpatches.Patch(color=COL_TRAIN, label="Entrenamiento"),
           mpatches.Patch(color=COL_VAL, label="Validación"),
           mpatches.Patch(color=COL_RES, label="Reserva 2019")]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3, frameon=False)
fig.tight_layout()
save_fig(fig, "validacion_temporal")

# ================================================================ FIGURA 3: diagnóstico 2019 (paneles separados)
fig, ax = plt.subplots(figsize=(8.4, 5.2))
for c in C:
    h = hourly[hourly["country"] == c]
    mc = mu[C.index(c)]
    ax.plot(h["target_hour"], h["ae_ref"] / mc, ls="--", color=CCOL[c], lw=1.6)
    ax.plot(h["target_hour"], h["ae_rf"] / mc, color=CCOL[c], lw=2.6, label=c)
ax.set_xlabel("Hora local del instante objetivo")
ax.set_ylabel("NMAE")
ax.set_xticks(range(0, 24, 4))
ax.grid(alpha=0.25)
ax.plot([], [], color="black", lw=2.6, label="RF cíclico")
ax.plot([], [], color="black", ls="--", lw=1.6, label="Estacional")
ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.22))
fig.tight_layout()
save_fig(fig, "error_por_hora")

pred["_ae"] = (pred[TARGET] - pred["pred_rf"]).abs()
pred["_date"] = pd.to_datetime(pred["target_local_timestamp"].str[:10])
dly = pred.groupby(["country", "_date"])["_ae"].mean().reset_index()
fig, ax = plt.subplots(figsize=(8.4, 5.2))
for c in C:
    s = dly[dly["country"] == c]
    ax.plot(s["_date"], s["_ae"] / mu[C.index(c)], color=CCOL[c], lw=1.2, label=c)
ax.set_ylabel("NMAE diario (RF cíclico)")
ax.set_xlabel("Día objetivo (2019)")
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=2))
ax.grid(alpha=0.25)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
fig.tight_layout()
save_fig(fig, "error_diario")

# ================================================================ FIGURA 4: perfiles de demanda
dd = base[base["period"] == "development"].dropna(subset=[TARGET])
specs4 = [
    ("perfil_hora", "target_hour", range(0, 24, 4), None, "Hora local del instante objetivo"),
    ("perfil_dia", "target_day", range(7), list("LMXJVSD"), "Día de la semana"),
    ("perfil_mes", "target_month", range(1, 13, 2), None, "Mes"),
]


def profile_plot(ax, col, ticks, labs, xl):
    for c in C:
        s = dd[dd["country"] == c]
        prof = s.groupby(col)[TARGET].mean() / s[TARGET].mean()
        ax.plot(prof.index, prof.values, marker="o", ms=4, lw=2.2, color=CCOL[c], label=c)
    ax.set_xticks(list(ticks))
    if labs:
        ax.set_xticklabels(labs)
    ax.set_xlabel(xl)
    ax.grid(alpha=0.25)


for name, col, ticks, labs, xl in specs4:
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    profile_plot(ax, col, ticks, labs, xl)
    ax.set_ylabel("Demanda media / media del país")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3)
    fig.tight_layout()
    save_fig(fig, name)

fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.6))
for ax, (name, col, ticks, labs, xl) in zip(axes, specs4):
    profile_plot(ax, col, ticks, labs, xl)
axes[0].set_ylabel("Demanda media / media del país")
axes[0].legend(frameon=False, fontsize=13)
fig.tight_layout()
save_fig(fig, "perfil_demanda")

# ================================================================ bootstrap por bloques (compartido: figura 5 y tabla 9)
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

# verificación contra lo exportado por el notebook 05 (misma comprobación que make_report_assets.py)
chk = pd.read_csv(RESULTS / "test_bootstrap_ci.csv", index_col=0)
o, lo, hi, _, _ = boot_res["ref"]
assert np.allclose(np.r_[o[:5]], chk["observado"].to_numpy(), atol=1e-9), "bootstrap: observado difiere del notebook"
assert np.allclose(np.r_[lo[:5]], chk["IC95 inf"].to_numpy(), atol=1e-6), "bootstrap: IC inf difiere del notebook"
assert np.allclose(np.r_[hi[:5]], chk["IC95 sup"].to_numpy(), atol=1e-6), "bootstrap: IC sup difiere del notebook"

# FIGURA 5: distribución bootstrap (respaldo)
obs, lo, hi, _, bs = boot_res["ref"]
fig, ax = plt.subplots(figsize=(8.0, 4.8))
ax.hist(bs[:, 0], bins=40, color="#8fbfae", edgecolor="white")
ax.axvline(obs[0], color="black", lw=2, label="Observado")
for q in (lo[0], hi[0]):
    ax.axvline(q, color="black", ls="--", lw=1.3)
ax.axvline(0, color=CYCLIC_COLOR, lw=2, label="Sin diferencia")
ax.set_xlabel("$\\Delta$ NMAE macro (RF cíclico $-$ estacional semanal)")
ax.set_ylabel("Réplicas (de 2\u2009000)")
ax.legend(frameon=False)
fig.tight_layout()
save_fig(fig, "bootstrap_respaldo")

# TABLA 9 simplificada: intervalos de confianza (solo nivel macro)
rows9 = [("ref", "vs. Estacional semanal"), ("ridge", "vs. M1: Ridge"),
         ("tree", "vs. M2: Árbol podado"), ("dummy", "vs. M3: RF (dummies)")]
lines = []
for key, lab in rows9:
    obs_k, lo_k, hi_k, _, _ = boot_res[key]
    lines.append(f"{lab} & {pct0(100 * obs_k[4])} & [{pct0(100 * lo_k[4])}; {pct0(100 * hi_k[4])}] \\\\")
write_tex(GEN, "intervalos_confianza.tex", "\n".join([
    "\\begin{tabular}{lcc}", "\\toprule",
    "Comparación (M3$^\\ast$ vs.) & Mejora relativa & IC 95\\,\\% \\\\", "\\midrule",
    *lines, "\\bottomrule", "\\end{tabular}"]))

# ================================================================ TABLA 5: comparación de modelos (desarrollo)
models_plot = [LABELS[m] for m in MODELS] + ["M3*: RF (cíclico)"]
means = [glob_tbl.loc[m, "Mean NMAE"] for m in MODELS] + [cyc_w.mean()]
colors_plot = [COLORS[m] for m in MODELS] + [CYCLIC_COLOR]
order = np.argsort(means)
models_s = [models_plot[i] for i in order]
means_s = [means[i] for i in order]
colors_s = [colors_plot[i] for i in order]
fig, ax = plt.subplots(figsize=(8.6, 4.8))
y = np.arange(len(models_s))
ax.barh(y, means_s, color=colors_s, height=0.6)
ax.set_yticks(y)
ax.set_yticklabels(models_s)
ax.invert_yaxis()
ax.set_xlabel("NMAE macro (periodo de desarrollo, 2015\u20132018)")
for yi, v in zip(y, means_s):
    ax.text(v + 0.0008, yi, f"{v:.4f}".replace(".", ","), va="center", fontsize=13)
ax.axvline(glob_tbl.loc[REFERENCE, "Mean NMAE"], color="#555555", ls="--", lw=1.3)
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
save_fig(fig, "comparacion_modelos_desarrollo")

# ================================================================ TABLAS 7 y 8: comparación y detalle por país (2019)
means19 = [tm.loc[(m, "Macro"), "NMAE"] for m in TEST_MODELS]
labels19 = [TLABEL_FIG[m] for m in TEST_MODELS]
colors19 = [TCOLORS[m] for m in TEST_MODELS]
order19 = np.argsort(means19)
labels19_s = [labels19[i] for i in order19]
means19_s = [means19[i] for i in order19]
colors19_s = [colors19[i] for i in order19]
fig, ax = plt.subplots(figsize=(8.6, 4.8))
y = np.arange(len(labels19_s))
ax.barh(y, means19_s, color=colors19_s, height=0.6)
ax.set_yticks(y)
ax.set_yticklabels(labels19_s)
ax.invert_yaxis()
ax.set_xlabel("NMAE macro (reserva 2019)")
for yi, v in zip(y, means19_s):
    ax.text(v + 0.0008, yi, f"{v:.4f}".replace(".", ","), va="center", fontsize=13)
ax.axvline(tm.loc[(REFERENCE, "Macro"), "NMAE"], color="#555555", ls="--", lw=1.3)
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
save_fig(fig, "comparacion_modelos_2019")

lines = []
for c in C:
    ref_r, rf_r = tm.loc[(REFERENCE, c)], tm.loc[("Random Forest", c)]
    mejora = 100 * (1 - rf_r["NMAE"] / ref_r["NMAE"])
    lines.append(f"{COUNTRY_NAME[c]} & {num(ref_r['NMAE'])} & {num(rf_r['NMAE'])} & {pct0(mejora)} \\\\")
ref_m, rf_m = tm.loc[(REFERENCE, "Macro")], tm.loc[("Random Forest", "Macro")]
mejora_m = 100 * (1 - rf_m["NMAE"] / ref_m["NMAE"])
lines.append("\\midrule")
lines.append(f"\\textbf{{Macro}} & {num(ref_m['NMAE'])} & {num(rf_m['NMAE'])} & \\textbf{{{pct0(mejora_m)}}} \\\\")
write_tex(GEN, "resultados_paises.tex", "\n".join([
    "\\begin{tabular}{lccc}", "\\toprule",
    "País & NMAE estacional & NMAE RF cíclico & Mejora \\\\", "\\midrule",
    *lines, "\\bottomrule", "\\end{tabular}"]))

# ================================================================ TABLA 6: sensibilidades S1 y S2 (gráficos separados)
sm = sens[sens["country"] == "Macro"]
base_a = {m: sm[(sm["analysis"] == "A_calendar") & (sm["model"] == m) & (sm["scenario"] == "Dummy")]
          .sort_values("window")["NMAE"].to_numpy() for m in ["Random Forest", "Ridge"]}
alt_a = {m: sm[(sm["analysis"] == "A_calendar") & (sm["model"] == m) & (sm["scenario"] != "Dummy")]
         .sort_values("window")["NMAE"].to_numpy() for m in ["Random Forest", "Ridge"]}
alt_b = {m: sm[(sm["analysis"] == "B_history") & (sm["model"] == m)]
         .sort_values("window")["NMAE"].to_numpy() for m in ["Random Forest", "Ridge"]}


def sensitivity_chart(name, base_vals, alt_vals, base_label, alt_label):
    labels = ["Ridge", "Random Forest"]
    x = np.arange(2)
    w = 0.32
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.bar(x - w / 2, base_vals, w, color="#999999", label=base_label)
    ax.bar(x + w / 2, alt_vals, w, color="#0072B2", label=alt_label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("NMAE macro promedio")
    for xi, b, a in zip(x, base_vals, alt_vals):
        ax.text(xi - w / 2, b + 0.0008, f"{b:.4f}".replace(".", ","), ha="center", fontsize=12)
        ax.text(xi + w / 2, a + 0.0008, f"{a:.4f}".replace(".", ","), ha="center", fontsize=12)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_fig(fig, name)


sensitivity_chart("sensibilidad_calendario",
                   [base_a["Ridge"].mean(), base_a["Random Forest"].mean()],
                   [alt_a["Ridge"].mean(), alt_a["Random Forest"].mean()],
                   "Indicadores (base)", "Seno\u2013coseno (cíclico)")
sensitivity_chart("sensibilidad_historia",
                   [base_a["Ridge"][1:].mean(), base_a["Random Forest"][1:].mean()],
                   [alt_b["Ridge"][1:].mean(), alt_b["Random Forest"][1:].mean()],
                   "Ventana creciente (base)", "Solo último año")

# ================================================================ TABLAS 10 y 18: subgrupos y casos extremos
d_idx = diag.set_index(["country", "grupo", "nivel"])
GL = [("day_type", ["Laborable", "Fin de semana"]),
      ("season", ["Invierno", "Primavera", "Verano", "Otoño"]),
      ("high_demand", ["Alta (>p90)", "Resto"])]
sub_labels, sub_vals, sub_colors = [], [], []
group_colors = {"day_type": "#0072B2", "season": "#009E73", "high_demand": "#E69F00"}
for g, levels in GL:
    for lv in levels:
        n_sum = sum(d_idx.loc[(c, g, lv), "n"] for c in C)
        mae_rf_sum = sum(d_idx.loc[(c, g, lv), "n"] * d_idx.loc[(c, g, lv), "MAE_rf"] for c in C)
        mae_ref_sum = sum(d_idx.loc[(c, g, lv), "n"] * d_idx.loc[(c, g, lv), "MAE_ref"] for c in C)
        mejora = 100 * (1 - (mae_rf_sum / n_sum) / (mae_ref_sum / n_sum))
        sub_labels.append(lv)
        sub_vals.append(mejora)
        sub_colors.append(group_colors[g])

fig, ax = plt.subplots(figsize=(8.6, 5.6))
y = np.arange(len(sub_labels))
ax.barh(y, sub_vals, color=sub_colors, height=0.6)
ax.set_yticks(y)
ax.set_yticklabels(sub_labels)
ax.invert_yaxis()
ax.set_xlabel("Reducción del MAE frente a la referencia semanal (%)")
for yi, v in zip(y, sub_vals):
    ax.text(v + 0.6, yi, f"{v:.0f}%", va="center", fontsize=12)
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
save_fig(fig, "errores_subgrupo")

lines = []
for c in C:
    r = worst[worst["country"] == c].iloc[0]
    ev = HOL.get((c, r["target_date"]), "--")
    y_, mth, dd_ = r["target_date"].split("-")
    lines.append(f"{c} & {dd_}-{mth}-{y_} & {ev} & {integer(round(r['MAE_rf']))} & "
                 f"{integer(round(r['MAE_ref']))} & {sint(r['bias_rf'])} \\\\")
write_tex(GEN, "casos_extremos.tex", "\n".join([
    "\\begin{tabular}{llp{4.0cm}rrr}", "\\toprule",
    "País & Fecha & Feriado & MAE RF & MAE estac. & Sesgo RF \\\\",
    "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))

# ================================================================ TABLA 2: variables admisibles (simplificada)
write_tex(GEN, "tabla_variables.tex", "\n".join([
    "\\begin{tabular}{@{}p{4.4cm}p{6.4cm}p{2.0cm}@{}}", "\\toprule",
    "Variable & Qué mide & Decisión \\\\", "\\midrule",
    "\\texttt{country} & País (DE, ES, FR) & incluida \\\\",
    "\\texttt{demand\\_mw} & Demanda en la hora de emisión $t$ & incluida \\\\",
    "\\texttt{lag\\_48h}, \\texttt{lag\\_72h}, \\texttt{lag\\_168h} & Demanda en $t-24$, $t-48$, $t-144$ & incluidas \\\\",
    "\\texttt{rolling\\_mean\\_24h}, \\texttt{rolling\\_mean\\_168h} & Media móvil de 24 y 168 h hasta $t$ & incluidas \\\\",
    "\\texttt{target\\_hour}, \\texttt{target\\_day}, \\texttt{target\\_month} & Calendario local del objetivo & incluidas \\\\",
    "\\midrule",
    "\\texttt{target\\_demand\\_mw} & Demanda en $t+24$ & respuesta \\\\",
    "Identificadores y marcadores de auditoría & Claves temporales y variables derivadas & excluidas \\\\",
    "\\bottomrule", "\\end{tabular}"]))

# ================================================================ TABLA 3: resumen de ventanas de validación
span = {"W1": ("2015", "2016"), "W2": ("2015\u20132016", "2017"), "W3": ("2015\u20132017", "2018")}
lines = [f"{w} & {span[w][0]} & {span[w][1]} & {integer(window_sizes.loc[w, 'n_train'])} & "
         f"{integer(window_sizes.loc[w, 'n_val'])} \\\\" for w in ["W1", "W2", "W3"]]
write_tex(GEN, "tabla_ventanas.tex", "\n".join([
    "\\begin{tabular}{lllrr}", "\\toprule",
    "Ventana & Entrenamiento & Validación & $n$ entren. & $n$ valid. \\\\",
    "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))

# ================================================================ TABLAS DE RESPALDO (anexo técnico, sin adaptar en detalle)
val = pd.read_csv(RESULTS / "validation_results.csv")

# T2 informe: descripción de la respuesta
dev = base[base["period"] == "development"]
lines = []
for c in C:
    ycol = dev.loc[dev["country"] == c, TARGET].dropna()
    lines.append(f"{c} & {integer(len(ycol))} & {integer(round(ycol.mean()))} & "
                 f"{integer(round(ycol.std()))} & {integer(ycol.min())} & "
                 f"{integer(ycol.quantile(.5))} & {integer(ycol.max())} \\\\")
write_tex(BACKUP, "tabla_respuesta.tex", "\n".join([
    "\\begin{tabular}{lrrrrrr}", "\\toprule",
    "País & $n$ & Media & Desv. est. & Mínimo & Mediana & Máximo \\\\",
    "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))

# T6 informe: detalle completo por modelo, ventana y país
lines = []
for m in MODELS:
    for wd in ["W1", "W2", "W3"]:
        cells = []
        for ctry in C:
            r = val[(val["model"] == m) & (val["window"] == wd) & (val["country"] == ctry)].iloc[0]
            cells.append(f"{integer(round(r['MAE']))} & {integer(round(r['RMSE']))} & {num(r['NMAE'], 4)}")
        macro = glob_tbl.loc[m, wd]
        name = LABELS[m] if wd == "W1" else ""
        lines.append(f"{name} & {wd} & " + " & ".join(cells) + f" & {num(macro, 4)} \\\\")
    if m != MODELS[-1]:
        lines.append("\\midrule")
write_tex(BACKUP, "tabla_detalle.tex", "\n".join([
    "\\begin{tabular}{llrrrrrrrrrr}", "\\toprule",
    " & & \\multicolumn{3}{c}{DE} & \\multicolumn{3}{c}{ES} & \\multicolumn{3}{c}{FR} & Macro \\\\",
    "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-11}",
    "Procedimiento & Ventana & MAE & RMSE & NMAE & MAE & RMSE & NMAE & MAE & RMSE & NMAE & NMAE \\\\",
    "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))

# T11 informe: detalle de la reserva por procedimiento y país
lines = []
for m in TEST_MODELS:
    cells = []
    for c in C:
        r = tm.loc[(m, c)]
        cells.append(f"{integer(round(r['MAE']))} & {integer(round(r['RMSE']))} & {num(r['NMAE'])}")
    lines.append(f"{TLABEL[m]} & " + " & ".join(cells) + f" & {num(tm.loc[(m, 'Macro'), 'NMAE'])} \\\\")
    if m != TEST_MODELS[-1]:
        lines.append("\\midrule")
write_tex(BACKUP, "tabla_test_detalle.tex", "\n".join([
    "\\begin{tabular}{lrrrrrrrrrr}", "\\toprule",
    " & \\multicolumn{3}{c}{DE} & \\multicolumn{3}{c}{ES} & \\multicolumn{3}{c}{FR} & Macro \\\\",
    "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
    "Procedimiento & MAE & RMSE & NMAE & MAE & RMSE & NMAE & MAE & RMSE & NMAE & NMAE \\\\",
    "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))

# Búsqueda de Ridge
rs = pd.read_csv(RESULTS / "ridge_search.csv")
rows = [(fa(r.alpha), num(r.mean_NMAE, 5), num(r.std_NMAE, 5)) for r in rs.itertuples()]
write_tex(BACKUP, "tabla_ridge.tex", two_columns(
    rows, "$\\alpha$ & Media & Desv. est. & $\\alpha$ & Media & Desv. est. \\\\", "rrr@{\\hspace{1cm}}rrr"))

# Búsqueda de Random Forest
rf_s = pd.read_csv(RESULTS / "rf_search.csv")
lines = []
for r in rf_s.itertuples():
    md_ = "ninguna" if pd.isna(r.max_depth) else str(int(r.max_depth))
    lines.append(f"{md_} & {int(r.min_samples_leaf)} & {num(r.max_features, 1)} & {num(r.mean_nmae, 5)} & "
                 f"{num(r.std_nmae, 5)} & {num(r.min_nmae, 5)} & {num(r.max_nmae, 5)} \\\\")
write_tex(BACKUP, "tabla_rf_busqueda.tex", "\n".join([
    "\\begin{tabular}{rrrrrrr}", "\\toprule",
    "\\texttt{max\\_depth} & \\texttt{min\\_samples\\_leaf} & \\texttt{max\\_features} & Media & Desv. est. & Mínimo & Máximo \\\\",
    "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))

# Importancia y complejidad del modelo final
imp = pd.read_csv(RESULTS / "rf_importance_final.csv")
rows = [(f"\\texttt{{{r.predictor.replace('_', chr(92) + '_')}}}", num(r.importancia, 4)) for r in imp.itertuples()]
write_tex(BACKUP, "tabla_importancia.tex", two_columns(
    rows, "Predictor & Importancia & Predictor & Importancia \\\\", "lr@{\\hspace{1.2cm}}lr"))

cx = pd.read_csv(RESULTS / "final_model_complexity.csv")
lines = [f"{r.modelo} & {integer(r.arboles)} & {num(r.profundidad_media, 1)} & {integer(r.profundidad_max)} & "
         f"{integer(r.nodos_totales)} & {integer(r.hojas_totales)} \\\\" for r in cx.itertuples()]
write_tex(BACKUP, "tabla_complejidad.tex", "\n".join([
    "\\begin{tabular}{lrrrrr}", "\\toprule",
    "Modelo final & Árboles & Profundidad media & Profundidad máxima & Nodos & Hojas \\\\",
    "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))

ftimes = pd.read_csv(RESULTS / "final_fit_times.csv", index_col=0).iloc[:, 0]
tn = {"Ridge": "M1: Ridge", "Decision Tree": "M2: Árbol podado",
      "Random Forest (dummy)": "M3: RF (dummies)", "Random Forest": "M3$^\\ast$: RF (cíclico)"}
lines = [f"{tn[k]} & {num(ftimes[k], 1)} \\\\" for k in tn]
write_tex(BACKUP, "tabla_tiempos.tex", "\n".join([
    "\\begin{tabular}{lr}", "\\toprule", "Procedimiento & Ajuste final (s) \\\\", "\\midrule",
    *lines, "\\bottomrule", "\\end{tabular}"]))

# Correlación entre predictores
cr = pd.read_csv(RESULTS / "predictor_correlation.csv")
VARS = ["demand_mw", "lag_48h", "lag_72h", "lag_168h", "rolling_mean_24h", "rolling_mean_168h", "target_demand_mw"]
VL = {v: f"\\texttt{{{v.replace('_', chr(92) + '_')}}}" for v in VARS}
cm = {(r.var_a, r.var_b): r.promedio for r in cr.itertuples()}
lines = []
for i, a in enumerate(VARS[1:], start=1):
    cells = [num(cm[(VARS[j], a)], 2) for j in range(i)]
    cells += [""] * (len(VARS) - 1 - i)
    lines.append(f"{VL[a]} & " + " & ".join(cells) + " \\\\")
write_tex(BACKUP, "tabla_correlacion.tex", "\n".join([
    "\\begin{tabular}{l" + "r" * (len(VARS) - 1) + "}", "\\toprule",
    " & " + " & ".join(VL[v] for v in VARS[:-1]) + " \\\\", "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))

# Ausentes: bloques y efecto de la exclusión
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
write_tex(BACKUP, "tabla_ausentes_bloques.tex", "\n".join([
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
write_tex(BACKUP, "tabla_ausentes_efecto.tex", "\n".join([
    "\\begin{tabular}{llrrrrrrrr}", "\\toprule",
    " & & \\multicolumn{2}{c}{$n$} & \\multicolumn{2}{c}{Estacional} & \\multicolumn{2}{c}{Ridge} & \\multicolumn{2}{c}{RF cíclico} \\\\",
    "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}",
    "Periodo & País & Propio & Común & Propio & Común & Propio & Común & Propio & Común \\\\",
    "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))

print("listo: recursos de presentación en", OUT)
