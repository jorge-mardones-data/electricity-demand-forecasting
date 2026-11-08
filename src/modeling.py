from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "demanda_base_analitica.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

COUNTRIES = ["DE", "ES", "FR"]
TARGET = "target_demand_mw"

CATEGORICAL_FEATURES = ["country", "target_hour", "target_day", "target_month"]
NUMERIC_FEATURES = [
    "demand_mw", "lag_48h", "lag_72h", "lag_168h",
    "rolling_mean_24h", "rolling_mean_168h",
]
CANDIDATE_PREDICTORS = CATEGORICAL_FEATURES[:1] + NUMERIC_FEATURES + CATEGORICAL_FEATURES[1:]

CALENDAR_PERIODS = {"target_hour": 24, "target_day": 7, "target_month": 12}
CYCLIC_FEATURES = [f"{c}_{f}" for c in CALENDAR_PERIODS for f in ("sin", "cos")]

# Configuraciones seleccionadas en el notebook 03 (results/selected_configurations.csv)
RIDGE_ALPHA = 316.22777
TREE_CCP_ALPHA = 256.86578
RF_PARAMS = dict(n_estimators=200, max_depth=None, min_samples_leaf=2,
                 max_features=0.7, random_state=42)

VALIDATION_WINDOWS = [
    ("W1", "2016-01-01", "2017-01-01"),
    ("W2", "2017-01-01", "2018-01-01"),
    ("W3", "2018-01-01", "2019-01-01"),
]


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["utc_timestamp", "target_timestamp"])
    return add_cyclic_features(df)


def add_cyclic_features(df):
    df = df.copy()
    for col, period in CALENDAR_PERIODS.items():
        angle = 2 * np.pi * df[col] / period
        df[f"{col}_sin"] = np.sin(angle)
        df[f"{col}_cos"] = np.cos(angle)
    return df


def make_partitions(df, train_years=None):
    required = CANDIDATE_PREDICTORS + [TARGET]
    partitions = {}
    for name, start_date, end_date in VALIDATION_WINDOWS:
        start = pd.Timestamp(start_date, tz="UTC")
        end = pd.Timestamp(end_date, tz="UTC")
        train_mask = (
            (df["period"] == "development")
            & (df["utc_timestamp"] < start)
            & (df["target_timestamp"] < start)
        )
        if train_years is not None:
            train_mask &= df["utc_timestamp"] >= start - pd.DateOffset(years=train_years)
        valid_mask = (
            (df["period"] == "development")
            & (df["utc_timestamp"] >= start)
            & (df["utc_timestamp"] < end)
        )
        train_df = df.loc[train_mask].dropna(subset=required).copy()
        valid_df = df.loc[valid_mask].dropna(subset=required).copy()
        assert train_df["target_timestamp"].max() < valid_df["utc_timestamp"].min()
        partitions[name] = {"train": train_df, "validation": valid_df}
    return partitions


def make_final_split(df):
    required = CANDIDATE_PREDICTORS + [TARGET]
    start = pd.Timestamp("2019-01-01", tz="UTC")
    train_df = df[
        (df["period"] == "development") & (df["target_timestamp"] < start)
    ].dropna(subset=required).copy()
    test_df = df[df["period"] == "holdout"].dropna(subset=required).copy()
    assert train_df["target_timestamp"].max() < test_df["utc_timestamp"].min()
    return train_df, test_df


def feature_sets(calendar="dummy"):
    if calendar == "dummy":
        return NUMERIC_FEATURES, CATEGORICAL_FEATURES
    if calendar == "cyclic":
        return NUMERIC_FEATURES + CYCLIC_FEATURES, ["country"]
    raise ValueError(calendar)


def make_ridge(calendar="dummy", alpha=RIDGE_ALPHA):
    num, cat = feature_sets(calendar)
    pre = ColumnTransformer([
        ("numeric", StandardScaler(), num),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    return Pipeline([("preprocessor", pre), ("model", Ridge(alpha=alpha))]), num + cat


def make_rf(calendar="dummy", n_jobs=-1, **overrides):
    num, cat = feature_sets(calendar)
    pre = ColumnTransformer([
        ("num", "passthrough", num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    params = {**RF_PARAMS, **overrides}
    rf = RandomForestRegressor(n_jobs=n_jobs, **params)
    return Pipeline([("preprocessor", pre), ("model", rf)]), num + cat


def make_tree(ccp_alpha=TREE_CCP_ALPHA):
    from sklearn.tree import DecisionTreeRegressor
    pre = ColumnTransformer([
        ("numeric", "passthrough", NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    tree = DecisionTreeRegressor(random_state=42, ccp_alpha=ccp_alpha)
    return Pipeline([("preprocessor", pre), ("model", tree)]), NUMERIC_FEATURES + CATEGORICAL_FEATURES


def evaluate_predictions(train_df, valid_df, predictions, model_name, window_name):
    predictions = pd.Series(np.asarray(predictions), index=valid_df.index)
    rows = []
    for country in COUNTRIES:
        tr = train_df[train_df["country"] == country]
        va = valid_df[valid_df["country"] == country]
        y_true = va[TARGET].to_numpy()
        y_pred = predictions.loc[va.index].to_numpy()
        mae = mean_absolute_error(y_true, y_pred)
        rows.append({
            "window": window_name, "model": model_name, "country": country,
            "n": len(va), "MAE": mae,
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "NMAE": mae / tr[TARGET].mean(),
        })
    out = pd.DataFrame(rows)
    macro = {"window": window_name, "model": model_name, "country": "Macro",
             "n": out["n"].sum(), "MAE": np.nan, "RMSE": np.nan,
             "NMAE": out["NMAE"].mean()}
    return pd.concat([out, pd.DataFrame([macro])], ignore_index=True)
