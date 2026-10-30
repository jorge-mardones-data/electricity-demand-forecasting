from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "demanda_horaria.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "demanda_modelado.csv"

DEMAND_COLUMNS = {
    "DE_load_actual_entsoe_transparency": "DE",
    "ES_load_actual_entsoe_transparency": "ES",
    "FR_load_actual_entsoe_transparency": "FR",
}

#SE ESTABLECE 2015-2018 PARA DESARROLLO Y 2019 PARA EVALUACION

def main():
    # 1. Leer la base original
    df = pd.read_csv(INPUT_PATH)

    df["utc_timestamp"] = pd.to_datetime(
        df["utc_timestamp"], utc=True, errors="raise"
    )

    # 2. Transformar de formato ancho a formato largo
    long_df = df.melt(
        id_vars=["utc_timestamp"],
        value_vars=list(DEMAND_COLUMNS.keys()),
        var_name="source_column",
        value_name="demand_mw",
    )

    long_df["country"] = long_df["source_column"].map(DEMAND_COLUMNS)
    long_df = long_df.drop(columns="source_column")

    long_df = long_df[
        ["utc_timestamp", "country", "demand_mw"]
    ].sort_values(["country", "utc_timestamp"]).reset_index(drop=True)

    # 3. Validar la unidad de análisis
    expected_rows = len(df) * len(DEMAND_COLUMNS)

    assert len(long_df) == expected_rows
    assert not long_df.duplicated(
        ["country", "utc_timestamp"]
    ).any()

    # 4. Construir el instante objetivo: 24 horas después del origen
    long_df["target_timestamp"] = (
        long_df["utc_timestamp"] + pd.Timedelta(hours=24)
    )

    # 5. Vincular la demanda observada en el instante objetivo
    target_df = long_df[
        ["utc_timestamp", "country", "demand_mw"]
    ].rename(
        columns={
            "utc_timestamp": "target_timestamp",
            "demand_mw": "target_demand_mw",
        }
    )

    model_df = long_df.merge(
        target_df,
        on=["country", "target_timestamp"],
        how="left",
        validate="many_to_one",
    )

    # 6. Definir los períodos según el año del origen
    origin_year = model_df["utc_timestamp"].dt.year

    is_development = origin_year.between(2015, 2018)

    # Excluir los últimos 24 orígenes de 2018:
    # sus respuestas se encuentran en 2019.
    last_development_origin = pd.Timestamp(
        "2018-12-31 00:00:00", tz="UTC"
    )

    is_development &= (
        model_df["utc_timestamp"] < last_development_origin
    )

    is_holdout = origin_year.eq(2019)

    # Exigir que el objetivo esté dentro del período observado.
    has_observed_target_time = (
        model_df["target_timestamp"] <= model_df["utc_timestamp"].max()
    )

    model_df["period"] = pd.NA
    model_df.loc[is_development, "period"] = "development"
    model_df.loc[is_holdout, "period"] = "holdout"

    # El indicador de período no garantiza que la respuesta exista.
    # Los objetivos ausentes se conservan como NA.
    print("\n--- OBSERVACIONES POR PERÍODO ---")
    print(model_df["period"].value_counts(dropna=False))

    print("\n--- RESPUESTAS AUSENTES POR PERÍODO ---")
    print(
        model_df.groupby("period", dropna=False)["target_demand_mw"]
        .apply(lambda s: s.isna().sum())
    )

    print("\n--- ORÍGENES CUYO OBJETIVO ESTÁ FUERA DE LA BASE ---")
    print((~has_observed_target_time).sum())

    # 7. Resumen de validación
    print("\n--- BASE ANALÍTICA ---")
    print(f"Filas: {len(model_df):,}")
    print(f"Columnas: {len(model_df.columns)}")
    print("\nFilas por país:")
    print(model_df.groupby("country").size())

    print("\nDemanda de origen ausente por país:")
    print(model_df.groupby("country")["demand_mw"].apply(
        lambda s: s.isna().sum()
    ))

    print("\nRespuesta a 24 horas ausente por país:")
    print(model_df.groupby("country")["target_demand_mw"].apply(
        lambda s: s.isna().sum()
    ))

    print("\nFilas por período:")
    print(model_df["period"].value_counts(dropna=False))

    # 8. Guardar la base transformada
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nBase guardada en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()


#Como se toma 2015-2018 para el desarrollo y 2019 para validacion
#se definen los conjuntos segun el año del origen del pronostico
#las ultimas 24 horas del 2018 no deben entrar en el desarrollo
#si sus respuestas ocurren en 2019
#(es una precaucion por el horizonte de prediccion: sus respuestas
#pertenecen al 2019. asi evitamos resultados contenidos en el conjunto
#reservado para la evaluacion)

#y la validacion corresponde a todos los orignenes de 2019
#cuya respuesta esté disponible en la base
#por lo tanto, no incluye objetivos del primer dia de 2019 que necesitan
#un origen en 2018.
#esto es una consecuencia explicita de definir el periodo por origen

#durante la validacion temporal, para cada origen t, solo pueden
#utilizarse en el entrenamiento observaciones cuya respuesta ya se 
#conozca en ese instante


model_df = pd.read_csv(OUTPUT_PATH)

# Inspección de la base analítica
print("\n--- PRIMERAS FILAS ---")
print(model_df.head(10).to_string(index=False))

print("\n--- ÚLTIMAS FILAS ---")
print(model_df.tail(10).to_string(index=False))

print("\n--- TIPOS DE DATOS ---")
print(model_df.dtypes)

print("\n--- DIMENSIONES ---")
print(model_df.shape)

print("\n--- MUESTRA DE CADA PAÍS ---")
for country in ["DE", "ES", "FR"]:
    print(f"\n{country}")
    print(
        model_df.loc[model_df["country"] == country]
        .head(3)
        .to_string(index=False)
    )
