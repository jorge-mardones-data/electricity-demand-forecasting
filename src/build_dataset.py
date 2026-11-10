from pathlib import Path
import pandas as pd

PROJECT_ROOT= Path(__file__).resolve().parent.parent
INPUT_PATH= PROJECT_ROOT / "data" / "raw" / "demanda_horaria.csv"
OUTPUT_PATH= PROJECT_ROOT / "data" / "processed" / "demanda_modelado.csv"
DEMAND_COLUMNS= {"DE_load_actual_entsoe_transparency": "DE","ES_load_actual_entsoe_transparency": "ES","FR_load_actual_entsoe_transparency": "FR",}

#SE ESTABLECE 2015-2018 PARA DESARROLLO Y 2019 PARA EVALUACION

# 1. Leer la base original
df= pd.read_csv(INPUT_PATH)
df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True, errors="raise")

# 2. Transformar de formato ancho a formato largo
long_df= df.melt(id_vars=["utc_timestamp"],value_vars=list(DEMAND_COLUMNS.keys()),var_name="source_column",value_name="demand_mw",)
long_df["country"]= long_df["source_column"].map(DEMAND_COLUMNS)
long_df= long_df.drop(columns="source_column")
long_df= long_df[["utc_timestamp", "country", "demand_mw"]].sort_values(["country", "utc_timestamp"]).reset_index(drop=True)

# 3. Validar la unidad de análisis
expected_rows= len(df) * len(DEMAND_COLUMNS)
assert len(long_df) == expected_rows
assert not long_df.duplicated(["country", "utc_timestamp"]).any()

# 4. Construir el instante objetivo: 24 horas después del origen
long_df["target_timestamp"] = (long_df["utc_timestamp"] + pd.Timedelta(hours=24))

# 5. Vincular la demanda observada en el instante objetivo
target_df= long_df[["utc_timestamp", "country", "demand_mw"]].rename(columns={"utc_timestamp": "target_timestamp","demand_mw": "target_demand_mw",})
model_df= long_df.merge(target_df,on=["country", "target_timestamp"],how="left",validate="many_to_one",)

# 6. Definir los períodos según el año del origen
origin_year= model_df["utc_timestamp"].dt.year
is_development= origin_year.between(2015, 2018)

# Excluir los últimos 24 orígenes de 2018:
# sus respuestas se encuentran en 2019.
last_development_origin= pd.Timestamp("2018-12-31 00:00:00", tz="UTC")
is_development &= (model_df["utc_timestamp"] < last_development_origin)
is_holdout= origin_year.eq(2019)

# Exigir que el objetivo esté dentro del período observado.
has_observed_target_time= (model_df["target_timestamp"] <= model_df["utc_timestamp"].max())
model_df["period"]= pd.NA
model_df.loc[is_development, "period"] = "development"
model_df.loc[is_holdout, "period"] = "holdout"

# El indicador de período no garantiza que la respuesta exista.
# Los objetivos ausentes se conservan como NA.
print("\n--- OBSERVACIONES POR PERÍODO ---")
print(model_df["period"].value_counts(dropna=False))
print("\n--- RESPUESTAS AUSENTES POR PERÍODO ---")
print(model_df.groupby("period", dropna=False)["target_demand_mw"].apply(lambda s: s.isna().sum()))
print("\n--- ORÍGENES CUYO OBJETIVO ESTÁ FUERA DE LA BASE ---")
print((~has_observed_target_time).sum())

# 7. Resumen de validación
print("\n--- BASE ANALÍTICA ---")
print(f"Filas: {len(model_df):,}")
print(f"Columnas: {len(model_df.columns)}")
print("\nFilas por país:")
print(model_df.groupby("country").size())

print("\nDemanda de origen ausente por país:")
print(model_df.groupby("country")["demand_mw"].apply(lambda s: s.isna().sum()))
print("\nRespuesta a 24 horas ausente por país:")
print(model_df.groupby("country")["target_demand_mw"].apply(lambda s: s.isna().sum()))
print("\nFilas por período:")
print(model_df["period"].value_counts(dropna=False))

# 8. Guardar la base transformada
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
model_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nBase guardada en: {OUTPUT_PATH}")


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


#model_df = pd.read_csv(OUTPUT_PATH)

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
    print(model_df.loc[model_df["country"] == country].head(3).to_string(index=False))

#------------ Decisiones sobre el huso horario --------------
#Los tres paises mantienen el mismo UTC durante todo el año
#pero igualmente debemos realizar la conversion por pais
#Alemania: Europe/Berlin, España: EUrope/Madrid,, Francia: Europe/Paris
#Conservaremos el UTC referecnia

#------------ Elección de predictores y Rezagos -----------------
#Queremos estimar Y_t+24, los predictores deben estar disponibles en t o antes

#Los candidatos proncipales a predictor son:
# Y_t: demanda del dia anterior
# Y_t-24: demanda dos dias antes del objetivo
# Y_t-48: demanda tres dias antes del objetivo
# Y_t-144: demanda de la misma hora,una semana antes del objetivo

#Los Promedios retrospectivos--------
#rolling_mean_24h: promedio de las ultimas 24 horas hasta t
#rolling_mean_168h: promedio de la ultima semana (168 horas) hasta t

#Varaibles de calendario-----------------
#target_hour: hora local del objetivo
#target_day: dia de la semana local
#target_is_weekend: indicador de fin de semana
#target_month: mes local

# Zonas horarias oficiales por país
COUNTRY_TIMEZONES = {"DE": "Europe/Berlin","ES": "Europe/Madrid","FR": "Europe/Paris"}

# Inicializar columnas de tiempo local
model_df["local_timestamp"] = pd.Series(pd.NA, index=model_df.index, dtype="string")
model_df["target_local_timestamp"] = pd.Series(pd.NA, index=model_df.index, dtype="string")

# Conversión por país
for country, timezone in COUNTRY_TIMEZONES.items():
    mask= model_df["country"] == country

    # Hora local del origen
    origin_local= (model_df.loc[mask, "utc_timestamp"].dt.tz_convert(timezone))

    # Hora local del objetivo
    target_local= (model_df.loc[mask, "target_timestamp"].dt.tz_convert(timezone))

    # Guardar como ISO 8601, incluyendo el offset UTC
    model_df.loc[mask, "local_timestamp"]= (origin_local.map(lambda x: x.isoformat()).to_numpy())
    model_df.loc[mask, "target_local_timestamp"]= (target_local.map(lambda x: x.isoformat()).to_numpy())

# Inicializar variables de calendario
model_df["target_hour"] = pd.NA
model_df["target_day"] = pd.NA
model_df["target_is_weekend"] = pd.NA
model_df["target_month"] = pd.NA

# Extraer componentes del instante objetivo local
for country, timezone in COUNTRY_TIMEZONES.items():
    mask = model_df["country"] == country
    target_local= (model_df.loc[mask, "target_timestamp"].dt.tz_convert(timezone))
    model_df.loc[mask, "target_hour"]= target_local.dt.hour
    model_df.loc[mask, "target_day"]= target_local.dt.dayofweek
    model_df.loc[mask, "target_is_weekend"]= (target_local.dt.dayofweek.isin([5, 6]))
    model_df.loc[mask, "target_month"]= target_local.dt.month

model_df["target_hour"]= model_df["target_hour"].astype("Int64")
model_df["target_day"]= model_df["target_day"].astype("Int64")
model_df["target_month"]= model_df["target_month"].astype("Int64")
model_df["target_is_weekend"]= (model_df["target_is_weekend"].astype("boolean"))

# Base auxiliar con las observaciones históricas
demand_history= model_df[["country", "utc_timestamp", "demand_mw"]].copy()

# Diccionario de rezagos respecto del origen t
lag_offsets= {"lag_48h": 24,"lag_72h": 48,"lag_168h": 144}

# Identificador temporal para recuperar el orden original
model_df["_row_order"]= range(len(model_df))

for feature, offset in lag_offsets.items():
    # Timestamp histórico que necesitamos consultar
    lag_keys = model_df[["_row_order", "country", "utc_timestamp"]].copy()
    lag_keys["source_timestamp"] = (lag_keys["utc_timestamp"] - pd.Timedelta(hours=offset))
    # Renombrar la base histórica para realizar el enlace
    historical_values = demand_history.rename(columns={"utc_timestamp": "source_timestamp","demand_mw": feature})
    # Vincular por país y timestamp
    lag_values= lag_keys.merge(historical_values,on=["country", "source_timestamp"],how="left",validate="many_to_one")
    lag_values= lag_values.sort_values("_row_order")
    # Incorporar el predictor
    model_df[feature]= lag_values[feature].to_numpy()

# Eliminar identificador auxiliar
model_df= model_df.drop(columns="_row_order")

#aseguramos el orden temporal
model_df= model_df.sort_values(["country", "utc_timestamp"]).reset_index(drop=True)

#promedio de 24 horas
model_df["rolling_mean_24h"]= (model_df.groupby("country")["demand_mw"].transform(lambda s: s.rolling(window=24,min_periods=24).mean()))

#promedio de 168 horas
model_df["rolling_mean_168h"]= (model_df.groupby("country")["demand_mw"].transform(lambda s: s.rolling(window=168,min_periods=168).mean()))

#construccion de indicadores que describen informacion disponible
# Respuesta observada
model_df["has_observed_target"]= (model_df["target_demand_mw"].notna())

# Todos los rezagos disponibles
model_df["has_complete_lags"]= (model_df[["demand_mw", "lag_48h", "lag_72h", "lag_168h"]].notna().all(axis=1))

# Promedio retrospectivo de 24 horas disponible
model_df["has_complete_rolling_24h"]= (model_df["rolling_mean_24h"].notna())

# Promedio retrospectivo de 168 horas disponible
model_df["has_complete_rolling_168h"]= (model_df["rolling_mean_168h"].notna())

# Todos los predictores candidatos disponibles
model_df["has_complete_candidate_features"] = (
    model_df[["demand_mw","lag_48h","lag_72h","lag_168h","rolling_mean_24h","rolling_mean_168h"]].notna().all(axis=1))

#has_complete_candidate_features, indica que podemos generar una prediccion con todos los predictores candidatos
#has_observed_target, indica que conocemos el valor real para evaluar esa prediccion

#inspeccion
# Dimensiones
print("Dimensiones de model_df:", model_df.shape)

# Columnas
print("\nColumnas:")
print(model_df.columns.tolist())

# Primeras observaciones
print("\nPrimeras filas:")
print(model_df.head().to_string(index=False))

# Tipos de datos
print("\nTipos de datos:")
print(model_df.dtypes)

# NA por variable
print("\nValores faltantes por variable:")
print(model_df.isna().sum().sort_values(ascending=False))

# Disponibilidad de predictores por país
print("\nPredictores completos por país:")
print(model_df.groupby("country")["has_complete_candidate_features"].sum())

# Respuestas observadas por país
print("\nRespuestas observadas por país:")
print(model_df.groupby("country")["has_observed_target"].sum())

#Guardamos la nueva version
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
model_df.to_csv(OUTPUT_PATH,index=False)

print(f"\nBase enriquecida guardada en: {OUTPUT_PATH}")

#interpretacion de los NAs

#rolling_mean_168h, 1582 NAs, ventanas semanales incompletas
#lag_168h,           480 NAs, demanda no observada en el rezago
#rolling_mean_24h,   280 NAs, ventanas diarias incompletas
#lag_72h,            192 NAs, demanda no observada en el rezago
#lag_48h,            120 NAs, demanda no observada en el rezago
#target_demand_mw,   118 NAs, respuesta no observada
#demand_mw,           48 NAs, demanda no observada en el origen

#auditoria datos faltantes

#filas sin clasificacion temporal para period (72 NAs)
period_na= model_df.loc[model_df["period"].isna(),["country", "utc_timestamp", "target_timestamp"]].copy()

print("Cantidad de filas sin period:", len(period_na))
print("\nDistribución por país:")
print(period_na["country"].value_counts().sort_index())
print("\nFechas mínimas y máximas por país:")
print(period_na.groupby("country")["utc_timestamp"].agg(["min", "max", "count"]))
print("\nPrimeras y últimas filas sin clasificación:")
print(period_na.head(10))
print(period_na.tail(10))

#origen de los target NA

# Última observación temporal disponible para cada país
last_timestamp= (model_df.groupby("country")["utc_timestamp"].max())

# Última fecha disponible de la serie de demanda
model_df["last_available_timestamp"]= (model_df["country"].map(last_timestamp))

# Identificar si el target está fuera del período disponible
model_df["target_outside_dataset"]= (model_df["target_timestamp"]> model_df["last_available_timestamp"])

# Revisar exclusivamente respuestas faltantes
target_na= model_df.loc[model_df["target_demand_mw"].isna()].copy()

print("Total de target NA:", len(target_na))
print("\nClasificación de los target NA:")
print(target_na["target_outside_dataset"].value_counts(dropna=False))

print("\nClasificación por país:")
print(pd.crosstab(target_na["country"],target_na["target_outside_dataset"]))

#Auditoria, continuidad temporal por pais
model_df= model_df.sort_values(["country", "utc_timestamp"]).reset_index(drop=True)

# Diferencia entre observaciones consecutivas
model_df["time_diff"]= (model_df.groupby("country")["utc_timestamp"].diff())

print("Diferencias temporales por país:")
print(model_df.groupby("country")["time_diff"].value_counts(dropna=False))

# Verificar duplicados de la clave país-hora
duplicate_keys= model_df.duplicated(subset=["country", "utc_timestamp"]).sum()
print("\nDuplicados país-hora:", duplicate_keys)

# Verificar intervalos distintos de una hora
irregular_intervals= model_df.loc[model_df["time_diff"].notna()& (model_df["time_diff"] != pd.Timedelta(hours=1)),["country", "utc_timestamp", "time_diff"]]
print("\nIntervalos distintos de una hora:")
print(irregular_intervals.head(20))
print("Total:", len(irregular_intervals))

#Validacion de rezagos
# Cada offset indica cuántas horas retrocedemos desde el origen t.
# demand_mw ya representa la demanda en t.

lag_offsets= {"lag_48h": 24,"lag_72h": 48,"lag_168h": 144}

for variable, offset in lag_offsets.items():
    # Reconstruir el valor esperado en t - offset
    expected = model_df[["country", "utc_timestamp", "demand_mw"]].copy()
    expected["utc_timestamp"] = (expected["utc_timestamp"]+ pd.Timedelta(hours=offset))
    expected = expected.rename(columns={"demand_mw": "expected_value"})

    # Comparar con el predictor construido
    comparison= model_df[["country", "utc_timestamp", variable]].merge(expected,on=["country", "utc_timestamp"],how="left",validate="one_to_one")
    # Un valor es correcto si ambos son NA o si son iguales
    same_values = (comparison[variable].eq(comparison["expected_value"])| (comparison[variable].isna()& comparison["expected_value"].isna()))
    print(f"\n{variable}")
    print("Coincidencias:", same_values.sum())
    print("Diferencias:", (~same_values).sum())

#Los tres reultados tienen cero diferencias, rezagos validados.

#validacion de medias moviles

# Asegurar orden temporal
model_df= model_df.sort_values(["country", "utc_timestamp"]).reset_index(drop=True)
grouped_demand= model_df.groupby("country")["demand_mw"]

# Reconstrucción independiente de las medias
expected_rolling_24h= grouped_demand.transform(lambda s: s.rolling(window=24, min_periods=24).mean())

expected_rolling_168h= grouped_demand.transform(lambda s: s.rolling(window=168, min_periods=168).mean())

# Comparación con las variables existentes
for variable, expected in {"rolling_mean_24h": expected_rolling_24h,"rolling_mean_168h": expected_rolling_168h}.items():
    same_values= (
        model_df[variable].eq(expected)| (model_df[variable].isna()& expected.isna()))
    differences= (model_df[variable] - expected).abs()
    print(f"\n{variable}")
    print("Coincidencias:", same_values.sum())
    print("Diferencias:", (~same_values).sum())
    if differences.notna().any():
        print("Máxima diferencia absoluta:",differences.max())

# Auditoria, ¿Cuantas observaciones tenemos realmente disponibles para desarrollar y evaluar modelos en cada pais?

availability= (model_df.groupby(["country", "period"], dropna=False).agg(total_origins=("utc_timestamp", "size"),observed_targets=("target_demand_mw", "count"),missing_targets=("target_demand_mw",lambda s: s.isna().sum()),complete_candidate_features=("has_complete_candidate_features", "sum")).reset_index())
availability["target_availability_pct"]= (100* availability["observed_targets"]/ availability["total_origins"])

availability["complete_features_pct"]= (100 * availability["complete_candidate_features"]/ availability["total_origins"])
print(availability.to_string(index=False))

#disponibilidad de cada predictor

candidate_predictors= ["demand_mw","lag_48h","lag_72h","lag_168h","rolling_mean_24h","rolling_mean_168h","target_hour","target_day","target_is_weekend","target_month"]

predictor_availability= (model_df.groupby(["country", "period"], dropna=False)[candidate_predictors].agg(lambda s: s.notna().sum()))
print(predictor_availability)

#Auditoria, coherencia de banderas

expected_complete= model_df[candidate_predictors].notna().all(axis=1)
print("Diferencias en has_complete_candidate_features:",(model_df["has_complete_candidate_features"]!= expected_complete).sum())
expected_lags = model_df[["demand_mw", "lag_48h", "lag_72h", "lag_168h"]].notna().all(axis=1)
print("Diferencias en has_complete_lags:",(model_df["has_complete_lags"]!= expected_lags).sum())
expected_rolling_24h = model_df["rolling_mean_24h"].notna()
print("Diferencias en has_complete_rolling_24h:",(model_df["has_complete_rolling_24h"]!= expected_rolling_24h).sum())
expected_rolling_168h = model_df["rolling_mean_168h"].notna()
print("Diferencias en has_complete_rolling_168h:",(model_df["has_complete_rolling_168h"]!= expected_rolling_168h).sum())

# Guardar la base analitica final
OUTPUT_FINAL_PATH = (PROJECT_ROOT/ "data"/ "processed"/ "demanda_base_analitica.csv")
model_df.to_csv(OUTPUT_FINAL_PATH,index=False)
print("Base analítica guardada en:", OUTPUT_FINAL_PATH)
print("Dimensiones:", model_df.shape)