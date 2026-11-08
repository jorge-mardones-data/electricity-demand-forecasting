from pathlib import Path
import pandas as pd

# 1. Localizar el archivo original desde la raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "demanda_horaria.csv"

df = pd.read_csv(DATA_PATH)

df["utc_timestamp"] = pd.to_datetime(
    df["utc_timestamp"],
    utc=True,
    errors="coerce"
)

# 3. Definir las columnas de demanda
load_cols = [
    "DE_load_actual_entsoe_transparency",
    "ES_load_actual_entsoe_transparency",
    "FR_load_actual_entsoe_transparency",
]

def main():
    # 4. Revisar estructura general
    print("\n--- DIMENSIONES ---")
    print(f"Filas: {len(df):,}")
    print(f"Columnas: {len(df.columns)}")

    print("\n--- COLUMNAS Y TIPOS ---")
    print(df.dtypes)

    print("\n--- PRIMERAS FILAS ---")
    print(df.head())

    # 5. Revisar calidad temporal
    print("\n--- FECHAS UTC ---")
    print("Fecha mínima:", df["utc_timestamp"].min())
    print("Fecha máxima:", df["utc_timestamp"].max())
    print(
        "Fechas inválidas:",
        df["utc_timestamp"].isna().sum()
    )
    print(
        "Marcas temporales duplicadas:",
        df["utc_timestamp"].duplicated().sum()
    )

    ordered = df["utc_timestamp"].dropna().sort_values()
    gaps = ordered.diff().ne(pd.Timedelta(hours=1))
    print(
        "Saltos distintos de una hora en la serie ordenada:",
        gaps.sum() - 1 if len(gaps) else 0
    )

    # 6. Revisar ausencia y rangos de demanda
    print("\n--- VALORES AUSENTES ---")
    print(df[load_cols].isna().sum())

    print("\n--- RESUMEN DE DEMANDA ---")
    print(df[load_cols].describe().T)

    # 7. Revisar si hay valores de demanda no positivos
    print("\n--- VALORES NO POSITIVOS ---")
    print((df[load_cols] <= 0).sum())

    # 8. No se modifica ni se guarda la base original.


if __name__ == "__main__":
    main()


#¿Cuantas filas y columnas tiene el archivo?
#Filas: 43.824 y Columnas: 5

#¿Las fechas estan ordenadas y son unicas?
#estan ordenadas por UTC

#¿Existen horas faltantes o duplicadas?
#No existen saltos distintos de una hora en la serie ortdenada y no existen marcas temporales duplicadas

#¿Cuantos valores ausentes hay en cada pais?
#DE: 0 NA (0%)
#ES: 18 NA (0.041%)
#FR: 30 NA (0.068%)

#¿Los valores de demanda tienen razgos plausibles?
#Los rangos de cada pais son:
#DE con un minimo de 31307 y maximo de 77549, con una media de 55859.3
#ES con un minimo de 17168 y maximo de 41015, con una media de 28666.3
#FR con un minimo de 30426 y maximo de 94492, con una media de 53937.9
#cada uno presenta razgos plausibles en MW
#No existen valores negativos registrados

#---- Investigar valores faltantes------

for col in load_cols:
    missing = df.loc[df[col].isna(), ["utc_timestamp", col]]

    print(f"\n--- {col} ---")
    print(f"Cantidad de ausentes: {len(missing)}")

    if not missing.empty:
        print("Fechas con valores ausentes:")
        print(missing["utc_timestamp"].to_string(index=False))

        print("\nValores de demanda en las horas afectadas:")
        print(df.loc[
            df[col].isna(),
            ["utc_timestamp"] + load_cols
        ].to_string(index=False))



# --- CONVERSIÓN A HORA LOCAL ---
timezones = {
    "DE": "Europe/Berlin",
    "ES": "Europe/Madrid",
    "FR": "Europe/Paris",
}

for country, timezone in timezones.items():
    local_time = df["utc_timestamp"].dt.tz_convert(timezone)

    print(f"\n--- HORA LOCAL: {country} ---")
    print("Primeras fechas:")
    print(local_time.head(3).to_string(index=False))

    print("Últimas fechas:")
    print(local_time.tail(3).to_string(index=False))

    print("Cantidad de fechas locales sin hora de verano:")
    print((local_time.dt.month.isin([1, 2, 11, 12])).sum())

#Alemania, España y Francia comparten la misma regulación horaria durante todo el año
#todos se rigen por el CET (hora de Europa Central) y el CEST (hora de verano de Europa Central)
#en invierno todos tienen UTC+1 y el verano tienen UTC+2
#los tres paises cambian horario el mismo día y a la misma hora
#en el último domingo de Marzo y en el último domingo de Octubre 

#------- Interpretación de los valores faltantes------------
#Sabemos:
#Corresponden a la demanda de paises especificos y se concentran en algunos intervalos
#La ausencia de un pais no implica que también falten datos en los otros

#Hipótesis:
#Los intervalos podrían reflejar interrupciones en la publicación o recopilacion de datos,
#problemas con el sistema de reporte o registros que el proveedor no pudo obtener o validar.
#El hecho de que cada pais tenga ausecias distintas es compatible con problemas especificos
#de cada serie temporal

#No sabemos:
#No podemos afirmar que hubo fallas técnicas, problemas de medición o eventos particulares
#en el sistema electrico. Tampoco podemos concluir que los NA son aleatorios
#Para establecer la causa es necesario cdocumentacion adicional del proveedor  

#¿Cómo tratar los NAs?
#Por ahora, no los eliminaremos ni reemplazaremos por estimaciones sin necesidad
#Por ahora los conservaremos

#¿Por qué importan tanto si son tan pocos?
#Porque si queremos predecir la demanda a 24 horas (Y_(t+24))
#si es que falta un instante s, esto puede afectar varias observaciones:
#Puede afectar como respuesta, como predictor o como referencia de comparacion
#Por lo tanto, no debemos concluir que solo se pierden 18 observaciones utiles en ES
#el impacto depende de los predictores y rezagos que se definan

#----------Construccion de la base analítica--------------

#La base original tiene una fila por UTC y una columna de demanda por pais
#queremos definir la unidad de pais-hora
#transformaremos la base a un formato largo hora / pais / demanda
#se espera que queden 43824 x 3 = 131472 filas