EYP3407 - Caso 03: Open Power System Data
Copia de datos preparada el 12 de octubre de 2026.

SUBCONJUNTO ENTREGADO
Alemania, España y Francia; todas las horas de 2015–2019. Columnas de demanda observada; se conservan ausencias.

ARCHIVOS Y VARIABLES
demanda_horaria.csv contiene cinco columnas:
- utc_timestamp: fecha y hora UTC; clave temporal para ordenar y enlazar.
- cet_cest_timestamp: representación CET/CEST, no hora local de cada país.
- DE_load_actual_entsoe_transparency: demanda observada de Alemania, MW.
- ES_load_actual_entsoe_transparency: demanda observada de España, MW.
- FR_load_actual_entsoe_transparency: demanda observada de Francia, MW.
La conversión de UTC al huso de cada país debe ser explícita.
Versión de Open Power System Data: 2020-10-06; se entregan horas de 2015–2019.

LECTURA Y USO
Los CSV utilizan UTF-8, coma separadora y punto decimal. Los nombres y códigos
originales se conservan. Un valor ausente no equivale a cero.
Descomprima este paquete en datos/ de su repositorio. Trabaje exclusivamente
con estos archivos y con la población y respuesta de la ficha del ANEXO.
Las transformaciones y el análisis deben programarse en el repositorio del equipo.
No es necesario descargar de nuevo las fuentes ni ejecutar código de extracción.

PROCEDENCIA
La copia entregada es el punto de partida de la reproducción. Los portales
pueden revisar sus datos después de la descarga. Cite al productor original,
la edición o fecha de descarga y las referencias del ANEXO. Consulte las
condiciones de reutilización de la fuente al publicar los datos.

Fuentes utilizadas en la preparación:
time_series_60min_singleindex.csv
https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv

opsd_metadata.json
https://data.open-power-system-data.org/time_series/2020-10-06/datapackage.json
