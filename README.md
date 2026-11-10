# Predicción de la demanda eléctrica horaria a 24 horas (Alemania, España y Francia)

Proyecto del curso EYP3407 – Tópicos Aplicados a la Estadística (Pontificia Universidad Católica de Chile), Caso 03, Grupo 03.

Se estudia cuánto puede mejorarse una predicción estacional de la demanda eléctrica horaria a 24 horas usando la demanda histórica y el calendario. La unidad de análisis es el par país–hora de emisión *t*, y la respuesta es la demanda (MW) en *t*+24. Se comparan dos referencias (persistencia y estacionalidad semanal) con Ridge, un árbol de regresión podado y un Random Forest, con validación temporal en tres ventanas (2015–2018) y una evaluación única en 2019. El informe completo está en `reports/informe/`.

## Datos

Paquete entregado para el caso 03, basado en *Open Power System Data* (OPSD), serie horaria, versión 2020-10-06 (origen: ENTSO-E Transparency). Cubre Alemania (DE), España (ES) y Francia (FR) entre 2015 y 2019. El archivo original está en `data/raw/` y no se modifica; la base analítica se genera en `data/processed/`.

## Estructura del repositorio

| Carpeta o archivo | Contenido |
|---|---|
| `analisis.py` | Punto de entrada único: reproduce todo el análisis y el informe |
| `data/raw/`, `data/processed/` | Datos originales y base analítica generada |
| `src/audit_data.py` | Auditoría de la base original |
| `src/build_dataset.py` | Base país–hora: respuesta, predictores y periodos |
| `src/modeling.py` | Particiones, especificación de los modelos y métricas compartidas |
| `src/run_validation.py` | Validación W1–W3 con los hiperparámetros fijados |
| `src/make_report_assets.py` | Tablas (`.tex`) y figuras del informe, a partir de `results/` |
| `notebooks/` | 02 diseño experimental; 03 desarrollo de modelos y búsqueda de hiperparámetros; 04 sensibilidades; 05 reserva 2019, incertidumbre y diagnóstico; 06 análisis complementarios |
| `results/` | Resultados (CSV) y figuras generados |
| `reports/informe/` | Informe en LaTeX (XeLaTeX) y su PDF |
| `docs/protocolo_experimental.md` | Protocolo experimental |

## Reproducción con un solo comando

Requisitos:

- Python 3.11 (probado con 3.11.9).
- Una distribución LaTeX con **XeLaTeX** (TeX Live o MiKTeX) para compilar el informe; no se instala con `pip`.

Desde la raíz del repositorio:

```bash
python -m pip install -r requirements.txt
python analisis.py
```

`analisis.py` ejecuta, en orden: auditoría de los datos, construcción de la base analítica, validación temporal, sensibilidades, evaluación en 2019 con bootstrap y diagnóstico, análisis complementarios, generación de tablas y figuras, y compilación del PDF del informe (`reports/informe/main.pdf`).

Opciones útiles:

```bash
python analisis.py --solo-informe          # solo tablas, figuras y PDF (usa los resultados ya existentes)
python analisis.py --pasos 1,2,7,8         # pasos elegidos (ver `python analisis.py --help`)
python analisis.py --sin-pdf               # omite la compilación LaTeX
python analisis.py --busqueda-completa     # repite la búsqueda de hiperparámetros (notebook 03)
```

Por defecto, la validación usa los hiperparámetros ya seleccionados (en `src/modeling.py` y `results/selected_configurations.csv`). La búsqueda completa de hiperparámetros del Random Forest (notebook 03) tarda varias horas; se activa solo con `--busqueda-completa`.

### Qué queda registrado en cada ejecución

- `results/run_info.json`: sistema, procesador, memoria, versiones de las librerías, tiempo de cada etapa y tiempo total (solo en la ejecución completa).
- `results/concordancia.csv`: diferencia entre los resultados regenerados y los que había antes de ejecutar (útil para la prueba desde una copia limpia).
- `results/ejecucion/`: registros de cada paso y copias ejecutadas de los notebooks (no se versionan). Los notebooks originales no se modifican.

### Tiempos por sección y comparación CPU vs. GPU

Cada notebook y `src/run_validation.py` guardan el tiempo de cada sección en `results/tiempos/`. Para repetir el análisis en una GPU NVIDIA (vía RAPIDS cuML, en Linux o WSL2) y comparar tiempos:

```bash
python analisis.py --gpu --sin-pdf
python src/comparar_tiempos.py
```

La corrida en GPU no reemplaza los resultados oficiales. Detalles en `docs/tiempos_cpu_gpu.md`.

### Notas sobre reproducibilidad

- Las semillas están fijadas (`random_state=42`), pero el Random Forest puede dar predicciones levemente distintas con otra versión de las librerías o con otro equipo. Por eso se fijan las versiones en `requirements.txt` y se registra la concordancia en `results/concordancia.csv`. Las cifras del informe corresponden a la ejecución registrada en `results/run_info.json`.
- Las tablas de búsqueda de hiperparámetros de Ridge, árbol y Random Forest (`results/ridge_search.csv`, `tree_search.csv`, `rf_search.csv`) se extraen de las salidas guardadas del notebook 03, que no las exporta por sí mismo.
- Los notebooks pueden ejecutarse también de forma individual desde Jupyter, en el orden 03 → 04 → 05 → 06.

## Datos de entrada

El archivo `data/raw/demanda_horaria.csv` proviene del paquete `caso03_demanda_electrica.zip` entregado por el curso (Open Power System Data, *Time series*, versión 2020-10-06; origen: ENTSO-E Transparency). No hace falta descargar nada más: `analisis.py` genera la base analítica y todos los resultados a partir de ese archivo.

## Pruebas de reproducción

| Fecha | Dónde | Comando | Resultado |
|---|---|---|---|
| 9-nov-2026 | Computador del autor del informe (Windows 11, Intel Core i5-14600K, 31,7 GB, Python 3.11.9): ejecución completa que genera los resultados del informe | `python analisis.py` | Sin errores, 12,2 min. Registro en `results/run_info.json` y `results/concordancia.csv` |
| 9-nov-2026 | Copia limpia de la carpeta del repositorio (sin `.git` ni registros previos), mismo computador | `python analisis.py` | Sin errores, 13,8 min, misma concordancia que la ejecución principal |
| _pendiente_ | Computador de otro integrante | `python analisis.py` | _por registrar (fecha, resultado y concordancia)_ |

## Contribuciones

- **Jorge Mardones:** creación y estructuración del repositorio y todo el contenido de su primera versión (datos, base analítica, protocolo experimental, notebooks 02 y 03).
- **Benjamín Ruiz:** secciones 3.3, 4.1 y 4.2 del enunciado (sensibilidades, evaluación en la reserva y métricas; notebooks 04 y 05 y `src/modeling.py`) y ejecución de la comparación de modelos.
- **Matías Pineda:** informe, análisis complementarios (notebook 06), tablas y figuras, `analisis.py`, `README.md` y `requirements.txt`.
- **Diego Paredes:** presentación del trabajo.

## Uso de inteligencia artificial

Se usó Claude Code (Anthropic) el 8 y 9 de noviembre de 2026 para el borrador y recorte del informe, la integración de resultados, y los archivos `src/make_report_assets.py`, `src/run_validation.py`, `analisis.py`, `README.md`, `requirements.txt` y el notebook 06. La declaración completa, con las comprobaciones realizadas, está en la sección 7 del informe.

## Contexto académico

EYP3407 – Tópicos Aplicados a la Estadística, Pontificia Universidad Católica de Chile. Profesor: Eloy Alvarado. Ayudante: Andrés Díaz.

## Equipo

Jorge Mardones, Diego Paredes, Matías Pineda y Benjamín Ruiz.
