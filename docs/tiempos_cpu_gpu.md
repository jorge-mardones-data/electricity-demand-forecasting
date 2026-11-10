# Tiempos de ejecución: CPU vs. GPU

Cada notebook (02–06) y `src/run_validation.py` registran el tiempo de reloj de cada sección en
`results/tiempos/<notebook>__<backend>__<equipo>.csv`. El backend se elige con la variable de entorno
`EYP_BACKEND` (`cpu` por defecto, o `gpu`), que `analisis.py --gpu` fija automáticamente.

## Importante: scikit-learn no usa la GPU por sí solo

scikit-learn se ejecuta solo en CPU. Para usar una GPU NVIDIA (por ejemplo, una RTX 5060 de 8 GB) se usa
`cuml.accel`, de RAPIDS cuML. Este acelerador ejecuta en GPU los estimadores de scikit-learn que soporta,
sin cambiar el código, y deja en CPU los que no soporta.

- **Sistema:** cuML funciona en Linux o en **WSL2** (Windows Subsystem for Linux), no en Windows nativo.
- **Qué se acelera:** principalmente el Random Forest. Ridge ya tarda menos de 1 s. El árbol de decisión, el
  bootstrap (NumPy) y las tablas (pandas) pueden seguir en CPU.
- **Los resultados no son idénticos:** el Random Forest de cuML construye los árboles de otra forma (por
  ejemplo, discretiza los cortes en intervalos), así que el NMAE puede diferir levemente. La corrida en GPU
  **sirve para medir tiempos, no reemplaza los resultados oficiales**.

## Cómo comparar

Para que la comparación sea justa, conviene correr ambas versiones en el **mismo entorno** (WSL2).

1. Instalar WSL2 con Ubuntu y el driver NVIDIA reciente en Windows. Verificar dentro de WSL2 con `nvidia-smi`.
2. En WSL2, crear un entorno con Python 3.11, instalar `requirements.txt` y luego RAPIDS cuML siguiendo el
   selector oficial: https://docs.rapids.ai/install (elegir pip o conda, la versión de CUDA que indique
   `nvidia-smi` y WSL2). Las RTX serie 50 (Blackwell) requieren CUDA 12.8 o superior.
3. Desde la raíz del repositorio, en WSL2:

   ```bash
   python analisis.py --sin-pdf          # corrida CPU: results/tiempos/*__cpu__*.csv
   python analisis.py --gpu --sin-pdf    # corrida GPU: results/tiempos/*__gpu__*.csv
   python src/comparar_tiempos.py        # tabla y gráfico de la comparación
   ```

   Para comparar también la búsqueda de hiperparámetros (notebook 03, la parte más lenta), agregar
   `--busqueda-completa` a ambas corridas.

## Qué se genera

| Archivo | Contenido |
|---|---|
| `results/tiempos/<notebook>__<backend>__<equipo>.csv` | Segundos por sección, más CPU, GPU y versiones |
| `results/tiempos/run_info_gpu.json` | Tiempos por etapa de la corrida GPU (el oficial sigue siendo `results/run_info.json`) |
| `results/tiempos/concordancia_gpu.csv` | Cuánto difieren los resultados en GPU de los oficiales |
| `results/tiempos/comparacion_tiempos.csv` | Una fila por sección: CPU, GPU, aceleración (CPU/GPU) y ahorro |
| `results/tiempos/resumen_por_notebook.csv` | Tiempo total de cada notebook en cada backend |
| `results/figures/tiempos_cpu_vs_gpu.png` | Secciones más lentas en CPU y su tiempo en GPU |

La corrida con `--gpu` respalda los resultados oficiales antes de empezar y los restaura al terminar, aunque
haya un error. Así, los CSV y las figuras del informe siguen siendo los de la corrida en CPU.

## Notas

- Si un notebook se abre en Jupyter con `EYP_BACKEND=gpu`, la primera celda de código activa la GPU. Debe
  ejecutarse antes que cualquier otra celda que importe scikit-learn (el orden normal del notebook lo cumple).
- El tiempo de la primera sección en GPU incluye la carga de cuML y la inicialización de CUDA.
- `n_jobs=-1` no tiene efecto en GPU. En CPU usa todos los núcleos lógicos, por eso conviene comparar en el
  mismo computador.
