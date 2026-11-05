## Protocolo experimental - Predicción de demanda eléctrica a 24 horas

1. Objetivo

Evaluar cuánto puede mejorarse una predicción de la demanda eléctrica horaria a 24 horas con modelos estadísticos y aprendizaje automático.

Se consideran datos de demanda eléctrica de Alemania (DE), España (ES) y Francia (FR), usando datos horarios de 2015-2019.

2. Unidad de análisis y variable objetivo

- Unidad de análisis: país-hora de demanda en t.
- Horizonte: 24 horas.
- Variable respuesta: `target_demand_mw`, que es la demanda en MW observada en t+24.
- Referencia temporal: UTC para ordenar las observaciones, construir las particiones y controlar el horizonte.

3. Predictores candidatos

Se utilizarán inicialmente estos diez predictores:

| Predictor | Descripción |
| ------------ | ------------ |
| `country` | País |
| `demand_mw` | Demanda en el origen de predicción t |
| `lag_48h` | Demanda histórica |
| `lag_72h` | Demanda histórica |
| `lag_168h` | Demanda de la misma hora en la semana anterior |
| `rolling_mean_24h` | Media móvil de 24 horas |
| `rolling_mean_168h` | Media móvil de 168 horas |
| `target_hour` | Hora local del objetivo |
| `target_day` | Día de la semana local del objetivo |
| `target_month` | Mes local del objetivo |

Para la regresión regularizada, las variables categóricas se van a cidificar mediante variables dummy y las numéricas se escalarán. Los parámetros del procesamiento se van a estimar con los datos del entrenamiento unicamente.

4. Separación temporal

Se separaron los datos en dos periodos:

- Desarrollo: 2015-2018.
- Validación externa: 2019.

EL periodo final de 2019 será reservado hasta finalizar la selección de modelos y todas las decisiones metodológicas.

### Validación temporal interna

Se usarán tres ventanas expansivas:

| Ventana | Entrenamiento | Validación |
| ------------ | ------------ | ------------ |
| W1 | 2015 | 2016 |
| W2 | 2015-2016 | 2017 |
| W3 | 2015-2017 | 2018 |

Para evitar fuga de información, las observaciones de entrenamiento deben cumplir que el `target_timestamp` sea siempre menor que el primer origen de validación. Así, las observaciones donde el objetivo atraviesa el inicio de la validación serán excluidas del entrenamiento.

5. Procedimientos de predicción

Se compararán dos referencias y tres familias de modelos:

### Modelos de referencia

- 1. Persistencia: donde la predicción es igual a la demanda observada en el origen (`demand_mw`).
- 2. Estacional semanal: donde la predicción es igual a la demanda de la misma hora, pero de la semana anterior (`lag_168h`).

Se optará por la referencia con menor NMAE macro promedio en las tres ventanas.

### Modelos de aprendizaje

- 1. M1 - regresión regularizada: Ridge o Lasso.
- 2. M2 - Modelo flexible: árbol de regresión con poda.
- 3. M3 - Ensamble: Random Forest o Boosting.

La búsqueda de hiperparámetros se determinará en el periodo de desarrollo, con a lo más 20 configuraciones por familia y tres particiones temporales internas.

6. Tratamiento de observaciones faltantes

Se optó por no eliminar valores faltantes, ni reemplazarlos con imputación automática.

Para la comparación de los procedimientos sobre una base común, serán utilizados observaciones con variable respuesta y los diez predictores disponibles. Se usará este criterio para el entrenamiento y para la evaluación de cada ventana.

Las cantidades de observaciones que serán utilizadas y excluidas van a ser informadas.

7. Métricas de evaluación

La métrica principal que se usará para comparar modelos es NMAE macro. Para cada país, `NMAE_c` será igual a `MAE_c` dividido por la media de la demanda observada en el conjunto de entrenamiento para el país `c`. Y la métrica macro será el promedio de las tres ventanas:

$$NMAE_{macro}= (NMAE_{DE}+NMAE_{ES}+NMAE_{FR})/3$$

Algunas métricas complementarias serían `MAE` y `RMSE` por país, expresado en MW

 8. Análisis de sensibilidad

 Se realizarán dos análisis de sensibilidad:

 - Representacion del calendario: Donde se compara la codificación categórica del calendario con una representación cíclica.
 - Información histórica: Donde se evalúa la sensibilidad del desempeño al incluir información histórica a largo plazo versus incluir información historica a corto plazo.

 Ambos análisis se realizarán con los datos de desarrollo, sin usar los datos de validación final.

 9. Evaluación final

 Una vez seleccionados los procedimientos y las decisiones metodológicas, se ajustan los datos de desarrollo disponibles y se evaluarán una sola vez sobre el conjunto de validación correspondiente al año 2019.

10. Principios de reproducibilidad

- Conservar los datos originales, sin modificaciones.
- Construir la base analítica mediante un código reproducible.
- Documentar las decisiones metodológicas.
- Fijar las semillas cuando corresponda.
- Ajustar las transformaciones solamente con el conjunto de entrenamiento.
- Registrar las configuraciones, métricas y resultados.
- Usar los datos de 2019 exclusivamente en la evaluación final. 
