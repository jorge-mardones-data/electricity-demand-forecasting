# Recursos de la presentación

Figuras y tablas para armar después la presentación en Beamer, generadas por
[`src/make_presentation_assets.py`](../src/make_presentation_assets.py) a partir de
`results/` y `data/processed/demanda_base_analitica.csv`. El script **no modifica**
datos, scripts ni resultados originales del proyecto; solo lee de ahí y escribe en
esta carpeta.

Para regenerar todo (sin reentrenar modelos, solo con los resultados ya calculados):

```bash
python src/make_presentation_assets.py
```

## Estructura

```
presentacion/
├── content/            reservado para los .tex de las diapositivas (aún no creados)
├── generated/           tablas .tex listas para \input en Beamer
│   └── respaldo/        tablas técnicas de respaldo (anexo / preguntas), sin adaptar en detalle
└── img/                 figuras .pdf (vectorial) y .png (300 dpi)
```

Todas las figuras se exportan en PDF y PNG con el mismo nombre base. Estilo: fuentes
grandes (16 pt base), sin títulos internos, sin bordes superior/derecho, paleta
Okabe-Ito consistente entre figuras y con el informe (azul `#0072B2` = DE/Ridge/
entrenamiento, naranja `#E69F00` = M2/validación, verde `#009E73` = M3, gris
`#999999`/`#555555` = referencias, naranja oscuro `#D55E00` = M3$^\ast$ cíclico/FR).

## Inventario

### Figuras principales

| Archivo | Contenido | Origen (informe) | Diapositiva sugerida |
|---|---|---|---|
| `img/flujo_muestra.pdf/.png` | Diagrama simplificado de construcción de la muestra (datos originales → desarrollo/reserva → casos completos) | Figura 1 (`img:flujo`), rehecha | Datos: construcción de la muestra |
| `img/validacion_temporal.pdf/.png` | Esquema de ventanas expansivas W1–W3 y evaluación final en la reserva 2019 | Figura 2 (`img:ventanas`), rehecha | Diseño de evaluación |
| `img/error_por_hora.pdf/.png` | NMAE por hora local, RF cíclico vs. referencia semanal, por país | Figura 3a (`img:diagnostico`), separada | Diagnóstico de errores |
| `img/error_diario.pdf/.png` | NMAE diario del RF cíclico en 2019, por país | Figura 3b (`img:diagnostico`), separada | Diagnóstico de errores |
| `img/perfil_hora.pdf/.png` | Perfil medio de demanda por hora local | Figura 4 (`img:descriptiva`), panel individual | Datos: patrones estacionales (o respaldo) |
| `img/perfil_dia.pdf/.png` | Perfil medio de demanda por día de la semana | Figura 4, panel individual | Datos: patrones estacionales (o respaldo) |
| `img/perfil_mes.pdf/.png` | Perfil medio de demanda por mes | Figura 4, panel individual | Datos: patrones estacionales (o respaldo) |
| `img/perfil_demanda.pdf/.png` | Los tres perfiles anteriores en un solo panel (1×3) | Figura 4, versión conjunta | Datos: patrones estacionales |
| `img/bootstrap_respaldo.pdf/.png` | Distribución bootstrap de la diferencia de NMAE macro (RF cíclico − referencia) | Figura 5 (`img:bootstrap`) | Respaldo de incertidumbre (preguntas) |

### Gráficos de comparación (reemplazan tablas numéricas)

| Archivo | Contenido | Origen (informe) | Diapositiva sugerida |
|---|---|---|---|
| `img/comparacion_modelos_desarrollo.pdf/.png` | Barras horizontales de NMAE macro por procedimiento, periodo de desarrollo (2015–2018), con línea de referencia semanal | Tabla 5 (`tab:global`) | Resultados en desarrollo |
| `img/comparacion_modelos_2019.pdf/.png` | Barras horizontales de NMAE macro por procedimiento, reserva 2019 | Tablas 7–8 (`tab:test`, `tab:test_rf`) | Evaluación final 2019 |
| `img/sensibilidad_calendario.pdf/.png` | Barras agrupadas: indicadores vs. seno–coseno (S1), Ridge y Random Forest | Tabla 6 (`tab:sens`), análisis S1 | Análisis de sensibilidad |
| `img/sensibilidad_historia.pdf/.png` | Barras agrupadas: ventana creciente vs. último año (S2), Ridge y Random Forest | Tabla 6 (`tab:sens`), análisis S2 | Análisis de sensibilidad |
| `img/errores_subgrupo.pdf/.png` | Reducción de MAE (%) frente a la referencia semanal, por subgrupo (laborable/fin de semana, estación, alta demanda) | Tabla 10 (`tab:diag`) | Diagnóstico de errores |

### Tablas `.tex` principales

| Archivo | Contenido | Origen (informe) | Diapositiva sugerida |
|---|---|---|---|
| `generated/tabla_variables.tex` | Variables admisibles y excluidas, versión de 3 columnas | Tabla 2 (`tab:variables`), simplificada | Problema y datos |
| `generated/tabla_ventanas.tex` | Ventanas de validación: periodos y tamaños | Tabla 3 (`tab:ventanas`) | Diseño de evaluación |
| `generated/resultados_paises.tex` | NMAE estacional vs. RF cíclico y mejora, por país y macro (reserva 2019) | Tablas 7–8 (`tab:test`, `tab:test_rf`), resumida | Evaluación final 2019 |
| `generated/intervalos_confianza.tex` | Mejora relativa e IC 95 % del RF cíclico frente a cada comparador (nivel macro) | Tabla 9 (`tab:boot`), resumida | Incertidumbre de la diferencia |
| `generated/casos_extremos.tex` | El día de mayor MAE por país en 2019, con feriado y sesgo | Tabla 18 (`tab:peores`), un caso por país | Casos extremos / límites del modelo |

### Tablas de respaldo (`generated/respaldo/`, sin adaptar en detalle)

| Archivo | Contenido | Origen (informe) |
|---|---|---|
| `tabla_respuesta.tex` | Descripción de la demanda a 24 h por país | Tabla 11 (`tab:respuesta`) |
| `tabla_detalle.tex` | MAE/RMSE/NMAE por procedimiento, ventana y país (desarrollo) | Tabla 12 (`tab:detalle`) |
| `tabla_test_detalle.tex` | MAE/RMSE/NMAE por procedimiento y país (reserva 2019) | Tabla 19 (`tab:test_detalle`) |
| `tabla_ridge.tex` | Búsqueda de hiperparámetros de Ridge | Tabla 13 (`tab:ridge`) |
| `tabla_rf_busqueda.tex` | Búsqueda de hiperparámetros de Random Forest | Tabla 14 (`tab:rf_busqueda`) |
| `tabla_complejidad.tex` | Complejidad de los modelos finales | Tabla 15 (`tab:complejidad`) |
| `tabla_tiempos.tex` | Tiempo de ajuste final por procedimiento | Tabla 16 (`tab:tiempos`) |
| `tabla_importancia.tex` | Importancia por impureza del Random Forest cíclico | Tabla 17 (`tab:importancia`) |
| `tabla_correlacion.tex` | Correlación entre predictores y la respuesta | Tabla 20 (`tab:correlacion`) |
| `tabla_ausentes_bloques.tex` | Bloques de horas consecutivas sin demanda observada | Tabla 21 (`tab:ausentes_bloques`) |
| `tabla_ausentes_efecto.tex` | Efecto de restringir a fechas con caso completo en los tres países | Tabla 22 (`tab:ausentes_efecto`) |

Estas se dejan listas por si surgen preguntas de detalle durante la defensa, pero no
tienen un rediseño visual pensado para proyectarse (mismo formato que el informe).

## Notas de reproducibilidad

- El bootstrap por bloques semanales (figura `bootstrap_respaldo` y tabla
  `intervalos_confianza`) se recalcula con la misma semilla (42) y se verifica contra
  `results/test_bootstrap_ci.csv` (exportado por el notebook 05), igual que en
  `make_report_assets.py`.
- Los números mostrados en `flujo_muestra` se calculan desde
  `data/processed/demanda_base_analitica.csv`, no están escritos a mano.
- `tabla_variables.tex` es contenido editorial (igual que su equivalente en el
  informe), no se deriva de datos.
