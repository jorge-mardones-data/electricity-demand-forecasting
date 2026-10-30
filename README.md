(versión inicial, pendiente incluir procedimientos predictivos, la metodología de validación e instrucciones de reproducción)

# European Electricity Demand Ferecasting

## Proyect Overview

This project investigates how much the prediction of hourly electricity demand can be improved over seasonal forecasting baselines, using data from Germany, Spain and France.

The analisis focuses on 24-hour-ahead demand forecasting using historical electricity demand and calendar information.

## Data

The project uses the dataset provided for EYP3407, based on Open Power System Data (OPSD), edition 2020-10-06.

The original data covers Germany, Spain, and France from 2015 to 2019.

The original dataset is stored in `data/raw`. Processed dataset are generated through scripts in `src/`.

## Repository Structure

- `data/`: Original and processed datasets.
- `src/`: Data processing and analysis scripts.
- `notebooks/`: Exploratory analysis and model development.
- `reports/`: Project reports and supporting documents.

## Enviroment Setup

Python dependencies are listed in `requirements.txt`.

Install them using:

```bash
python -m pip install -r requirements.txt
```

## Reproducibility

Data transformations and analyses are implemented in the project repository.

Instructions for reproducting the complete analysis will be added as the project develops.

## Academic Context

EYP3407 - Tópicos Aplicados a la Estadística
Pontificia Universidad Católica de Chile

## Team

Jorge Mardones
Diego Paredes
Matías Pineda
Benjamin Ruiz