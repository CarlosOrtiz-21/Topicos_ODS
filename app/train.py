"""Entrena el pipeline de clasificacion de ODS y lo guarda en disco.

Se entrena con el 100% de los datos disponibles (no solo el train del
notebook): la evaluacion honesta del modelo ya se hizo en el notebook con
un conjunto de test separado, asi que el artefacto que se despliega en la
app puede aprovechar todos los textos disponibles.

Uso:
    python train.py
"""

from pathlib import Path

import joblib
import pandas as pd

from pipeline import construir_pipeline

DIRECTORIO_ACTUAL = Path(__file__).resolve().parent
RUTA_DATOS = DIRECTORIO_ACTUAL.parent / "Data" / "Datos_textosODS.xlsx"
RUTA_MODELO = DIRECTORIO_ACTUAL / "modelo_ods.pkl"


def main():
    print(f"Cargando datos desde: {RUTA_DATOS}")
    datos = pd.read_excel(RUTA_DATOS)

    pipeline = construir_pipeline()

    print("Entrenando pipeline (limpieza + TF-IDF + LSA + regresion logistica)...")
    pipeline.fit(datos["textos"], datos["ODS"])

    joblib.dump(pipeline, RUTA_MODELO)
    print(f"Modelo entrenado y guardado en: {RUTA_MODELO}")


if __name__ == "__main__":
    main()
