"""App de Streamlit: clasifica un texto libre segun el ODS con el que
mas se relacione, usando el pipeline entrenado en train.py.

Uso:
    streamlit run app.py
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from pipeline import NOMBRES_ODS

RUTA_MODELO = Path(__file__).resolve().parent / "modelo_ods.pkl"


@st.cache_resource
def cargar_modelo():
    if not RUTA_MODELO.exists():
        return None
    return joblib.load(RUTA_MODELO)


st.set_page_config(page_title="Clasificador de textos ODS", page_icon="🌍")
st.title("🌍 Clasificador de textos por ODS")
st.write(
    "Escribe o pega un texto en español y el modelo predecirá a cuál de los "
    "Objetivos de Desarrollo Sostenible (ODS) está más relacionado."
)

modelo = cargar_modelo()

if modelo is None:
    st.error(
        "No se encontró el modelo entrenado (`modelo_ods.pkl`). "
        "Corre primero `python train.py` en esta misma carpeta."
    )
    st.stop()

texto_usuario = st.text_area(
    "Texto a clasificar",
    height=200,
    placeholder="Escribe aquí tu texto...",
)

if st.button("Predecir ODS", type="primary"):
    texto_limpio = texto_usuario.strip()

    if not texto_limpio:
        st.warning("Por favor ingresa un texto antes de predecir.")
    else:
        entrada = pd.Series([texto_limpio])
        prediccion = modelo.predict(entrada)[0]
        nombre_ods = NOMBRES_ODS.get(prediccion, "Desconocido")

        st.success(f"**ODS {prediccion} — {nombre_ods}**")

        if hasattr(modelo.named_steps["clf"], "predict_proba"):
            probabilidades = modelo.predict_proba(entrada)[0]
            clases = modelo.named_steps["clf"].classes_

            tabla_prob = (
                pd.DataFrame(
                    {
                        "ODS": clases,
                        "Nombre": [NOMBRES_ODS.get(c, "") for c in clases],
                        "Probabilidad": probabilidades,
                    }
                )
                .sort_values("Probabilidad", ascending=False)
                .head(5)
                .set_index("ODS")
            )
            tabla_prob["Probabilidad"] = (tabla_prob["Probabilidad"] * 100).round(1).astype(str) + "%"

            st.write("Top 5 ODS más probables:")
            st.dataframe(tabla_prob, use_container_width=True)
