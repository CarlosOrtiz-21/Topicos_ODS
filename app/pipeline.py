"""Pipeline de clasificacion de textos por ODS.

Reproduce la seccion "Modelado adicional - busqueda del mejor rendimiento"
del notebook (Notebook/microproyecto_2.ipynb): limpieza de texto, TF-IDF,
LSA (TruncatedSVD de 150 componentes) y regresion logistica, con los
hiperparametros que resultaron ganadores en el GridSearchCV de esa seccion.

Este modulo solo define la "receta" del pipeline (sin entrenar). El
entrenamiento y guardado del modelo vive en train.py.
"""

import re

import nltk
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LimpiezaTexto(BaseEstimator, TransformerMixin):
    """Quita URLs y etiquetas HTML, y normaliza espacios en blanco."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(self._limpiar)

    @staticmethod
    def _limpiar(texto):
        texto = str(texto)
        texto = re.sub(r"http[s]?://\S+", " ", texto)
        texto = re.sub(r"<[^>]+>", " ", texto)
        texto = re.sub(r"\s+", " ", texto).strip()
        return texto


def obtener_stopwords_es():
    """Devuelve la lista de stopwords en espanol, descargandola si hace falta."""
    try:
        return stopwords.words("spanish")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return stopwords.words("spanish")


def construir_pipeline():
    """Arma el pipeline completo (sin entrenar) con los hiperparametros
    ganadores del notebook: LSA de 150 componentes + regresion logistica
    (C=0.01, class_weight=None)."""
    stopwords_es = obtener_stopwords_es()

    return Pipeline(
        [
            ("limpieza", LimpiezaTexto()),
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words=stopwords_es,
                    ngram_range=(1, 2),
                    min_df=3,
                    max_df=0.9,
                    token_pattern=r"(?u)\b[^\W\d_]{3,}\b",
                ),
            ),
            ("svd", TruncatedSVD(n_components=150, random_state=42)),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=0.01,
                    class_weight=None,
                    max_iter=3000,
                    random_state=42,
                ),
            ),
        ]
    )


NOMBRES_ODS = {
    1: "Fin de la pobreza",
    2: "Hambre cero",
    3: "Salud y bienestar",
    4: "Educación de calidad",
    5: "Igualdad de género",
    6: "Agua limpia y saneamiento",
    7: "Energía asequible y no contaminante",
    8: "Trabajo decente y crecimiento económico",
    9: "Industria, innovación e infraestructura",
    10: "Reducción de las desigualdades",
    11: "Ciudades y comunidades sostenibles",
    12: "Producción y consumo responsables",
    13: "Acción por el clima",
    14: "Vida submarina",
    15: "Vida de ecosistemas terrestres",
    16: "Paz, justicia e instituciones sólidas",
    17: "Alianzas para lograr los objetivos",
}
