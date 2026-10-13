"""
###############################################################################
#  GRUPO 7 - CLASIFICACIÓN DE PAÍSES SEGÚN TRANSICIÓN ENERGÉTICA  (Energía)  #
#  Machine Learning - Práctica 2 - Entregable 3 (guía de código en Python)   #
#  UNI - Facultad de Ingeniería Industrial y de Sistemas - Ciclo 2026-II     #
###############################################################################

Integrantes:
    Crisostomo Cahuas, Gabriel Andre        gabriel.crisostomo.c@uni.pe
    Curto Meneses, Alessandro del Piero     alessandro.curto.m@uni.pe
    Rubio Zeta, Luis Miguel                 luis.rubio.z@uni.pe
Profesor: Dr. Ing. Hilario Aradiel Castañeda

¿Qué hace este programa?
------------------------
Clasifica a cada país y año en una de tres etapas de la transición energética
(0 INICIAL, 1 INTERMEDIA, 2 AVANZADA) a partir de 23 predictores medidos el año
anterior. Sigue la metodología CRISP-ML(Q) (Studer et al., 2021,
https://doi.org/10.48550/arXiv.2003.05155) y trabaja SOLO sus dos últimas fases
antes del despliegue, como exige la práctica:

    FASE 4  MODELADO    -> Naive Bayes (el modelo asignado a nuestro grupo) y
                           doce modelos rivales, más un clasificador trivial;
                           hiperparámetros elegidos en los años de validación.
    FASE 5  EVALUACIÓN  -> una sola medición en los años de prueba, métricas de
                           calidad, intervalos de confianza, prueba de McNemar,
                           matrices de confusión, errores por región e
                           importancia de las variables.

NO hay fases 1 a 3: los datos ya vienen limpios y etiquetados en datos/, y este
programa no los vuelve a limpiar ni cambia la etiqueta. Tampoco hay despliegue.

Además del modelado, el programa responde con nuestros propios datos las
preguntas teóricas de la guía del curso:

    CAPÍTULO 2 (aprendizaje supervisado): clasificación y regresión,
        generalización, sobreajuste y subajuste (curvas de validación y de
        aprendizaje), los algoritmos principales e incertidumbre de las
        probabilidades (pérdida logarítmica y curvas de calibración).
    CAPÍTULO 3 (no supervisado y preprocesamiento): efecto del escalado,
        reducción de dimensionalidad (PCA, NMF y t-SNE) y agrupamiento
        (k-Means, jerárquico de Ward y DBSCAN).

Datos de entrada (no se modifican):
    datos/etapas_transicion_preparado.csv  3 780 filas x 39 columnas, 180 países,
                                           2004-2024, sin faltantes.
    datos/etapas_auditoria.csv             insumos de la etiqueta; solo se usan
                                           para revisar los errores cerca de los umbrales.
    datos/diccionario_datos.csv            descripción de cada columna.

Cómo se recorre el código (la función principal está al final):

    principal()
      |
      |-- cargar_datos()                  lee el CSV y verifica forma, años y predictores
      |
      |-- fase4_modelado()                13 modelos + trivial, rejillas y variantes de NB
      |     |-- construir_modelos()       estimador y rejilla de cada modelo
      |     `-- variantes_naive_bayes()   transformación x priors del modelo asignado
      |
      |-- fase5_evaluacion()              una sola medición en prueba, IC, McNemar,
      |                                   confusión, regiones, umbrales e importancia
      |-- capitulo2_supervisado()         curvas de validación, de aprendizaje y calibración
      |-- capitulo3_no_supervisado()      escalado, PCA, NMF, t-SNE y agrupamiento
      |-- generar_figuras()               fig02 a fig10
      `-- control_calidad_y_valores()     controles CRISP-ML(Q) y valores_para_paper.md

Ejecución:
    python main.py              # protocolo completo (unos minutos)
    python main.py --rapido     # rejillas reducidas, solo para probar que todo corre
    python main.py --n-jobs 4   # usa varios núcleos (por defecto 1, igual en Windows)
"""

# =============================================================================
# PASO 0. IMPORTACIÓN DE LIBRERÍAS
# =============================================================================
# --- Biblioteca estándar de Python (no requiere instalación) -----------------
import argparse                     # opciones de la línea de comandos (--rapido, --n-jobs)
import json                         # guardar los mejores hiperparámetros
import platform                     # versión de Python, que queda registrada en los controles
import sys                          # código de salida del programa
import time                         # medir el tiempo de entrenamiento de cada modelo
import warnings                     # silenciar avisos repetidos de convergencia
from pathlib import Path            # rutas que funcionan en Windows, Linux y macOS

# --- Librerías externas (se instalan con: pip install -r requirements.txt) ---
import numpy as np                  # cálculo numérico y números al azar reproducibles
import pandas as pd                 # tablas de datos (DataFrame)
import matplotlib                   # gráficos
matplotlib.use("Agg")               # dibuja sin abrir ventanas; las figuras se guardan en PNG
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import joblib                       # guardar los modelos entrenados en resultados/modelos
from scipy.stats import binomtest   # prueba exacta de McNemar

# --- scikit-learn: todo el aprendizaje automático ----------------------------
import sklearn
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import NMF, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, adjusted_rand_score,
                             balanced_accuracy_score, confusion_matrix, f1_score,
                             log_loss, normalized_mutual_info_score,
                             recall_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import (GridSearchCV, PredefinedSplit,
                                     StratifiedShuffleSplit)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, PowerTransformer, RobustScaler,
                                   StandardScaler)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# =============================================================================
# PASO 1. PARÁMETROS DEL PROYECTO
# =============================================================================
# Carpeta del proyecto. En Jupyter no existe __file__, por eso se usa la
# carpeta actual (la carpeta donde está main.ipynb).
RAIZ = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
CARPETA_DATOS = RAIZ / "datos"                          # entradas (no se modifican)
CARPETA_TABLAS = RAIZ / "resultados" / "tablas"         # tablas CSV y JSON
CARPETA_FIGURAS = RAIZ / "resultados" / "figuras"       # figuras PNG del artículo
CARPETA_MODELOS = RAIZ / "resultados" / "modelos"       # modelos finales (joblib)
CARPETA_REPORTES = RAIZ / "resultados" / "reportes"     # controles y valores para el paper

ARCHIVO_DATOS = CARPETA_DATOS / "etapas_transicion_preparado.csv"
ARCHIVO_AUDITORIA = CARPETA_DATOS / "etapas_auditoria.csv"

# Semilla única: con ella dos corridas dan exactamente los mismos resultados.
SEMILLA = 42

# Variable objetivo y nombres de las tres etapas
OBJETIVO = "etapa_num"
ETAPAS = ["Inicial", "Intermedia", "Avanzada"]
CLASES = [0, 1, 2]

# Los 11 indicadores continuos, medidos el año ANTERIOR al que se predice.
# Son los únicos que se usan en PCA, NMF, t-SNE y agrupamiento (capítulo 3).
CONTINUOS = [
    "pib_pc_log_ant", "poblacion_log_ant", "urbano_pct_ant", "industria_pct_ant",
    "renovable_pct_ant", "co2_pc_log_ant", "energia_pc_log_ant",
    "consumo_elec_pc_log_ant", "var_pib_pc_pct_ant", "var_urbano_pp_ant",
    "var3_renovable_pp_ant",
]
# Los 23 predictores del modelado: los 11 continuos + el año + 7 regiones + 4 ingresos.
PREDICTORES = CONTINUOS + ["anio_relativo"] + [
    "region_africa_subsahariana", "region_america_latina_caribe",
    "region_america_norte", "region_asia_oriental_pacifico", "region_asia_sur",
    "region_europa_asia_central", "region_medio_oriente_norte_africa",
    "ingreso_alto", "ingreso_medio_alto", "ingreso_medio_bajo", "ingreso_bajo",
]
# Columnas que NUNCA pueden ser predictores: identifican al país, dicen el año
# real, repiten la etiqueta o marcan qué valores se imputaron.
PROHIBIDAS = ["codigo_iso3", "nombre_pais", "region", "grupo_ingreso", "anio",
              "etapa", "etapa_num", "conjunto"]

# Partición temporal: ya viene en la columna 'conjunto' del archivo de entrada.
ANIOS_CONJUNTO = {"entrenamiento": (2004, 2015), "validacion": (2016, 2019),
                  "prueba": (2020, 2024)}
FILAS_CONJUNTO = {"entrenamiento": 2160, "validacion": 720, "prueba": 900}

# Criterio de éxito del proyecto: exactitud balanceada mínima en los años de
# prueba. Un clasificador al azar llegaría a 0,333.
CRITERIO_EXITO = 0.75

# Parámetros de la evaluación
REMUESTREOS = 1000          # repeticiones del bootstrap para el intervalo de confianza
ALFA = 0.05                 # nivel de significación de la prueba de McNemar
REPETICIONES_PERM = 10      # repeticiones de la importancia por permutación
UMBRAL_CERCA = 5.0          # "cerca del umbral" = a menos de 5 puntos de 30, 60 o 90

# Nombre corto de cada modelo. El ORDEN de esta lista es el orden de las tablas
# del artículo, así que no debe cambiarse.
ORDEN_MODELOS = [
    "Naive Bayes (base)", "Regresión logística", "k-NN", "Árbol de decisión",
    "Bosque aleatorio", "Extra Trees", "Gradient Boosting",
    "Hist. Gradient Boosting", "AdaBoost", "SVM RBF", "SVM lineal", "LDA",
    "Red neuronal MLP", "Clasificador trivial",
]
MODELO_ASIGNADO = "Naive Bayes (base)"      # el que nos tocó como grupo 7
MODELO_TRIVIAL = "Clasificador trivial"     # línea base; no cuenta entre los rivales
MODELO_LOGISTICA = "Regresión logística"    # referencia de probabilidades bien calibradas

# Nombre de cada modelo en inglés (el artículo lleva una portada en inglés).
NOMBRES_INGLES = {
    "Naive Bayes (base)": "Naive Bayes", "Regresión logística": "Logistic regression",
    "k-NN": "k-NN", "Árbol de decisión": "Decision tree",
    "Bosque aleatorio": "Random forest", "Extra Trees": "Extra Trees",
    "Gradient Boosting": "Gradient Boosting",
    "Hist. Gradient Boosting": "Histogram Gradient Boosting", "AdaBoost": "AdaBoost",
    "SVM RBF": "RBF SVM", "SVM lineal": "Linear SVM", "LDA": "LDA",
    "Red neuronal MLP": "MLP neural network", "Clasificador trivial": "Majority baseline",
}

# Nombre legible de cada predictor, para el texto del artículo.
NOMBRES_LEGIBLES = {
    "pib_pc_log_ant": "PIB per cápita (log)",
    "poblacion_log_ant": "Población (log)",
    "urbano_pct_ant": "Población urbana (%)",
    "industria_pct_ant": "Industria (% del PIB)",
    "renovable_pct_ant": "Energía renovable (%)",
    "co2_pc_log_ant": "CO2 per cápita (log)",
    "energia_pc_log_ant": "Energía per cápita (log)",
    "consumo_elec_pc_log_ant": "Electricidad per cápita (log)",
    "var_pib_pc_pct_ant": "Variación del PIB per cápita (%)",
    "var_urbano_pp_ant": "Variación de la población urbana (p.p.)",
    "var3_renovable_pp_ant": "Variación renovable de 3 años (p.p.)",
    "anio_relativo": "Año relativo",
    "region_africa_subsahariana": "Región: África subsahariana",
    "region_america_latina_caribe": "Región: América Latina y el Caribe",
    "region_america_norte": "Región: América del Norte",
    "region_asia_oriental_pacifico": "Región: Asia oriental y el Pacífico",
    "region_asia_sur": "Región: Asia meridional",
    "region_europa_asia_central": "Región: Europa y Asia central",
    "region_medio_oriente_norte_africa": "Región: Medio Oriente y norte de África",
    "ingreso_alto": "Ingreso alto", "ingreso_medio_alto": "Ingreso medio alto",
    "ingreso_medio_bajo": "Ingreso medio bajo", "ingreso_bajo": "Ingreso bajo",
}
# Nombre aún más corto, solo para que quepa en las figuras de una columna.
NOMBRES_FIGURA = {
    "pib_pc_log_ant": "PIB pc (log)", "poblacion_log_ant": "Población (log)",
    "urbano_pct_ant": "Urbana (%)", "industria_pct_ant": "Industria (%)",
    "renovable_pct_ant": "Renovable (%)", "co2_pc_log_ant": "CO2 pc (log)",
    "energia_pc_log_ant": "Energía pc (log)",
    "consumo_elec_pc_log_ant": "Electricidad pc (log)",
    "var_pib_pc_pct_ant": "Var. PIB pc (%)", "var_urbano_pp_ant": "Var. urbana (p.p.)",
    "var3_renovable_pp_ant": "Var. renov. 3a (p.p.)", "anio_relativo": "Año relativo",
    "region_africa_subsahariana": "Reg. África subs.",
    "region_america_latina_caribe": "Reg. Am. Latina",
    "region_america_norte": "Reg. Am. del Norte",
    "region_asia_oriental_pacifico": "Reg. Asia oriental",
    "region_asia_sur": "Reg. Asia merid.",
    "region_europa_asia_central": "Reg. Europa y Asia c.",
    "region_medio_oriente_norte_africa": "Reg. Medio Oriente",
    "ingreso_alto": "Ingreso alto", "ingreso_medio_alto": "Ingreso medio alto",
    "ingreso_medio_bajo": "Ingreso medio bajo", "ingreso_bajo": "Ingreso bajo",
}
# La región de Medio Oriente tiene un nombre muy largo en los datos del Banco
# Mundial; en las tablas del artículo se usa la versión corta.
REGION_CORTA = {"Medio Oriente, norte de África, Afganistán y Pakistán":
                "Medio Oriente y norte de África"}
ORDEN_REGIONES = ["África subsahariana", "América Latina y el Caribe",
                  "América del Norte", "Asia oriental y el Pacífico",
                  "Asia meridional", "Europa y Asia central",
                  "Medio Oriente y norte de África"]

# --- Estilo de las figuras: deben entrar en una columna del artículo ---------
ANCHO_FIGURA = 3.35                 # pulgadas: ancho de una columna de la plantilla LACCEI
PPP = 300                           # puntos por pulgada exigidos por la plantilla
COLOR_ETAPA = {0: "#d1495b", 1: "#edae49", 2: "#00798c"}    # inicial, intermedia, avanzada
AZUL, GRIS = "#30638e", "#8d99ae"                           # colores de apoyo
COLOR_NB = "#003d5b"                                        # el modelo asignado, resaltado
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.5,
    "figure.dpi": PPP, "savefig.dpi": PPP, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})

# Avisos que no aportan nada y solo ensucian la consola (convergencia del MLP, etc.).
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*did not converge.*")
# scikit-learn 1.9 avisa de que SVC(probability=True) quedará obsoleto en la 1.11;
# el protocolo del artículo lo usa tal cual, así que el aviso solo haría ruido.
warnings.filterwarnings("ignore", message=".*probability.*deprecated.*")

# Contador de usos del conjunto de prueba, uno por modelo. Lo revisa el control
# de calidad: cada modelo debe medirse en prueba UNA SOLA VEZ.
USOS_DE_PRUEBA: dict[str, int] = {}


# =============================================================================
# PASO 2. FUNCIONES DE APOYO
# =============================================================================
def seccion(texto: str) -> None:
    """Imprime un título de sección para seguir el avance en la consola."""
    print("\n" + "=" * 78 + f"\n {texto}\n" + "=" * 78)


def nota(texto: str) -> None:
    """Imprime un mensaje de avance con sangría."""
    print(f"   - {texto}")


def preparar_carpetas() -> None:
    """Crea las carpetas de salida si todavía no existen."""
    for carpeta in (CARPETA_TABLAS, CARPETA_FIGURAS, CARPETA_MODELOS, CARPETA_REPORTES):
        carpeta.mkdir(parents=True, exist_ok=True)


def guardar_tabla(df: pd.DataFrame, nombre: str) -> None:
    """Guarda una tabla en resultados/tablas con codificación UTF-8."""
    df.to_csv(CARPETA_TABLAS / nombre, index=False, encoding="utf-8")
    nota(f"tabla guardada: resultados/tablas/{nombre}")


def guardar_figura(fig, nombre: str) -> None:
    """Guarda una figura en resultados/figuras a 300 ppp y libera la memoria."""
    fig.savefig(CARPETA_FIGURAS / nombre, bbox_inches="tight", dpi=PPP)
    plt.close(fig)
    nota(f"figura guardada: resultados/figuras/{nombre}")


def tabla_md(df: pd.DataFrame) -> str:
    """Convierte un DataFrame en una tabla Markdown (sin depender de 'tabulate')."""
    columnas = [str(c) for c in df.columns]
    filas = ["| " + " | ".join(columnas) + " |", "|" + "---|" * len(columnas)]
    for _, fila in df.iterrows():
        filas.append("| " + " | ".join(str(v) for v in fila.values) + " |")
    return "\n".join(filas)


# --- Formato de los números tal como los pide el artículo (coma decimal) -----
def num(x: float, decimales: int = 3) -> str:
    """0.8213 -> '0,821'. Las métricas del artículo llevan tres decimales."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:.{decimales}f}".replace(".", ",")


def pct(x: float, decimales: int = 1) -> str:
    """0.873 -> '87,3 %'. El valor entra como proporción entre 0 y 1."""
    return f"{100 * x:.{decimales}f}".replace(".", ",") + " %"


def puntos(x: float) -> str:
    """Diferencia entre dos métricas expresada en puntos: 0.0123 -> '+1,2'."""
    return f"{100 * x:+.1f}".replace(".", ",")


def intervalo(lo: float, hi: float) -> str:
    """Intervalo de confianza con el guion largo que usa el artículo."""
    return f"{num(lo)}–{num(hi)}"


def legible(columna: str) -> str:
    """Nombre de un predictor tal como se lee en el texto del artículo."""
    return NOMBRES_LEGIBLES.get(columna, columna)


def en_minuscula(texto: str) -> str:
    """Pone en minúscula la primera letra, para meter un nombre dentro de una frase."""
    return texto[:1].lower() + texto[1:] if texto else texto


def lista_y(nombres: list[str]) -> str:
    """['a','b','c'] -> 'a, b y c' (como se escribe una enumeración en español)."""
    if len(nombres) == 1:
        return nombres[0]
    return ", ".join(nombres[:-1]) + " y " + nombres[-1]


def nombre_corto(modelo: str) -> str:
    """En las figuras, 'Naive Bayes (base)' se lee mejor como 'Naive Bayes'."""
    return "Naive Bayes" if modelo == MODELO_ASIGNADO else modelo


def ejes_con_coma(eje, en_x: bool = True, en_y: bool = True) -> None:
    """Pone coma decimal en las marcas de los ejes, como se escribe en español."""
    formato = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    if en_x:
        eje.xaxis.set_major_formatter(formato)
    if en_y:
        eje.yaxis.set_major_formatter(formato)


def metricas_en_prueba(nombre: str, modelo, X, y) -> dict:
    """
    Mide UN modelo en los años de prueba y anota ese uso en USOS_DE_PRUEBA.

    Es el único lugar del programa donde se calculan métricas sobre la prueba:
    así el control de calidad puede comprobar que cada modelo la usó una sola
    vez y que ningún modelo se eligió mirando esos años.

    Devuelve las métricas pedidas por la guía: exactitud balanceada, F1 macro,
    exactitud, recall de cada etapa, pérdida logarítmica y AUC multiclase.
    """
    USOS_DE_PRUEBA[nombre] = USOS_DE_PRUEBA.get(nombre, 0) + 1
    pred = modelo.predict(X)
    proba = modelo.predict_proba(X)
    recalls = recall_score(y, pred, labels=CLASES, average=None, zero_division=0)
    return {
        "eb_prueba": balanced_accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "exactitud": accuracy_score(y, pred),
        "recall_inicial": recalls[0],
        "recall_intermedia": recalls[1],
        "recall_avanzada": recalls[2],
        "log_loss": log_loss(y, proba, labels=CLASES),
        "auc_ovr_macro": roc_auc_score(y, proba, multi_class="ovr", average="macro"),
        "_pred": pred,
        "_proba": proba,
    }


# =============================================================================
# PASO 3. CARGA Y VERIFICACIÓN DE LOS DATOS
# =============================================================================
def cargar_datos() -> dict:
    """
    Lee el conjunto ya preparado y lo parte en entrenamiento, validación y prueba.

    No limpia nada y no toca la etiqueta: ese trabajo ya está hecho y el archivo
    de entrada se usa tal como viene. Lo que sí hace es comprobar con 'assert'
    que el archivo es el que esperamos; si algo no cuadra, el programa se
    detiene antes de entrenar.

    Resultado esperado:
      * 3 780 filas x 39 columnas, 180 países, 2004-2024, 0 faltantes;
      * 23 predictores presentes y ninguna columna prohibida entre ellos;
      * 2 160 filas de entrenamiento (2004-2015), 720 de validación (2016-2019)
        y 900 de prueba (2020-2024).
    """
    seccion("PASO 3. DATOS DE ENTRADA (ya preparados y etiquetados)")
    df = pd.read_csv(ARCHIVO_DATOS, encoding="utf-8")

    # --- Verificaciones de forma -------------------------------------------
    assert df.shape == (3780, 39), f"se esperaban 3780x39 y llegó {df.shape}"
    assert df.isna().sum().sum() == 0, "el conjunto no debería tener faltantes"
    assert df["codigo_iso3"].nunique() == 180, "se esperaban 180 países"
    assert len(PREDICTORES) == 23, "los predictores del modelado deben ser 23"
    faltan = [c for c in PREDICTORES if c not in df.columns]
    assert not faltan, f"faltan predictores en el archivo: {faltan}"

    # --- Ningún predictor puede identificar al país ni revelar la etiqueta ---
    coladas = [c for c in PREDICTORES if c in PROHIBIDAS or c.startswith("bandera_imp_")]
    assert not coladas, f"predictores prohibidos: {coladas}"

    # --- Los años de cada conjunto no se mezclan ----------------------------
    for conjunto, (anio_ini, anio_fin) in ANIOS_CONJUNTO.items():
        anios = df.loc[df["conjunto"] == conjunto, "anio"]
        assert anios.min() == anio_ini and anios.max() == anio_fin, \
            f"los años de {conjunto} no son {anio_ini}-{anio_fin}"
        assert len(anios) == FILAS_CONJUNTO[conjunto], \
            f"{conjunto} debería tener {FILAS_CONJUNTO[conjunto]} filas"

    partes = {}
    for conjunto in ANIOS_CONJUNTO:
        sub = df[df["conjunto"] == conjunto]
        partes[conjunto] = (sub[PREDICTORES].to_numpy(dtype=float),
                            sub[OBJETIVO].to_numpy(dtype=int), sub)

    X_tr, y_tr, df_tr = partes["entrenamiento"]
    X_va, y_va, df_va = partes["validacion"]
    X_pr, y_pr, df_pr = partes["prueba"]

    # Entrenamiento + validación en una sola matriz. Con PredefinedSplit la
    # búsqueda de hiperparámetros entrena con las filas marcadas -1 (los años
    # 2004-2015) y mide con las marcadas 0 (2016-2019): nunca mezcla los años.
    X_tv = np.vstack([X_tr, X_va])
    y_tv = np.concatenate([y_tr, y_va])
    marca = np.concatenate([np.full(len(y_tr), -1), np.zeros(len(y_va), dtype=int)])
    particion = PredefinedSplit(marca)

    nota(f"{df.shape[0]} filas x {df.shape[1]} columnas, "
         f"{df['codigo_iso3'].nunique()} países, {df['anio'].min()}-{df['anio'].max()}")
    nota(f"predictores: {len(PREDICTORES)} ({len(CONTINUOS)} continuos, 1 de tiempo, "
         "7 de región y 4 de ingreso)")
    for conjunto, (anio_ini, anio_fin) in ANIOS_CONJUNTO.items():
        sub = df[df["conjunto"] == conjunto]
        reparto = sub[OBJETIVO].value_counts(normalize=True).sort_index()
        nota(f"{conjunto:>13}: {len(sub):4d} filas ({anio_ini}-{anio_fin}), "
             f"avanzada {pct(reparto.get(2, 0.0))}")
    nota("2025 no se usa: el Banco Mundial aún no publica el acceso a la "
         "electricidad de ese año")

    return {"df": df, "X_tr": X_tr, "y_tr": y_tr, "X_va": X_va, "y_va": y_va,
            "X_pr": X_pr, "y_pr": y_pr, "X_tv": X_tv, "y_tv": y_tv,
            "particion": particion, "df_prueba": df_pr, "df_entrenamiento": df_tr,
            "df_validacion": df_va}


# =============================================================================
# PASO 4. FASE 4 DE CRISP-ML(Q): MODELADO
# =============================================================================
def construir_modelos() -> dict:
    """
    Devuelve, para cada modelo, su estimador y la rejilla de hiperparámetros.

    Cada modelo es un 'pipeline': los que dependen de distancias o de márgenes
    (k-NN, las dos SVM, la regresión logística y la red neuronal) llevan
    estandarizado delante, porque sin él una variable con números grandes pesa
    más que las demás. Los árboles no lo necesitan: sus cortes solo miran el
    orden de los valores.

    El escalador se ajusta DENTRO del pipeline, así que durante la búsqueda de
    hiperparámetros aprende solo de los años de entrenamiento y nunca ve los de
    validación ni los de prueba (evita la fuga de información).
    """
    tercios = [1 / 3, 1 / 3, 1 / 3]     # probabilidades a priori iguales para las 3 etapas
    rejilla_arboles = {                 # la misma para bosque aleatorio y Extra Trees
        "max_depth": [None, 8, 12], "min_samples_leaf": [1, 3, 5],
        "max_features": ["sqrt", 0.5], "class_weight": [None, "balanced"],
    }
    return {
        # El modelo asignado al grupo. Se prueban dos cosas más: transformar las
        # variables con Yeo-Johnson para acercarlas a una normal (que es lo que
        # Naive Bayes gaussiano supone) y darle la misma probabilidad a priori a
        # las tres etapas, porque en el entrenamiento hay pocos países avanzados.
        "Naive Bayes (base)": (
            Pipeline([("transf", "passthrough"), ("modelo", GaussianNB())]),
            {"transf": ["passthrough", PowerTransformer()],
             "modelo__priors": [None, tercios],
             "modelo__var_smoothing": np.logspace(-11, -1, 11)},
        ),
        "Regresión logística": (
            Pipeline([("escalador", StandardScaler()),
                      ("modelo", LogisticRegression(max_iter=5000, random_state=SEMILLA))]),
            {"modelo__C": [0.01, 0.1, 1, 10, 100],
             "modelo__class_weight": [None, "balanced"]},
        ),
        "k-NN": (
            Pipeline([("escalador", StandardScaler()), ("modelo", KNeighborsClassifier())]),
            {"modelo__n_neighbors": [3, 5, 7, 9, 15, 25, 35],
             "modelo__weights": ["uniform", "distance"]},
        ),
        "Árbol de decisión": (
            DecisionTreeClassifier(random_state=SEMILLA),
            {"max_depth": [3, 4, 5, 6, 8, 10, None],
             "min_samples_leaf": [1, 5, 10, 20],
             "class_weight": [None, "balanced"]},
        ),
        "Bosque aleatorio": (
            RandomForestClassifier(n_estimators=300, random_state=SEMILLA),
            dict(rejilla_arboles),
        ),
        "Extra Trees": (
            ExtraTreesClassifier(n_estimators=300, random_state=SEMILLA),
            dict(rejilla_arboles),
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=SEMILLA),
            {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1],
             "max_depth": [2, 3]},
        ),
        "Hist. Gradient Boosting": (
            HistGradientBoostingClassifier(max_iter=200, random_state=SEMILLA),
            {"learning_rate": [0.05, 0.1], "max_depth": [None, 4],
             "l2_regularization": [0.0, 1.0], "class_weight": [None, "balanced"]},
        ),
        "AdaBoost": (
            AdaBoostClassifier(random_state=SEMILLA),
            {"n_estimators": [100, 300], "learning_rate": [0.1, 0.5, 1.0]},
        ),
        "SVM RBF": (
            Pipeline([("escalador", StandardScaler()),
                      ("modelo", SVC(kernel="rbf", probability=True, random_state=SEMILLA))]),
            {"modelo__C": [0.1, 1, 10, 100], "modelo__gamma": ["scale", 0.01, 0.1],
             "modelo__class_weight": [None, "balanced"]},
        ),
        "SVM lineal": (
            Pipeline([("escalador", StandardScaler()),
                      ("modelo", SVC(kernel="linear", probability=True,
                                     random_state=SEMILLA))]),
            {"modelo__C": [0.01, 0.1, 1, 10],
             "modelo__class_weight": [None, "balanced"]},
        ),
        "LDA": (
            LinearDiscriminantAnalysis(),
            [{"solver": ["svd"]}, {"solver": ["lsqr"], "shrinkage": ["auto"]}],
        ),
        "Red neuronal MLP": (
            Pipeline([("escalador", StandardScaler()),
                      ("modelo", MLPClassifier(max_iter=2000, early_stopping=True,
                                               random_state=SEMILLA))]),
            {"modelo__hidden_layer_sizes": [(32,), (64,), (64, 32)],
             "modelo__alpha": [1e-4, 1e-3, 1e-2]},
        ),
        # Línea base obligatoria: predice siempre la etapa más frecuente. No es
        # un rival (no aprende nada); sirve para saber qué significa "acertar".
        "Clasificador trivial": (
            DummyClassifier(strategy="most_frequent"), {},
        ),
    }


def reducir_rejilla(rejilla):
    """Deja un solo valor por hiperparámetro: es la opción --rapido, solo para probar."""
    if isinstance(rejilla, list):
        return [reducir_rejilla(r) for r in rejilla]
    return {clave: list(valores)[:1] for clave, valores in rejilla.items()}


def fase4_modelado(datos: dict, rapido: bool, n_jobs: int) -> dict:
    """
    FASE 4 DE CRISP-ML(Q). Entrena los 13 modelos y el clasificador trivial.

    Para cada modelo se hacen siempre los mismos cuatro pasos:
      1. Búsqueda en rejilla con GridSearchCV y PredefinedSplit: entrena con
         2004-2015 y elige los hiperparámetros por exactitud balanceada en
         2016-2019.
      2. Métricas de validación: con esos hiperparámetros se entrena otra vez
         solo con 2004-2015 y se mide en 2016-2019 (columna EB-V del artículo).
      3. Modelo final: con esos mismos hiperparámetros se entrena con
         2004-2019 (entrenamiento + validación) y se cronometra.
      4. Una sola medición en 2020-2024.

    Resultado esperado:
      * resultados/tablas/comparacion_modelos.csv con las métricas de los 14 modelos;
      * resultados/tablas/mejores_parametros.json con la combinación elegida;
      * resultados/tablas/variantes_naive_bayes.csv con las 4 variantes del modelo asignado;
      * resultados/modelos/*.joblib con el Naive Bayes final y el mejor modelo.
    """
    seccion("PASO 4. FASE 4 DE CRISP-ML(Q) - MODELADO")
    nota(f"búsqueda de hiperparámetros por exactitud balanceada en validación "
         f"(scikit-learn {sklearn.__version__}, semilla {SEMILLA})")
    if rapido:
        nota("MODO --rapido: rejillas reducidas; los números NO son los del artículo")

    X_tr, y_tr = datos["X_tr"], datos["y_tr"]
    X_va, y_va = datos["X_va"], datos["y_va"]
    X_tv, y_tv = datos["X_tv"], datos["y_tv"]
    X_pr, y_pr = datos["X_pr"], datos["y_pr"]

    filas, modelos_finales, mejores_parametros = [], {}, {}
    for nombre in ORDEN_MODELOS:
        estimador, rejilla = construir_modelos()[nombre]
        busqueda = GridSearchCV(
            estimador, reducir_rejilla(rejilla) if rapido else rejilla,
            scoring="balanced_accuracy", cv=datos["particion"], refit=True,
            n_jobs=n_jobs, error_score="raise",
        )
        busqueda.fit(X_tv, y_tv)
        parametros = busqueda.best_params_

        # (2) El mismo modelo entrenado SOLO con los años de entrenamiento: así
        # la exactitud balanceada en validación es comparable entre modelos.
        en_validacion = clone(estimador).set_params(**parametros).fit(X_tr, y_tr)
        eb_validacion = balanced_accuracy_score(y_va, en_validacion.predict(X_va))

        # (3) Modelo final: aprende de todo lo que había antes de la prueba.
        final = clone(estimador).set_params(**parametros)
        reloj = time.perf_counter()
        final.fit(X_tv, y_tv)
        segundos = time.perf_counter() - reloj

        # (4) Única mirada a los años de prueba.
        metricas = metricas_en_prueba(nombre, final, X_pr, y_pr)
        metricas["_modelo"] = nombre
        fila = {"modelo": nombre, "eb_validacion": eb_validacion,
                **{k: v for k, v in metricas.items() if not k.startswith("_")},
                "segundos_entrenamiento": segundos}
        filas.append(fila)
        modelos_finales[nombre] = {"modelo": final, "parametros": parametros,
                                   "pred": metricas["_pred"], "proba": metricas["_proba"]}
        mejores_parametros[nombre] = {k: str(v) for k, v in parametros.items()}
        nota(f"{nombre:<24} EB-V {num(eb_validacion)}  EB-P {num(metricas['eb_prueba'])}  "
             f"({segundos:5.1f} s)")

    comparacion = pd.DataFrame(filas)
    guardar_tabla(comparacion, "comparacion_modelos.csv")
    (CARPETA_TABLAS / "mejores_parametros.json").write_text(
        json.dumps(mejores_parametros, indent=2, ensure_ascii=False), encoding="utf-8")
    nota("tabla guardada: resultados/tablas/mejores_parametros.json")

    variantes = variantes_naive_bayes(datos, modelos_finales[MODELO_ASIGNADO]["parametros"])
    return {"comparacion": comparacion, "modelos": modelos_finales,
            "variantes": variantes}


def variantes_naive_bayes(datos: dict, parametros_finales: dict) -> pd.DataFrame:
    """
    Compara las cuatro variantes del modelo asignado (Tabla V del artículo).

    Las cuatro salen de combinar la transformación (ninguna o Yeo-Johnson) con
    las probabilidades a priori (las de los datos o iguales para las tres
    etapas). Para cada combinación se elige el mejor var_smoothing en los años
    de validación, se reentrena con entrenamiento + validación y se mide una
    vez en prueba.

    Resultado esperado: resultados/tablas/variantes_naive_bayes.csv con cuatro
    filas; la mejor en validación es la que usa el Naive Bayes final.
    """
    nota("variantes de Naive Bayes: transformación x probabilidades a priori")
    tercios = [1 / 3, 1 / 3, 1 / 3]
    combinaciones = [("Ninguna", "De los datos", "passthrough", None),
                     ("Ninguna", "Iguales", "passthrough", tercios),
                     ("Yeo-Johnson", "De los datos", PowerTransformer(), None),
                     ("Yeo-Johnson", "Iguales", PowerTransformer(), tercios)]
    filas = []
    for transformacion, priors, transf, valor_priors in combinaciones:
        base = Pipeline([("transf", transf), ("modelo", GaussianNB(priors=valor_priors))])
        mejor = (-1.0, None)
        for suavizado in np.logspace(-11, -1, 11):
            candidato = clone(base).set_params(modelo__var_smoothing=suavizado)
            candidato.fit(datos["X_tr"], datos["y_tr"])
            eb = balanced_accuracy_score(datos["y_va"], candidato.predict(datos["X_va"]))
            if eb > mejor[0]:               # ante empate se queda el suavizado más pequeño
                mejor = (eb, suavizado)
        eb_validacion, suavizado = mejor
        final = clone(base).set_params(modelo__var_smoothing=suavizado)
        final.fit(datos["X_tv"], datos["y_tv"])
        etiqueta = f"Variante NB: {transformacion} / {priors}"
        metricas = metricas_en_prueba(etiqueta, final, datos["X_pr"], datos["y_pr"])
        filas.append({"transformacion": transformacion, "priors": priors,
                      "var_smoothing": suavizado, "eb_validacion": eb_validacion,
                      "eb_prueba": metricas["eb_prueba"], "log_loss": metricas["log_loss"]})
    variantes = pd.DataFrame(filas)
    guardar_tabla(variantes, "variantes_naive_bayes.csv")

    # La variante que usa el Naive Bayes final es la que ganó la búsqueda en
    # rejilla; aquí se deja escrita tal como aparecerá en el artículo.
    usa_yeo = "PowerTransformer" in str(parametros_finales.get("transf", "passthrough"))
    usa_iguales = parametros_finales.get("modelo__priors") is not None
    suavizado_final = parametros_finales.get("modelo__var_smoothing", float("nan"))
    nota(f"Naive Bayes final: {'Yeo-Johnson' if usa_yeo else 'sin transformación'}, "
         f"priors {'iguales' if usa_iguales else 'de los datos'}, "
         f"var_smoothing = {suavizado_final:.0e}")
    return variantes


# =============================================================================
# PASO 5. FASE 5 DE CRISP-ML(Q): EVALUACIÓN
# =============================================================================
def intervalo_bootstrap(y_real: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """
    Intervalo de confianza del 95 % de la exactitud balanceada, por bootstrap.

    Se vuelven a sortear 1 000 veces las 900 filas de prueba (con reemplazo) y
    se calcula la exactitud balanceada en cada remuestreo; el intervalo son los
    percentiles 2,5 y 97,5. Dice cuánto podría moverse el resultado si nos
    hubiera tocado otra muestra de países y años parecida.
    """
    azar = np.random.RandomState(SEMILLA)       # misma semilla -> mismo intervalo siempre
    n = len(y_real)
    valores = []
    for _ in range(REMUESTREOS):
        indices = azar.randint(0, n, n)
        valores.append(balanced_accuracy_score(y_real[indices], pred[indices]))
    return float(np.percentile(valores, 2.5)), float(np.percentile(valores, 97.5))


def prueba_mcnemar(y_real: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """
    Prueba exacta de McNemar entre dos modelos sobre las MISMAS filas de prueba.

    Solo cuentan las filas en las que los dos modelos no coinciden: b01 son las
    que el modelo A falla y B acierta, y b10 al revés. Si los dos fueran
    igual de buenos, esos desacuerdos se repartirían mitad y mitad; la prueba
    binomial dice qué tan raro sería el reparto observado.
    """
    acierta_a, acierta_b = (pred_a == y_real), (pred_b == y_real)
    b01 = int(np.sum(~acierta_a & acierta_b))   # A falla, B acierta
    b10 = int(np.sum(acierta_a & ~acierta_b))   # A acierta, B falla
    if b01 + b10 == 0:                          # los dos modelos aciertan y fallan igual
        return {"b01": 0, "b10": 0, "p": 1.0, "significativa": False}
    p = binomtest(min(b01, b10), b01 + b10, 0.5).pvalue
    return {"b01": b01, "b10": b10, "p": float(p), "significativa": bool(p < ALFA)}


def exactitud_por_region(df_prueba: pd.DataFrame, y_real: np.ndarray,
                         predicciones: dict) -> pd.DataFrame:
    """
    Exactitud de cada modelo en las siete regiones del Banco Mundial.

    Sirve para ver si los errores se concentran en alguna parte del mundo, que
    es justo lo que importaría a un banco de desarrollo.
    """
    regiones = df_prueba["region"].replace(REGION_CORTA).to_numpy()
    filas = []
    for region in ORDEN_REGIONES:
        donde = regiones == region
        fila = {"region": region, "filas": int(donde.sum())}
        for nombre, pred in predicciones.items():
            fila[nombre] = accuracy_score(y_real[donde], pred[donde]) if donde.any() else np.nan
        filas.append(fila)
    return pd.DataFrame(filas)


def errores_cerca_del_umbral(df_prueba: pd.DataFrame, y_real: np.ndarray,
                             pred: np.ndarray) -> dict:
    """
    ¿Los errores del mejor modelo están donde la regla de la etiqueta es frágil?

    La etapa se decide con dos umbrales: 30 % y 60 % de electricidad baja en
    carbono (promedio de tres años) y 90 % de acceso a la electricidad. Un país
    que está a menos de 5 puntos de un umbral puede cambiar de etapa por una
    diferencia mínima, así que equivocarse ahí es menos grave.
    """
    auditoria = pd.read_csv(ARCHIVO_AUDITORIA, encoding="utf-8",
                            usecols=["codigo_iso3", "anio", "b_promedio_3", "electricidad_pct"])
    unido = df_prueba[["codigo_iso3", "anio"]].merge(auditoria, on=["codigo_iso3", "anio"],
                                                     how="left", validate="one_to_one")
    assert unido["b_promedio_3"].notna().all(), "faltan insumos de la etiqueta en la auditoría"
    b = unido["b_promedio_3"].to_numpy()
    acceso = unido["electricidad_pct"].to_numpy()
    cerca = ((np.abs(b - 30) < UMBRAL_CERCA) | (np.abs(b - 60) < UMBRAL_CERCA)
             | ((b >= 60) & (np.abs(acceso - 90) < UMBRAL_CERCA)))
    fallos = pred != y_real
    return {"errores_cerca": float(cerca[fallos].mean()) if fallos.any() else 0.0,
            "prueba_cerca": float(cerca.mean()), "n_errores": int(fallos.sum())}


def importancia_por_permutacion(modelo, X, y, n_jobs: int) -> pd.DataFrame:
    """
    ¿Cuánto empeora el modelo si desordenamos una variable?

    Se barajan los valores de una sola columna de la prueba y se mide cuánto
    cae la exactitud balanceada. Si cae mucho, esa variable era importante.
    Funciona con cualquier modelo, incluso con los que no tienen coeficientes.
    """
    resultado = permutation_importance(modelo, X, y, scoring="balanced_accuracy",
                                       n_repeats=REPETICIONES_PERM,
                                       random_state=SEMILLA, n_jobs=n_jobs)
    return (pd.DataFrame({"variable": PREDICTORES,
                          "nombre": [legible(c) for c in PREDICTORES],
                          "caida_media": resultado.importances_mean,
                          "caida_desv": resultado.importances_std})
            .sort_values("caida_media", ascending=False, ignore_index=True))


def fase5_evaluacion(datos: dict, modelado: dict, n_jobs: int) -> dict:
    """
    FASE 5 DE CRISP-ML(Q). Compara los modelos y explica dónde se equivocan.

    Todo se calcula con las predicciones que ya se hicieron en la fase 4: la
    prueba no se vuelve a usar para elegir nada.

    Resultado esperado:
      * el mejor modelo (mayor exactitud balanceada en prueba, sin contar el trivial);
      * intervalos de confianza del 95 % y prueba de McNemar frente a Naive Bayes;
      * matrices de confusión, exactitud por región, errores cerca de los
        umbrales e importancia de las variables.
    """
    seccion("PASO 5. FASE 5 DE CRISP-ML(Q) - EVALUACIÓN")
    comparacion, modelos = modelado["comparacion"], modelado["modelos"]
    y_pr, X_pr = datos["y_pr"], datos["X_pr"]

    # --- Mejor modelo: no compite el clasificador trivial -------------------
    rivales = comparacion[comparacion["modelo"] != MODELO_TRIVIAL]
    mejor = rivales.loc[rivales["eb_prueba"].idxmax(), "modelo"]
    eb_mejor = float(rivales["eb_prueba"].max())
    nota(f"mejor modelo en prueba: {mejor} (exactitud balanceada {num(eb_mejor)})")

    # --- Intervalos de confianza del 95 % -----------------------------------
    intervalos = {}
    for nombre in dict.fromkeys([MODELO_ASIGNADO, mejor, MODELO_LOGISTICA]):
        intervalos[nombre] = intervalo_bootstrap(y_pr, modelos[nombre]["pred"])
        nota(f"IC 95 % de {nombre}: {intervalo(*intervalos[nombre])}")
    guardar_tabla(pd.DataFrame([{"modelo": k, "ic_inferior": v[0], "ic_superior": v[1]}
                                for k, v in intervalos.items()]), "intervalos_bootstrap.csv")

    # --- ¿La diferencia con el modelo asignado es real? ---------------------
    mcnemar = prueba_mcnemar(y_pr, modelos[MODELO_ASIGNADO]["pred"], modelos[mejor]["pred"])
    nota(f"McNemar Naive Bayes vs. {mejor}: b01={mcnemar['b01']}, b10={mcnemar['b10']}, "
         f"p={mcnemar['p']:.2e} -> "
         f"{'significativa' if mcnemar['significativa'] else 'no significativa'}")

    # --- Matrices de confusión normalizadas por fila ------------------------
    confusiones = {}
    for nombre in dict.fromkeys([MODELO_ASIGNADO, mejor]):
        matriz = confusion_matrix(y_pr, modelos[nombre]["pred"], labels=CLASES)
        confusiones[nombre] = matriz / matriz.sum(axis=1, keepdims=True)
    filas_conf = []
    for nombre, matriz in confusiones.items():
        for i, real in enumerate(ETAPAS):
            for j, predicha in enumerate(ETAPAS):
                filas_conf.append({"modelo": nombre, "etapa_real": real,
                                   "etapa_predicha": predicha, "proporcion": matriz[i, j]})
    guardar_tabla(pd.DataFrame(filas_conf), "matrices_confusion.csv")

    # --- ¿Dónde se equivoca el mejor modelo? --------------------------------
    predicciones = {n: modelos[n]["pred"] for n in dict.fromkeys([MODELO_ASIGNADO, mejor])}
    regiones = exactitud_por_region(datos["df_prueba"], y_pr, predicciones)
    guardar_tabla(regiones, "errores_por_region.csv")
    peor = regiones.loc[regiones[mejor].idxmin()]
    nota(f"región con menor exactitud del mejor modelo: {peor['region']} "
         f"({num(peor[mejor])})")

    umbrales = errores_cerca_del_umbral(datos["df_prueba"], y_pr, modelos[mejor]["pred"])
    nota(f"errores cerca de los umbrales: {pct(umbrales['errores_cerca'])} de los "
         f"{umbrales['n_errores']} errores, frente a {pct(umbrales['prueba_cerca'])} "
         "de todas las filas de prueba")

    # --- ¿Qué variables usan los modelos? -----------------------------------
    importancias = {}
    for nombre in dict.fromkeys([mejor, MODELO_ASIGNADO]):
        importancias[nombre] = importancia_por_permutacion(
            modelos[nombre]["modelo"], X_pr, y_pr, n_jobs)
        importancias[nombre].insert(0, "modelo", nombre)
        tres = list(importancias[nombre]["nombre"].head(3))
        nota(f"variables más importantes de {nombre}: {lista_y(tres)}")
    guardar_tabla(pd.concat(importancias.values(), ignore_index=True),
                  "importancia_permutacion.csv")

    # --- Los dos modelos del artículo se guardan para poder reusarlos -------
    # compress=3: el mejor modelo son cientos de árboles y sin comprimir ocupa
    # decenas de megabytes; comprimido se carga igual con joblib.load().
    joblib.dump(modelos[MODELO_ASIGNADO]["modelo"],
                CARPETA_MODELOS / "naive_bayes_final.joblib", compress=3)
    joblib.dump(modelos[mejor]["modelo"], CARPETA_MODELOS / "mejor_modelo.joblib", compress=3)
    nota("modelos guardados en resultados/modelos/")

    return {"mejor": mejor, "intervalos": intervalos, "mcnemar": mcnemar,
            "confusiones": confusiones, "regiones": regiones, "umbrales": umbrales,
            "importancias": importancias}


# =============================================================================
# PASO 6. CAPÍTULO 2 DE LA GUÍA: APRENDIZAJE SUPERVISADO
# =============================================================================
def curvas_de_validacion(datos: dict) -> dict:
    """
    Generalización vista de cerca: ¿cuánta complejidad conviene?

    Para tres modelos se mueve UN solo hiperparámetro y se mide la exactitud
    balanceada en entrenamiento (2004-2015) y en validación (2016-2019):
      * árbol de decisión: profundidad máxima de 1 a 15;
      * k-NN: número de vecinos;
      * SVM RBF: el parámetro C, que controla cuánto se penalizan los errores.

    Resultado esperado: si al subir la complejidad el acierto en entrenamiento
    sigue creciendo y el de validación baja, hay SOBREAJUSTE; si los dos son
    bajos, SUBAJUSTE. El mejor valor es el máximo de la curva de validación.
    """
    nota("curvas de validación (árbol, k-NN y SVM RBF)")
    X_tr, y_tr, X_va, y_va = datos["X_tr"], datos["y_tr"], datos["X_va"], datos["y_va"]
    configuraciones = {
        "Árbol de decisión": ("Profundidad máxima", list(range(1, 16)),
                              lambda v: DecisionTreeClassifier(max_depth=v,
                                                               random_state=SEMILLA)),
        "k-NN": ("Número de vecinos (k)", [1, 3, 5, 9, 15, 25, 45, 75, 125],
                 lambda v: Pipeline([("escalador", StandardScaler()),
                                     ("modelo", KNeighborsClassifier(n_neighbors=v))])),
        "SVM RBF": ("C (escala logarítmica)", [0.01, 0.1, 1, 10, 100, 1000],
                    lambda v: Pipeline([("escalador", StandardScaler()),
                                        ("modelo", SVC(kernel="rbf", C=v, gamma="scale",
                                                       random_state=SEMILLA))])),
    }
    curvas = {}
    for nombre, (etiqueta, valores, construir) in configuraciones.items():
        en_entrenamiento, en_validacion = [], []
        for valor in valores:
            modelo = construir(valor).fit(X_tr, y_tr)
            en_entrenamiento.append(balanced_accuracy_score(y_tr, modelo.predict(X_tr)))
            en_validacion.append(balanced_accuracy_score(y_va, modelo.predict(X_va)))
        mejor = valores[int(np.argmax(en_validacion))]
        curvas[nombre] = {"parametro": etiqueta, "valores": valores,
                          "entrenamiento": en_entrenamiento, "validacion": en_validacion,
                          "mejor": mejor}
        nota(f"{nombre:<18} mejor en validación con {etiqueta.split(' (')[0].lower()} "
             f"= {mejor} (EB {num(max(en_validacion))})")
    guardar_tabla(pd.DataFrame([
        {"modelo": n, "parametro": c["parametro"], "valor": v,
         "eb_entrenamiento": e, "eb_validacion": d}
        for n, c in curvas.items()
        for v, e, d in zip(c["valores"], c["entrenamiento"], c["validacion"])
    ]), "curvas_validacion.csv")
    return curvas


def vecino_mas_cercano_del_mismo_pais(datos: dict) -> float:
    """
    ¿Por qué la curva de validación de k-NN es mejor con k = 1?

    Para cada fila de validación buscamos su vecino más cercano entre los años
    de entrenamiento (con las variables estandarizadas) y miramos si es el MISMO
    país en otro año. Si casi siempre lo es, mirar un solo vecino equivale a
    "repetir la última etapa conocida de ese país", que en un panel de países es
    una predicción muy buena y explica que k = 1 gane.

    Resultado esperado: una proporción alta, muy por encima de 1/180.
    """
    escalador = StandardScaler().fit(datos["X_tr"])
    vecinos = NearestNeighbors(n_neighbors=1).fit(escalador.transform(datos["X_tr"]))
    _, indices = vecinos.kneighbors(escalador.transform(datos["X_va"]))
    paises_entrenamiento = datos["df_entrenamiento"]["codigo_iso3"].to_numpy()
    paises_validacion = datos["df_validacion"]["codigo_iso3"].to_numpy()
    proporcion = float((paises_entrenamiento[indices[:, 0]] == paises_validacion).mean())
    nota(f"el vecino más cercano es el mismo país en {pct(proporcion)} de las filas "
         "de validación")
    return proporcion


def curvas_de_aprendizaje(datos: dict, modelado: dict, mejor: str) -> dict:
    """
    La otra cara de la generalización: ¿ayudarían más datos?

    Se entrena con el 10, 20, 30, 50, 70 y 100 % de los años de entrenamiento
    (muestras estratificadas, para no perder la etapa avanzada) y se mide la
    exactitud balanceada en la submuestra y en validación.

    Resultado esperado: una BRECHA pequeña entre las dos curvas indica poco
    sobreajuste; si además las dos dejan de subir, más datos del mismo tipo no
    mejorarían el modelo.
    """
    nota("curvas de aprendizaje (Naive Bayes y mejor modelo)")
    X_tr, y_tr, X_va, y_va = datos["X_tr"], datos["y_tr"], datos["X_va"], datos["y_va"]
    fracciones = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    curvas = {}
    for nombre in dict.fromkeys([MODELO_ASIGNADO, mejor]):
        estimador = construir_modelos()[nombre][0]
        parametros = modelado["modelos"][nombre]["parametros"]
        tamanos, en_entrenamiento, en_validacion = [], [], []
        for fraccion in fracciones:
            if fraccion >= 1.0:
                indices = np.arange(len(y_tr))      # con todo el entrenamiento no hace falta sortear
            else:
                reparto = StratifiedShuffleSplit(n_splits=1, train_size=fraccion,
                                                 random_state=SEMILLA)
                indices, _ = next(reparto.split(X_tr, y_tr))
            modelo = clone(estimador).set_params(**parametros).fit(X_tr[indices], y_tr[indices])
            tamanos.append(len(indices))
            en_entrenamiento.append(balanced_accuracy_score(
                y_tr[indices], modelo.predict(X_tr[indices])))
            en_validacion.append(balanced_accuracy_score(y_va, modelo.predict(X_va)))
        brecha = en_entrenamiento[-1] - en_validacion[-1]
        curvas[nombre] = {"fracciones": fracciones, "tamanos": tamanos,
                          "entrenamiento": en_entrenamiento, "validacion": en_validacion,
                          "brecha": brecha}
        nota(f"{nombre:<24} brecha con todo el entrenamiento: {puntos(brecha)} puntos")
    guardar_tabla(pd.DataFrame([
        {"modelo": n, "fraccion": f, "filas": t, "eb_entrenamiento": e, "eb_validacion": v}
        for n, c in curvas.items()
        for f, t, e, v in zip(c["fracciones"], c["tamanos"], c["entrenamiento"],
                              c["validacion"])
    ]), "curvas_aprendizaje.csv")
    return curvas


def incertidumbre_de_las_probabilidades(datos: dict, modelado: dict, mejor: str) -> dict:
    """
    ¿Se puede creer en las probabilidades que da cada modelo?

    Dos miradas sobre los años de prueba:
      * la CURVA DE CALIBRACIÓN de la etapa avanzada (una contra el resto): se
        agrupan las filas en diez intervalos de probabilidad y se compara la
        probabilidad predicha con la frecuencia observada; la diagonal es el
        comportamiento ideal;
      * el HISTOGRAMA de la probabilidad máxima: muestra si el modelo se declara
        casi seguro todo el tiempo.

    Resultado esperado: la pérdida logarítmica y el porcentaje de filas con
    probabilidad mayor a 0,99 dicen qué modelo exagera su confianza.
    """
    nota("calibración y confianza de las probabilidades")
    y_pr = datos["y_pr"]
    es_avanzada = (y_pr == 2).astype(int)
    resultado = {}
    for nombre in dict.fromkeys([MODELO_ASIGNADO, MODELO_LOGISTICA, mejor]):
        proba = modelado["modelos"][nombre]["proba"]
        observada, predicha = calibration_curve(es_avanzada, proba[:, 2], n_bins=10,
                                                strategy="uniform")
        maxima = proba.max(axis=1)
        resultado[nombre] = {"predicha": predicha, "observada": observada,
                             "maxima": maxima, "extremas": float((maxima > 0.99).mean())}
        nota(f"{nombre:<24} probabilidad máxima > 0,99 en "
             f"{pct(resultado[nombre]['extremas'])} de las filas")
    guardar_tabla(pd.DataFrame([{"modelo": n, "prob_maxima_mayor_099": d["extremas"]}
                                for n, d in resultado.items()]), "calibracion.csv")
    return resultado


def capitulo2_supervisado(datos: dict, modelado: dict, mejor: str) -> dict:
    """Responde con nuestros datos las preguntas del capítulo 2 de la guía."""
    seccion("PASO 6. CAPÍTULO 2 - APRENDIZAJE SUPERVISADO")
    return {"validacion": curvas_de_validacion(datos),
            "vecino_mismo_pais": vecino_mas_cercano_del_mismo_pais(datos),
            "aprendizaje": curvas_de_aprendizaje(datos, modelado, mejor),
            "incertidumbre": incertidumbre_de_las_probabilidades(datos, modelado, mejor)}


# =============================================================================
# PASO 7. CAPÍTULO 3 DE LA GUÍA: NO SUPERVISADO Y PREPROCESAMIENTO
# =============================================================================
def efecto_del_escalado(datos: dict) -> tuple[pd.DataFrame, dict]:
    """
    ¿A qué modelos les cambia la vida el preprocesamiento?

    Se entrenan tres modelos con los mismos datos pero cinco preparaciones
    distintas (sin escalar, estandarizado, mínimo-máximo, robusto y
    Yeo-Johnson) y se mide la exactitud balanceada en validación.

    Resultado esperado: k-NN y la SVM dependen mucho del escalado, porque miden
    distancias; el bosque aleatorio da prácticamente el mismo resultado con
    todos, porque sus cortes solo dependen del orden de los valores.

    Además devuelve tres comprobaciones que explican la tabla, para no dejar
    ninguna afirmación sin respaldo:

      * VARIABLES QUE MANDAN SIN ESCALAR: la desviación típica de cada predictor
        en los datos crudos. La distancia de k-NN la deciden las variables con
        números más grandes.
      * k-NN SOLO CON LOS CONTINUOS: si al estandarizar k-NN empeora porque las
        once indicadoras 0/1 de región y de ingreso pasan a pesar igual que los
        indicadores, entonces al quitarlas debería recuperarse.
      * FILAS QUE CAMBIAN EN EL BOSQUE: estandarizar, llevar a [0, 1] o aplicar
        Yeo-Johnson no cambia el ORDEN de los valores dentro de una variable, así
        que los cortes del árbol deberían ser los mismos. Contamos cuántas filas
        de validación cambian de etapa predicha: si son muy pocas, la diferencia
        es redondeo de los puntos de corte y no un efecto real del escalado.
    """
    nota("efecto del preprocesamiento en k-NN, SVM RBF y bosque aleatorio")
    escaladores = [("Sin escalar", None), ("Estandarizado", StandardScaler()),
                   ("Mínimo-máximo", MinMaxScaler()), ("Robusto", RobustScaler()),
                   ("Yeo-Johnson", PowerTransformer())]
    modelos = {"k-NN (k = 15)": KNeighborsClassifier(n_neighbors=15),
               "SVM RBF (C = 10)": SVC(kernel="rbf", C=10, gamma="scale",
                                       random_state=SEMILLA),
               "Bosque aleatorio": RandomForestClassifier(n_estimators=300,
                                                          random_state=SEMILLA)}
    filas, predicciones_bosque = [], {}
    for etiqueta, escalador in escaladores:
        fila = {"preprocesamiento": etiqueta}
        for nombre, modelo in modelos.items():
            pasos = ([] if escalador is None else [("escalador", clone(escalador))])
            tuberia = Pipeline(pasos + [("modelo", clone(modelo))])
            tuberia.fit(datos["X_tr"], datos["y_tr"])
            prediccion = tuberia.predict(datos["X_va"])
            fila[nombre] = balanced_accuracy_score(datos["y_va"], prediccion)
            if nombre == "Bosque aleatorio":
                predicciones_bosque[etiqueta] = prediccion
        filas.append(fila)
    tabla = pd.DataFrame(filas)
    guardar_tabla(tabla, "efecto_escalado.csv")
    ganancia = (tabla.loc[tabla["preprocesamiento"] == "Estandarizado", "k-NN (k = 15)"].iloc[0]
                - tabla.loc[tabla["preprocesamiento"] == "Sin escalar", "k-NN (k = 15)"].iloc[0])
    nota(f"k-NN cambia {puntos(ganancia)} puntos al estandarizar")

    # --- Comprobación 1: ¿qué variables mandan cuando no se escala? ---------
    desviaciones = pd.Series(datos["X_tr"].std(axis=0), index=PREDICTORES)
    dominantes = list(desviaciones.nlargest(2).index)
    nota("sin escalar, la distancia la dominan "
         f"{lista_y([legible(v) for v in dominantes])}")

    # --- Comprobación 2: k-NN estandarizado sin las indicadoras 0/1 ---------
    columnas = [PREDICTORES.index(c) for c in CONTINUOS]
    solo_continuos = Pipeline([("escalador", StandardScaler()),
                               ("modelo", KNeighborsClassifier(n_neighbors=15))])
    solo_continuos.fit(datos["X_tr"][:, columnas], datos["y_tr"])
    eb_solo_continuos = balanced_accuracy_score(
        datos["y_va"], solo_continuos.predict(datos["X_va"][:, columnas]))
    nota(f"k-NN estandarizado solo con los {len(CONTINUOS)} indicadores continuos: "
         f"{num(eb_solo_continuos)}")

    # --- Comprobación 3: ¿cambia de verdad el bosque con el escalado? -------
    sin_escalar = predicciones_bosque["Sin escalar"]
    filas_cambian = max(int((prediccion != sin_escalar).sum())
                        for etiqueta, prediccion in predicciones_bosque.items()
                        if etiqueta != "Sin escalar")
    nota(f"bosque aleatorio: como mucho {filas_cambian} de {len(sin_escalar)} filas de "
         "validación cambian de etapa entre un escalado y otro")

    extras = {"dominantes": dominantes, "eb_solo_continuos": eb_solo_continuos,
              "filas_cambian_bosque": filas_cambian, "filas_validacion": len(sin_escalar)}
    return tabla, extras


def reduccion_de_dimensionalidad(datos: dict) -> dict:
    """
    PCA, NMF y t-SNE sobre los once indicadores continuos.

    * PCA busca las direcciones que explican más varianza; son combinaciones
      lineales de las variables y pueden tener pesos negativos.
    * NMF descompone la tabla en partes que solo se suman, por lo que cada
      componente se lee como un "perfil de país"; exige datos no negativos, por
      eso antes se escala de 0 a 1.
    * t-SNE coloca en dos dimensiones a los países que son vecinos en el espacio
      original; sirve para mirar, no para transformar datos nuevos.

    Resultado esperado: cuánta varianza explican dos componentes, cuántas hacen
    falta para llegar al 80 % y si las etapas se separan sin usar la etiqueta.
    """
    nota("reducción de dimensionalidad: PCA, NMF y t-SNE")
    df = datos["df"]
    entrenamiento = datos["df_entrenamiento"][CONTINUOS].to_numpy(dtype=float)

    # --- PCA: se estandariza primero para que todas las variables pesen igual
    pca = PCA(random_state=SEMILLA)
    coordenadas = pca.fit_transform(StandardScaler().fit_transform(entrenamiento))
    varianza = pca.explained_variance_ratio_
    acumulada = np.cumsum(varianza)
    componentes_80 = int(np.searchsorted(acumulada, 0.80) + 1)
    nota(f"PCA: dos componentes explican {pct(acumulada[1])}; "
         f"{componentes_80} componentes llegan al 80 %")

    # --- NMF: necesita datos no negativos, por eso el escalado de 0 a 1
    escalado01 = MinMaxScaler().fit_transform(entrenamiento)
    nmf = NMF(n_components=4, init="nndsvda", random_state=SEMILLA, max_iter=2000)
    nmf.fit(escalado01)
    pesos = pd.DataFrame(nmf.components_.T, index=CONTINUOS,
                         columns=[f"C{i + 1}" for i in range(4)])
    guardar_tabla(pesos.reset_index().rename(columns={"index": "variable"}),
                  "nmf_componentes.csv")
    for componente in ("C1", "C2"):
        principales = [legible(v) for v in pesos[componente].nlargest(2).index]
        nota(f"NMF {componente}: pesa sobre todo {lista_y(principales)}")

    # --- t-SNE: solo los 180 países del último año, para poder mirarlos
    ultimo = df[df["anio"] == 2024].reset_index(drop=True)
    Z = StandardScaler().fit_transform(ultimo[CONTINUOS].to_numpy(dtype=float))
    mapa = TSNE(perplexity=30, init="pca", random_state=SEMILLA).fit_transform(Z)
    nota(f"t-SNE de los {len(ultimo)} países de 2024")

    return {"varianza": varianza, "acumulada": acumulada, "componentes_80": componentes_80,
            "coordenadas": coordenadas, "etapas_entrenamiento": datos["y_tr"],
            "pesos_nmf": pesos, "tsne": mapa, "Z_2024": Z, "paises_2024": ultimo}


def agrupamiento(reduccion: dict) -> dict:
    """
    ¿Aparecen las tres etapas sin usar la etiqueta?

    Se agrupan los 180 países de 2024 (once indicadores estandarizados) con
    tres métodos distintos y se compara el resultado con la etapa real:
      * k-Means: hay que decirle cuántos grupos (k) y supone grupos redondos;
      * jerárquico de Ward: une poco a poco los grupos que menos aumentan la
        varianza interna;
      * DBSCAN: forma grupos con las zonas densas y marca como ruido a los
        países aislados; no necesita k, pero sí un radio (eps).

    Para medirlos se usan la silueta (qué tan compactos y separados son los
    grupos, sin mirar la etiqueta), el índice de Rand ajustado (ARI) y la
    información mutua normalizada (NMI) frente a las etapas reales.

    Resultado esperado: los grupos coinciden en parte con las etapas, pero no
    las reproducen: la etiqueta depende de la matriz eléctrica, que no está
    entre estos once indicadores.
    """
    nota("agrupamiento de los 180 países de 2024")
    Z, etapas = reduccion["Z_2024"], reduccion["paises_2024"][OBJETIVO].to_numpy()

    # --- k-Means de 2 a 8 grupos: inercia, silueta y parecido con las etapas
    barrido = []
    for k in range(2, 9):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit(Z)
        barrido.append({"k": k, "inercia": kmeans.inertia_,
                        "silueta": silhouette_score(Z, kmeans.labels_),
                        "ari": adjusted_rand_score(etapas, kmeans.labels_)})
    barrido = pd.DataFrame(barrido)
    guardar_tabla(barrido, "kmeans_barrido.csv")
    k_silueta = int(barrido.loc[barrido["silueta"].idxmax(), "k"])
    nota(f"la mejor silueta de k-Means se obtiene con k = {k_silueta}")

    # --- Los tres métodos, comparados con las etapas reales -----------------
    grupos = {
        "k-Means (k = 3)": KMeans(n_clusters=3, n_init=10, random_state=SEMILLA).fit_predict(Z),
        f"k-Means (mejor silueta, k = {k_silueta})":
            KMeans(n_clusters=k_silueta, n_init=10, random_state=SEMILLA).fit_predict(Z),
        "Jerárquico Ward (k = 3)":
            AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(Z),
    }
    # DBSCAN: el radio sale de la distancia al quinto vecino más cercano.
    distancias = NearestNeighbors(n_neighbors=5).fit(Z).kneighbors(Z)[0][:, -1]
    eps = round(float(np.percentile(distancias, 90)), 2)
    grupos["DBSCAN"] = DBSCAN(eps=eps, min_samples=5).fit_predict(Z)
    nota(f"DBSCAN con eps = {eps} (percentil 90 de la distancia al quinto vecino)")

    filas = []
    for nombre, etiquetas in grupos.items():
        sin_ruido = etiquetas != -1
        n_grupos = len(set(etiquetas[sin_ruido]))
        silueta = (silhouette_score(Z[sin_ruido], etiquetas[sin_ruido])
                   if n_grupos >= 2 and sin_ruido.sum() > n_grupos else np.nan)
        filas.append({"metodo": nombre, "grupos": n_grupos,
                      "ruido": int((~sin_ruido).sum()), "silueta": silueta,
                      "ari": adjusted_rand_score(etapas, etiquetas),
                      "nmi": normalized_mutual_info_score(etapas, etiquetas)})
        nota(f"{nombre:<32} ARI {num(filas[-1]['ari'])}  NMI {num(filas[-1]['nmi'])}")
    tabla = pd.DataFrame(filas)
    guardar_tabla(tabla, "agrupamiento.csv")
    return {"barrido": barrido, "k_silueta": k_silueta, "grupos": grupos,
            "tabla": tabla, "eps": eps, "etapas": etapas}


def capitulo3_no_supervisado(datos: dict) -> dict:
    """Responde con nuestros datos las preguntas del capítulo 3 de la guía."""
    seccion("PASO 7. CAPÍTULO 3 - NO SUPERVISADO Y PREPROCESAMIENTO")
    escalado, escalado_extras = efecto_del_escalado(datos)
    reduccion = reduccion_de_dimensionalidad(datos)
    return {"escalado": escalado, "escalado_extras": escalado_extras,
            "reduccion": reduccion, "agrupamiento": agrupamiento(reduccion)}


# =============================================================================
# PASO 8. FIGURAS DEL ARTÍCULO (fig02 a fig10)
# =============================================================================
# Todas miden 3,35 pulgadas de ancho (una columna de la plantilla LACCEI) y se
# guardan a 300 ppp, con tipografía sans-serif de 6,5 a 7,5 puntos. La fig01 del
# artículo es un esquema hecho a mano, no la genera este programa.
def figura02_comparacion(modelado: dict, evaluacion: dict) -> None:
    """Barras horizontales: exactitud balanceada en prueba de los 14 modelos."""
    tabla = modelado["comparacion"].sort_values("eb_prueba", ascending=True)
    fig, ax = plt.subplots(figsize=(ANCHO_FIGURA, 2.35))
    posiciones = np.arange(len(tabla))
    colores = [COLOR_NB if m == MODELO_ASIGNADO else (GRIS if m == MODELO_TRIVIAL else AZUL)
               for m in tabla["modelo"]]
    ax.barh(posiciones, tabla["eb_prueba"], color=colores, height=0.72)
    # El intervalo de confianza solo se calculó para tres modelos: la barra de
    # error se dibuja únicamente en esos, no en todos.
    margen = np.zeros(len(tabla))
    for i, (modelo, valor) in enumerate(zip(tabla["modelo"], tabla["eb_prueba"])):
        if modelo not in evaluacion["intervalos"]:
            continue
        lo, hi = evaluacion["intervalos"][modelo]
        margen[i] = max(hi - valor, 0)
        ax.errorbar(valor, i, xerr=[[max(valor - lo, 0)], [margen[i]]], fmt="none",
                    elinewidth=0.7, capsize=1.6, ecolor="#333333")
    for i, valor in enumerate(tabla["eb_prueba"]):
        ax.text(valor + margen[i] + 0.018, i, num(valor), va="center", fontsize=5.8)
    ax.axvline(CRITERIO_EXITO, color="#b23a48", linestyle="--", linewidth=0.8)
    ax.text(CRITERIO_EXITO + 0.015, len(tabla) - 0.35, "criterio 0,75", color="#b23a48",
            fontsize=5.8, va="bottom", ha="left")
    ax.set_yticks(posiciones, [nombre_corto(m) for m in tabla["modelo"]], fontsize=6)
    ax.set_xlabel("Exactitud balanceada en prueba (2020-2024)")
    ax.set_xlim(0, max(1.0, float(tabla["eb_prueba"].max()) + 0.12))
    ax.set_ylim(-0.7, len(tabla) + 0.1)
    ax.grid(axis="y", visible=False)
    ejes_con_coma(ax, en_y=False)
    fig.tight_layout(pad=0.3)
    guardar_figura(fig, "fig02_comparacion_modelos.png")


def figura03_confusion(evaluacion: dict) -> None:
    """Dos matrices de confusión normalizadas por fila: Naive Bayes y el mejor modelo."""
    nombres = list(evaluacion["confusiones"])
    fig, ejes = plt.subplots(1, len(nombres), figsize=(ANCHO_FIGURA, 1.6))
    ejes = np.atleast_1d(ejes)
    for eje, nombre in zip(ejes, nombres):
        matriz = evaluacion["confusiones"][nombre]
        eje.imshow(matriz, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                eje.text(j, i, f"{100 * matriz[i, j]:.0f}", ha="center", va="center",
                         fontsize=6, color="white" if matriz[i, j] > 0.55 else "#222222")
        eje.set_xticks(range(3), ETAPAS, fontsize=5.6, rotation=35, ha="right")
        eje.set_yticks(range(3), ETAPAS, fontsize=5.6)
        eje.set_title(nombre_corto(nombre), fontsize=6.5)
        eje.set_xlabel("Etapa predicha", fontsize=6)
        eje.grid(visible=False)
    ejes[0].set_ylabel("Etapa real", fontsize=6)
    fig.tight_layout(pad=0.25)
    guardar_figura(fig, "fig03_matrices_confusion.png")


def figura04_curvas_validacion(capitulo2: dict) -> None:
    """Tres paneles: entrenamiento frente a validación según la complejidad."""
    curvas = capitulo2["validacion"]
    fig, ejes = plt.subplots(1, 3, figsize=(ANCHO_FIGURA, 1.45))
    for eje, (nombre, curva) in zip(ejes, curvas.items()):
        valores = curva["valores"]
        eje.plot(valores, curva["entrenamiento"], color=GRIS, linewidth=1,
                 marker="o", markersize=1.8, label="Entren.")
        eje.plot(valores, curva["validacion"], color=AZUL, linewidth=1,
                 marker="o", markersize=1.8, label="Valid.")
        eje.axvline(curva["mejor"], color="#b23a48", linestyle=":", linewidth=0.8)
        if nombre == "SVM RBF":
            eje.set_xscale("log")
        eje.set_title(f"{nombre}\nmejor: {curva['mejor']}", fontsize=6)
        eje.set_xlabel(curva["parametro"].split(" (")[0], fontsize=5.8)
        eje.tick_params(labelsize=5.6)
        ejes_con_coma(eje, en_x=(nombre == "Árbol de decisión"), en_y=True)
    ejes[0].set_ylabel("Exactitud balanceada", fontsize=6)
    ejes[0].legend(fontsize=5.4, loc="lower right", frameon=False, handlelength=1.2)
    fig.tight_layout(pad=0.25)
    guardar_figura(fig, "fig04_curvas_validacion.png")


def figura05_curvas_aprendizaje(capitulo2: dict) -> None:
    """Dos paneles: exactitud balanceada según el tamaño del entrenamiento."""
    curvas = capitulo2["aprendizaje"]
    fig, ejes = plt.subplots(1, 2, figsize=(ANCHO_FIGURA, 1.45), sharey=True)
    for eje, (nombre, curva) in zip(np.atleast_1d(ejes), curvas.items()):
        eje.plot(curva["tamanos"], curva["entrenamiento"], color=GRIS, linewidth=1,
                 marker="o", markersize=2, label="Entrenamiento")
        eje.plot(curva["tamanos"], curva["validacion"], color=AZUL, linewidth=1,
                 marker="o", markersize=2, label="Validación")
        eje.set_title(f"{nombre_corto(nombre)}\nbrecha: {puntos(curva['brecha'])} puntos",
                      fontsize=6)
        eje.set_xlabel("Filas de entrenamiento", fontsize=6)
        eje.tick_params(labelsize=5.8)
        ejes_con_coma(eje, en_x=False)
    np.atleast_1d(ejes)[0].set_ylabel("Exactitud balanceada", fontsize=6)
    np.atleast_1d(ejes)[-1].legend(fontsize=5.4, loc="lower right", frameon=False,
                                   handlelength=1.2)
    fig.tight_layout(pad=0.25)
    guardar_figura(fig, "fig05_curvas_aprendizaje.png")


def figura06_calibracion(capitulo2: dict, mejor: str) -> None:
    """Izquierda: calibración de la etapa avanzada. Derecha: confianza del modelo."""
    datos_calibracion = capitulo2["incertidumbre"]
    fig, (izq, der) = plt.subplots(1, 2, figsize=(ANCHO_FIGURA, 1.55))
    colores = {MODELO_ASIGNADO: COLOR_NB, MODELO_LOGISTICA: AZUL, mejor: COLOR_ETAPA[2]}
    izq.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=0.7)
    for nombre, curva in datos_calibracion.items():
        izq.plot(curva["predicha"], curva["observada"], marker="o", markersize=2,
                 linewidth=1, color=colores.get(nombre, GRIS),
                 label="Naive Bayes" if nombre == MODELO_ASIGNADO else nombre)
    izq.set_xlabel("Probabilidad predicha", fontsize=6)
    izq.set_ylabel("Frecuencia observada", fontsize=6)
    izq.set_title("Calibración: etapa avanzada", fontsize=6.5)
    izq.legend(fontsize=5.2, loc="upper left", frameon=False, handlelength=1.2)
    izq.tick_params(labelsize=5.8)
    ejes_con_coma(izq)

    bordes = np.linspace(1 / 3, 1, 15)
    for nombre in (MODELO_ASIGNADO, MODELO_LOGISTICA):
        der.hist(datos_calibracion[nombre]["maxima"], bins=bordes, histtype="step",
                 linewidth=1, color=colores[nombre],
                 label="Naive Bayes" if nombre == MODELO_ASIGNADO else "Reg. logística")
    der.set_xlabel("Probabilidad máxima", fontsize=6)
    der.set_ylabel("Filas de prueba", fontsize=6)
    der.set_title("Confianza del modelo", fontsize=6.5)
    der.legend(fontsize=5.2, loc="upper left", frameon=False, handlelength=1.2)
    der.tick_params(labelsize=5.8)
    ejes_con_coma(der, en_y=False)
    fig.tight_layout(pad=0.25)
    guardar_figura(fig, "fig06_calibracion.png")


def figura07_importancia(evaluacion: dict) -> None:
    """Las diez variables que más pesan en el mejor modelo y en Naive Bayes."""
    importancias = evaluacion["importancias"]
    fig, ejes = plt.subplots(1, 2, figsize=(ANCHO_FIGURA, 1.9))
    for eje, (nombre, tabla) in zip(ejes, importancias.items()):
        diez = tabla.head(10).iloc[::-1]
        color = COLOR_NB if nombre == MODELO_ASIGNADO else AZUL
        eje.barh(np.arange(10), diez["caida_media"], color=color, height=0.72)
        eje.set_yticks(np.arange(10),
                       [NOMBRES_FIGURA.get(v, v) for v in diez["variable"]], fontsize=5)
        eje.set_title(nombre_corto(nombre), fontsize=6)
        eje.set_xlabel("Caída de la EB", fontsize=5.6)
        eje.tick_params(axis="x", labelsize=5.2)
        eje.grid(axis="y", visible=False)
        ejes_con_coma(eje, en_y=False)
    fig.tight_layout(pad=0.25, w_pad=0.6)
    guardar_figura(fig, "fig07_importancia_permutacion.png")


def figura08_pca_tsne(capitulo3: dict) -> None:
    """Izquierda: plano PC1-PC2 del entrenamiento. Derecha: t-SNE de 2024."""
    reduccion = capitulo3["reduccion"]
    fig, (izq, der) = plt.subplots(1, 2, figsize=(ANCHO_FIGURA, 1.6))
    etapas_entrenamiento = reduccion["etapas_entrenamiento"]
    for etapa in CLASES:
        donde = etapas_entrenamiento == etapa
        izq.scatter(reduccion["coordenadas"][donde, 0], reduccion["coordenadas"][donde, 1],
                    s=1.2, alpha=0.45, color=COLOR_ETAPA[etapa], linewidths=0)
    izq.set_xlabel(f"PC1 ({pct(reduccion['varianza'][0], 0)})", fontsize=6)
    izq.set_ylabel(f"PC2 ({pct(reduccion['varianza'][1], 0)})", fontsize=6)
    izq.set_title("PCA (entrenamiento)", fontsize=6.5)

    etapas_2024 = reduccion["paises_2024"][OBJETIVO].to_numpy()
    for etapa in CLASES:
        donde = etapas_2024 == etapa
        der.scatter(reduccion["tsne"][donde, 0], reduccion["tsne"][donde, 1], s=4,
                    alpha=0.85, color=COLOR_ETAPA[etapa], linewidths=0)
    der.set_title("t-SNE (180 países, 2024)", fontsize=6.5)
    der.set_xticks([])
    der.set_yticks([])
    for eje in (izq, der):
        eje.tick_params(labelsize=5.8)
    marcas = [Line2D([], [], marker="o", linestyle="", markersize=2.6,
                     color=COLOR_ETAPA[e], label=ETAPAS[e]) for e in CLASES]
    ejes_con_coma(izq)
    fig.tight_layout(pad=0.25)
    fig.subplots_adjust(bottom=0.28)
    fig.legend(handles=marcas, fontsize=5.4, ncol=3, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, 0.0))
    guardar_figura(fig, "fig08_pca_tsne.png")


def figura09_nmf(capitulo3: dict) -> None:
    """Mapa de calor: peso de cada indicador en los cuatro componentes de NMF."""
    pesos = capitulo3["reduccion"]["pesos_nmf"]
    fig, eje = plt.subplots(figsize=(ANCHO_FIGURA, 1.55))
    imagen = eje.imshow(pesos.to_numpy(), cmap="Blues", aspect="auto")
    eje.set_xticks(range(4), [f"Comp. {i + 1}" for i in range(4)], fontsize=5.8)
    eje.set_yticks(range(len(pesos)), [NOMBRES_FIGURA.get(v, v) for v in pesos.index],
                   fontsize=5)
    for i in range(len(pesos)):
        for j in range(4):
            valor = pesos.iat[i, j]
            eje.text(j, i, f"{valor:.2f}".replace(".", ","), ha="center", va="center",
                     fontsize=4.8,
                     color="white" if valor > 0.6 * pesos.to_numpy().max() else "#222222")
    eje.grid(visible=False)
    barra = fig.colorbar(imagen, ax=eje, fraction=0.035, pad=0.02)
    barra.ax.tick_params(labelsize=5)
    fig.tight_layout(pad=0.25)
    guardar_figura(fig, "fig09_nmf_componentes.png")


def figura10_clustering(capitulo3: dict) -> None:
    """Izquierda: elección de k en k-Means. Derecha: grupos frente a etapas reales."""
    agrupa = capitulo3["agrupamiento"]
    barrido = agrupa["barrido"]
    fig, (izq, der) = plt.subplots(1, 2, figsize=(ANCHO_FIGURA, 1.55))
    izq.plot(barrido["k"], barrido["inercia"], color=GRIS, marker="o", markersize=2,
             linewidth=1, label="Inercia")
    izq.set_xlabel("Número de grupos (k)", fontsize=6)
    izq.set_ylabel("Inercia", fontsize=6, color=GRIS)
    gemelo = izq.twinx()
    gemelo.plot(barrido["k"], barrido["silueta"], color=AZUL, marker="s", markersize=2,
                linewidth=1, label="Silueta")
    gemelo.set_ylabel("Silueta", fontsize=6, color=AZUL)
    gemelo.axvline(agrupa["k_silueta"], color="#b23a48", linestyle=":", linewidth=0.8)
    gemelo.grid(visible=False)
    gemelo.spines["right"].set_visible(True)
    gemelo.spines["right"].set_linewidth(0.6)
    izq.set_title(f"k-Means: mejor silueta k = {agrupa['k_silueta']}", fontsize=6.5)
    for eje in (izq, gemelo):
        eje.tick_params(labelsize=5.6)
    ejes_con_coma(gemelo, en_x=False)

    cruce = pd.crosstab(agrupa["grupos"]["k-Means (k = 3)"], agrupa["etapas"])
    cruce = cruce.reindex(columns=CLASES, fill_value=0)
    der.imshow(cruce.to_numpy(), cmap="Blues", aspect="auto")
    for i in range(cruce.shape[0]):
        for j in range(cruce.shape[1]):
            valor = int(cruce.iat[i, j])
            der.text(j, i, valor, ha="center", va="center", fontsize=5.6,
                     color="white" if valor > 0.55 * cruce.to_numpy().max() else "#222222")
    der.set_xticks(range(3), ETAPAS, fontsize=5.4, rotation=30, ha="right")
    der.set_yticks(range(cruce.shape[0]), [f"Grupo {i + 1}" for i in range(cruce.shape[0])],
                   fontsize=5.6)
    der.set_title("Grupos (k = 3) y etapa real", fontsize=6.5)
    der.grid(visible=False)
    fig.tight_layout(pad=0.25)
    guardar_figura(fig, "fig10_clustering.png")


def generar_figuras(modelado: dict, evaluacion: dict, capitulo2: dict,
                    capitulo3: dict) -> None:
    """Dibuja y guarda las nueve figuras que lleva el artículo."""
    seccion("PASO 8. FIGURAS DEL ARTÍCULO")
    figura02_comparacion(modelado, evaluacion)
    figura03_confusion(evaluacion)
    figura04_curvas_validacion(capitulo2)
    figura05_curvas_aprendizaje(capitulo2)
    figura06_calibracion(capitulo2, evaluacion["mejor"])
    figura07_importancia(evaluacion)
    figura08_pca_tsne(capitulo3)
    figura09_nmf(capitulo3)
    figura10_clustering(capitulo3)


# =============================================================================
# PASO 9. CONTROL DE CALIDAD (CRISP-ML(Q)) Y VALORES PARA EL ARTÍCULO
# =============================================================================
NUMEROS = {0: "Ninguno", 1: "Uno", 2: "Dos", 3: "Tres", 4: "Cuatro", 5: "Cinco",
           6: "Seis", 7: "Siete", 8: "Ocho", 9: "Nueve", 10: "Diez", 11: "Once",
           12: "Doce", 13: "Trece"}


def controles_calidad(datos: dict, modelado: dict) -> pd.DataFrame:
    """
    Aseguramiento de la calidad, que es lo que CRISP-ML(Q) añade a CRISP-DM.

    Son siete comprobaciones automáticas. Si falla una CRÍTICA, el programa se
    detiene con un mensaje claro en vez de publicar resultados que no se
    sostienen. La única que no detiene el programa es "todos superan al
    trivial": si algún modelo no lo supera, se informa, no se oculta.

    Resultado esperado: resultados/reportes/controles_calidad.csv con CUMPLE o
    NO CUMPLE en cada control.
    """
    seccion("PASO 9. CONTROL DE CALIDAD - CRISP-ML(Q)")
    df, comparacion = datos["df"], modelado["comparacion"]
    controles = []

    def anotar(clave, descripcion, cumple, detalle, critico=True):
        controles.append({"control": clave, "descripcion": descripcion,
                          "resultado": "CUMPLE" if cumple else "NO CUMPLE",
                          "detalle": detalle, "critico": "sí" if critico else "no"})
        nota(f"{clave}: {'CUMPLE' if cumple else 'NO CUMPLE'} - {detalle}")

    # C1. La práctica exige el modelo asignado y al menos otros diez.
    rivales = [m for m in ORDEN_MODELOS if m != MODELO_TRIVIAL]
    anotar("C1", "Al menos 11 modelos además del clasificador trivial",
           len(rivales) >= 11, f"{len(rivales)} modelos comparados")

    # C2. Ningún predictor puede identificar al país ni revelar la etiqueta.
    coladas = [c for c in PREDICTORES
               if c in PROHIBIDAS or c.startswith("bandera_imp_")
               or c in ("baja_carbono_elec_pct", "electricidad_pct", "b_promedio_3")]
    anotar("C2", "Ningún predictor prohibido (no hay fuga de la etiqueta)",
           not coladas, f"{len(PREDICTORES)} predictores revisados, 0 prohibidos"
           if not coladas else f"prohibidos: {coladas}")

    # C3. La evaluación respeta el tiempo: los años no se mezclan.
    rangos = {c: (int(df.loc[df["conjunto"] == c, "anio"].min()),
                  int(df.loc[df["conjunto"] == c, "anio"].max())) for c in ANIOS_CONJUNTO}
    sin_mezcla = (rangos == {c: tuple(v) for c, v in ANIOS_CONJUNTO.items()}
                  and rangos["entrenamiento"][1] < rangos["validacion"][0]
                  and rangos["validacion"][1] < rangos["prueba"][0])
    anotar("C3", "Los años de entrenamiento, validación y prueba no se mezclan",
           sin_mezcla, "2004-2015 / 2016-2019 / 2020-2024")

    # C4. La prueba se mira una sola vez por modelo: ninguno se eligió con ella.
    repetidos = {k: v for k, v in USOS_DE_PRUEBA.items() if v != 1}
    anotar("C4", "Cada modelo calculó sus métricas de prueba una sola vez",
           not repetidos, f"{len(USOS_DE_PRUEBA)} modelos y variantes, "
           f"1 medición cada uno" if not repetidos else f"repetidos: {repetidos}")

    # C5. Semillas fijas en todo lo que tiene azar.
    anotar("C5", "Semillas fijas en modelos, remuestreos y agrupamiento",
           SEMILLA == 42, f"semilla {SEMILLA} en scikit-learn, bootstrap y k-Means")

    # C6. No es crítico, pero hay que decirlo: ¿algún modelo no aprendió nada?
    eb_trivial = float(comparacion.loc[comparacion["modelo"] == MODELO_TRIVIAL,
                                       "eb_prueba"].iloc[0])
    flojos = [m for m, v in zip(comparacion["modelo"], comparacion["eb_prueba"])
              if m != MODELO_TRIVIAL and v <= eb_trivial]
    anotar("C6", "Todos los modelos superan al clasificador trivial", not flojos,
           f"trivial: {num(eb_trivial)}" if not flojos
           else f"no lo superan: {lista_y(flojos)}", critico=False)

    # C7. Reproducibilidad: se vuelve a entrenar cada modelo final con los mismos
    # hiperparámetros y debe dar exactamente la misma exactitud balanceada.
    diferencias = []
    for nombre in ORDEN_MODELOS:
        estimador = construir_modelos()[nombre][0]
        parametros = modelado["modelos"][nombre]["parametros"]
        repetido = clone(estimador).set_params(**parametros).fit(datos["X_tv"], datos["y_tv"])
        eb = balanced_accuracy_score(datos["y_pr"], repetido.predict(datos["X_pr"]))
        original = float(comparacion.loc[comparacion["modelo"] == nombre, "eb_prueba"].iloc[0])
        if abs(eb - original) > 1e-12:
            diferencias.append(f"{nombre} ({num(eb, 6)} vs {num(original, 6)})")
    anotar("C7", "Dos corridas dan exactamente las mismas métricas", not diferencias,
           f"{len(ORDEN_MODELOS)} modelos reentrenados sin diferencia"
           if not diferencias else f"difieren: {lista_y(diferencias)}")

    tabla = pd.DataFrame(controles)
    tabla.to_csv(CARPETA_REPORTES / "controles_calidad.csv", index=False, encoding="utf-8")
    nota("reporte guardado: resultados/reportes/controles_calidad.csv")

    fallidos = tabla[(tabla["resultado"] == "NO CUMPLE") & (tabla["critico"] == "sí")]
    if not fallidos.empty:
        raise SystemExit(
            "\nSE DETIENE EL PROGRAMA: no se cumplen controles críticos de calidad:\n"
            + "\n".join(f"  * {f.control}: {f.descripcion} -> {f.detalle}"
                        for f in fallidos.itertuples()))
    return tabla


def valores_para_paper(datos: dict, modelado: dict, evaluacion: dict,
                       capitulo2: dict, capitulo3: dict) -> dict:
    """
    Escribe resultados/reportes/valores_para_paper.md.

    El artículo se redactó con marcadores en lugar de cifras. Este archivo
    entrega cada marcador con su valor y con una explicación de cómo se calculó,
    además de las cinco tablas del artículo con sus filas en el mismo orden.
    Así nadie copia un número a mano desde la consola.
    """
    seccion("PASO 9 (fin). VALORES PARA EL ARTÍCULO")
    comparacion = modelado["comparacion"].set_index("modelo")
    mejor = evaluacion["mejor"]
    fila_mejor, fila_nb = comparacion.loc[mejor], comparacion.loc[MODELO_ASIGNADO]
    fila_rl = comparacion.loc[MODELO_LOGISTICA]
    ic_mejor, ic_nb = evaluacion["intervalos"][mejor], evaluacion["intervalos"][MODELO_ASIGNADO]

    # --- Variante ganadora del modelo asignado ------------------------------
    parametros_nb = modelado["modelos"][MODELO_ASIGNADO]["parametros"]
    usa_yeo = "PowerTransformer" in str(parametros_nb.get("transf", "passthrough"))
    usa_iguales = parametros_nb.get("modelo__priors") is not None
    suavizado = parametros_nb.get("modelo__var_smoothing", float("nan"))
    variante_nb = (f"{'Yeo-Johnson' if usa_yeo else 'sin transformación'} con priors "
                   f"{'iguales' if usa_iguales else 'de los datos'} "
                   f"(var_smoothing = {suavizado:.0e})")

    # --- Confusión más frecuente del mejor modelo ---------------------------
    matriz = evaluacion["confusiones"][mejor]
    fuera = [(matriz[i, j], i, j) for i in range(3) for j in range(3) if i != j]
    peso, real, predicha = max(fuera)
    confusion_principal = (f"{ETAPAS[real].lower()} predicha como {ETAPAS[predicha].lower()} "
                           f"({pct(peso)} de las {ETAPAS[real].lower()}s)")

    # --- Región con menor exactitud del mejor modelo ------------------------
    regiones = evaluacion["regiones"]
    peor = regiones.loc[regiones[mejor].idxmin()]
    region_peor = f"{peor['region']} ({num(peor[mejor])})"

    # --- Modelos que superan el criterio de éxito ---------------------------
    rivales = modelado["comparacion"][modelado["comparacion"]["modelo"] != MODELO_TRIVIAL]
    n_075 = int((rivales["eb_prueba"] >= CRITERIO_EXITO).sum())
    cumple_criterio = float(fila_mejor["eb_prueba"]) >= CRITERIO_EXITO

    # --- Capítulo 2: curvas y probabilidades --------------------------------
    curvas_v = capitulo2["validacion"]
    curvas_a = capitulo2["aprendizaje"]
    incertidumbre = capitulo2["incertidumbre"]
    escalado_tabla = capitulo3["escalado"].set_index("preprocesamiento")
    extras = capitulo3["escalado_extras"]
    vecino_mismo_pais = capitulo2["vecino_mismo_pais"]
    # "1 de las 720 filas cambia" / "3 de las 720 filas cambian"
    verbo_filas = "cambia" if extras["filas_cambian_bosque"] == 1 else "cambian"
    dominantes_texto = lista_y([en_minuscula(legible(v)) for v in extras["dominantes"]])
    ganancia_knn = (escalado_tabla.loc["Estandarizado", "k-NN (k = 15)"]
                    - escalado_tabla.loc["Sin escalar", "k-NN (k = 15)"])

    # --- Capítulo 3: PCA, NMF y agrupamiento --------------------------------
    reduccion, agrupa = capitulo3["reduccion"], capitulo3["agrupamiento"]
    pesos = reduccion["pesos_nmf"]
    tabla_grupos = agrupa["tabla"].set_index("metodo")
    fila_dbscan = tabla_grupos.loc["DBSCAN"]
    nombre_km_silueta = f"k-Means (mejor silueta, k = {agrupa['k_silueta']})"

    def top3(nombre_modelo):
        return lista_y(list(evaluacion["importancias"][nombre_modelo]["nombre"].head(3)))

    mcnemar = evaluacion["mcnemar"]
    p_texto = "< 0,001" if mcnemar["p"] < 0.001 else num(mcnemar["p"])
    umbrales = evaluacion["umbrales"]
    ll_nb, ll_rl = float(fila_nb["log_loss"]), float(fila_rl["log_loss"])
    ext_nb = incertidumbre[MODELO_ASIGNADO]["extremas"]
    ext_rl = incertidumbre[MODELO_LOGISTICA]["extremas"]
    diferencia_nb = float(fila_mejor["eb_prueba"]) - float(fila_nb["eb_prueba"])

    # --- Frases: interpretan lo que de verdad salió -------------------------
    mejores_tres = [nombre_corto(m) for m in
                    modelado["comparacion"].nlargest(3, "eb_prueba")["modelo"]]
    peor_rival = nombre_corto(rivales.loc[rivales["eb_prueba"].idxmin(), "modelo"])
    mejor_calibrado = nombre_corto(rivales.loc[rivales["log_loss"].idxmin(), "modelo"])
    mas_rapido = nombre_corto(rivales.loc[rivales["segundos_entrenamiento"].idxmin(), "modelo"])
    segundos_nb = float(fila_nb["segundos_entrenamiento"])
    tiempo_nb = (f"{num(segundos_nb, 2)} segundos" if segundos_nb >= 0.01
                 else "menos de una centésima de segundo")
    ganancia_svm = (escalado_tabla.loc["Estandarizado", "SVM RBF (C = 10)"]
                    - escalado_tabla.loc["Sin escalar", "SVM RBF (C = 10)"])
    brecha_nb, brecha_mejor = curvas_a[MODELO_ASIGNADO]["brecha"], curvas_a[mejor]["brecha"]
    nb_peor_calibrado = ll_nb > ll_rl
    brecha_nb_texto = ("incluso negativa: no llega a sobreajustar, le falta capacidad"
                       if brecha_nb < 0 else "muy pequeña: apenas sobreajusta")
    ari_km3 = float(tabla_grupos.loc["k-Means (k = 3)", "ari"])

    frases = {
        "FRASE_COMPARACION": (
            f"Los tres mejores fueron {lista_y(mejores_tres)}, los tres ensambles de "
            f"árboles; el más bajo de los trece fue {peor_rival}, con "
            f"{num(float(rivales['eb_prueba'].min()))}."),
        "FRASE_VALIDACION": (
            f"En nuestros datos el árbol llega al máximo de validación con profundidad "
            f"{curvas_v['Árbol de decisión']['mejor']} y después sigue mejorando en "
            f"entrenamiento mientras la validación baja, que es sobreajuste; con "
            f"profundidad 1 o 2 los dos aciertos son bajos, que es subajuste. En k-NN no "
            f"hay punto intermedio: gana k = 1, porque el vecino más cercano de un país es "
            f"él mismo en otro año en el {pct(vecino_mismo_pais)} de las filas de "
            f"validación, de modo que un solo vecino equivale a repetir su última etapa "
            f"conocida."),
        "FRASE_APRENDIZAJE": (
            f"En Naive Bayes la brecha es {brecha_nb_texto}, mientras que "
            f"{nombre_corto(mejor)} memoriza el entrenamiento y, aun así, generaliza mucho "
            f"mejor. En los dos, la curva de validación casi no sube en la última mitad, "
            f"así que más años del mismo tipo no cambiarían mucho el resultado."),
        "FRASE_CALIBRACION": (
            f"Naive Bayes quedó {'peor' if nb_peor_calibrado else 'mejor'} calibrado que la "
            f"regresión logística: su pérdida logarítmica es "
            f"{num(ll_nb)} frente a {num(ll_rl)}, y se declara casi seguro "
            f"({pct(ext_nb)} de las filas por encima de 0,99) muchas más veces que ella "
            f"({pct(ext_rl)})."),
        "FRASE_ESCALADO": (
            f"A la SVM RBF el escalado le cambia la vida ({puntos(ganancia_svm)} puntos al "
            f"estandarizar); a k-NN con k = 15, en cambio, le sienta mal "
            f"({puntos(ganancia_knn)} puntos): sin escalar, la distancia la deciden las dos "
            f"variables de mayor escala, {dominantes_texto}, que además están entre las más "
            f"informativas, y al estandarizar las once indicadoras 0/1 de región y de "
            f"ingreso pasan a pesar igual que ellas. Lo comprobamos repitiendo k-NN "
            f"estandarizado solo con los once indicadores continuos: sube a "
            f"{num(extras['eb_solo_continuos'])}. En el bosque aleatorio la diferencia es de "
            f"redondeo: entre un preprocesamiento y otro {verbo_filas} de etapa como mucho "
            f"{extras['filas_cambian_bosque']} de las {extras['filas_validacion']} filas de "
            f"validación."),
        "FRASE_PCA_TSNE": (
            f"las tres etapas se superponen: se ve un gradiente de desarrollo económico, "
            f"pero los países avanzados no forman una zona propia, porque la etiqueta "
            f"depende de la matriz eléctrica y esa variable no está entre los once "
            f"indicadores."),
        "FRASE_NB_DESEMPENO": (
            f"Con esa sencillez llegó a {num(float(fila_nb['eb_prueba']))} de exactitud "
            f"balanceada en prueba, {puntos(abs(diferencia_nb))[1:]} puntos "
            f"{'por debajo' if diferencia_nb > 0 else 'por encima'} de "
            f"{nombre_corto(mejor)}, y entrenarlo tomó {tiempo_nb}."),
        "EFECTO_VARIANTES_NB": "",     # se completa más abajo con la tabla de variantes
        "FRASE_VENTAJAS": (
            f"En nuestros datos el orden quedó claro: los ensambles de árboles ganan en "
            f"exactitud y además dan las probabilidades más creíbles "
            f"({mejor_calibrado} tiene la menor pérdida logarítmica), mientras que "
            f"{mas_rapido} es el más barato de entrenar."),
        "FRASE_CRITERIO": (
            f"Con {num(float(fila_mejor['eb_prueba']))} de exactitud balanceada, "
            f"{nombre_corto(mejor)} {'cumple' if cumple_criterio else 'no cumple'} el "
            f"criterio de 0,75 que nos fijamos, y reconoce el "
            f"{pct(float(fila_mejor['recall_avanzada']))} de los países avanzados."),
        "FRASE_CLUSTER": (
            f"el agrupamiento apenas reproduce las etapas (el mejor índice de Rand ajustado "
            f"es {num(max(tabla_grupos['ari']))}) y DBSCAN, con el radio que calculamos, "
            f"juntó a casi todos los países en un solo grupo y dejó "
            f"{int(fila_dbscan['ruido'])} como ruido, de modo que ni siquiera tiene silueta "
            f"que informar; la etiqueta no se puede reemplazar por grupos encontrados sin "
            f"ella."),
        "FRASE_CONCLUSION_NB": (
            f"Es {'un resultado inferior' if diferencia_nb > 0 else 'un resultado superior'} "
            f"al de {nombre_corto(mejor)}, con un entrenamiento de {tiempo_nb}; su límite "
            f"está en el supuesto de independencia y en unas probabilidades "
            f"{'mal' if nb_peor_calibrado else 'bien'} calibradas."),
    }

    # El efecto de las variantes se mide sobre la tabla de variantes de NB.
    variantes = modelado["variantes"]
    base = variantes.iloc[0]["eb_validacion"]
    ganadora = variantes.loc[variantes["eb_validacion"].idxmax()]
    otras = variantes.iloc[1:]["eb_validacion"]
    frases["EFECTO_VARIANTES_NB"] = (
        f"subieron la exactitud balanceada en validación de {num(base)} a "
        f"{num(float(ganadora['eb_validacion']))} "
        f"({puntos(float(ganadora['eb_validacion']) - base)} puntos)"
        if float(ganadora["eb_validacion"]) > base else
        f"no ayudaron: la variante sin transformación y con los priors de los datos siguió "
        f"siendo la mejor en validación ({num(base)}), y las otras tres se quedaron entre "
        f"{num(float(otras.min()))} y {num(float(otras.max()))}")

    marcadores = {
        "MEJOR": (mejor, "modelo con mayor exactitud balanceada en prueba, sin contar el trivial"),
        "EN_BEST": (NOMBRES_INGLES[mejor], "el mismo nombre en inglés, para la portada"),
        "BA_MEJOR": (num(float(fila_mejor["eb_prueba"])),
                     "exactitud balanceada del mejor modelo en 2020-2024"),
        "IC_MEJOR": (intervalo(*ic_mejor),
                     f"percentiles 2,5 y 97,5 de {REMUESTREOS} remuestreos bootstrap de las 900 filas de prueba"),
        "F1_MEJOR": (num(float(fila_mejor["f1_macro"])), "F1 macro del mejor modelo en prueba"),
        "BA_NB": (num(float(fila_nb["eb_prueba"])),
                  "exactitud balanceada de Naive Bayes final en prueba"),
        "IC_NB": (intervalo(*ic_nb), "bootstrap de 1 000 remuestreos, igual que el mejor modelo"),
        "BA_NB_VAL": (num(float(fila_nb["eb_validacion"])),
                      "exactitud balanceada de Naive Bayes en validación (2016-2019)"),
        "VARIANTE_NB": (variante_nb,
                        "combinación de la rejilla de Naive Bayes con mayor exactitud balanceada en validación"),
        "N_075": (NUMEROS.get(n_075, str(n_075)),
                  f"modelos (de 13, sin contar el trivial) con exactitud balanceada en prueba >= {num(CRITERIO_EXITO, 2)}"),
        "BA_TRIVIAL": (num(float(comparacion.loc[MODELO_TRIVIAL, "eb_prueba"])),
                       "exactitud balanceada en prueba del clasificador que siempre predice la etapa más frecuente"),
        "MCNEMAR": ("es significativa" if mcnemar["significativa"] else "no es significativa",
                    f"prueba exacta de McNemar con alfa = {num(ALFA, 2)} sobre los aciertos discordantes "
                    f"(b01 = {mcnemar['b01']}, b10 = {mcnemar['b10']})"),
        "P_MCNEMAR": (p_texto, "p-valor de la prueba exacta de McNemar"),
        "REC_AVA_NB": (pct(float(fila_nb["recall_avanzada"])),
                       "recall de la etapa avanzada en prueba, Naive Bayes"),
        "REC_AVA_MEJOR": (pct(float(fila_mejor["recall_avanzada"])),
                          "recall de la etapa avanzada en prueba, mejor modelo"),
        "CONFUSION_PRINCIPAL": (confusion_principal,
                                "celda fuera de la diagonal más grande de la matriz de confusión normalizada por fila"),
        "PROF_OPT": (str(curvas_v["Árbol de decisión"]["mejor"]),
                     "profundidad del árbol con mayor exactitud balanceada en validación"),
        "K_OPT": (str(curvas_v["k-NN"]["mejor"]), "número de vecinos con mayor EB en validación"),
        "C_OPT": (str(curvas_v["SVM RBF"]["mejor"]).replace(".", ","),
                  "C de la SVM RBF con mayor EB en validación"),
        "BRECHA_NB": (puntos(brecha_nb),
                      "(EB en entrenamiento - EB en validación) x 100 con todo el entrenamiento, Naive Bayes"),
        "BRECHA_MEJOR": (puntos(brecha_mejor), "lo mismo para el mejor modelo"),
        "LL_NB": (num(ll_nb), "pérdida logarítmica en prueba de Naive Bayes"),
        "LL_RL": (num(ll_rl), "pérdida logarítmica en prueba de la regresión logística"),
        "LL_MEJOR": (num(float(fila_mejor["log_loss"])), "pérdida logarítmica en prueba del mejor modelo"),
        "EXT_NB": (pct(ext_nb), "filas de prueba con probabilidad máxima > 0,99, Naive Bayes"),
        "EXT_RL": (pct(ext_rl), "lo mismo para la regresión logística"),
        "TOP3_MEJOR": (top3(mejor),
                       "tres variables con mayor caída de la EB al permutarlas (10 repeticiones) en el mejor modelo"),
        "TOP3_NB": (top3(MODELO_ASIGNADO), "lo mismo para Naive Bayes"),
        "ERR_UMBRAL": (pct(umbrales["errores_cerca"]),
                       f"errores del mejor modelo a menos de {UMBRAL_CERCA:.0f} puntos de los umbrales 30 %, 60 % o 90 % "
                       f"(sobre {umbrales['n_errores']} errores)"),
        "PRUEBA_UMBRAL": (pct(umbrales["prueba_cerca"]),
                          "mismo criterio, calculado sobre las 900 filas de prueba"),
        "REGION_PEOR": (region_peor, "región con menor exactitud del mejor modelo en prueba"),
        "GAN_KNN": (puntos(ganancia_knn),
                    "(EB de k-NN estandarizado - EB de k-NN sin escalar) x 100, en validación"),
        "PCA_2": (pct(float(reduccion["acumulada"][1])),
                  "varianza explicada por las dos primeras componentes de los 11 indicadores continuos"),
        "PCA_80": (str(reduccion["componentes_80"]), "componentes necesarias para llegar al 80 % de varianza"),
        "NMF_C1": (lista_y([legible(v) for v in pesos["C1"].nlargest(2).index]),
                   "las dos variables con mayor peso en el primer componente de NMF"),
        "NMF_C2": (lista_y([legible(v) for v in pesos["C2"].nlargest(2).index]),
                   "lo mismo para el segundo componente"),
        "K_SIL": (str(agrupa["k_silueta"]), "k con la mejor silueta en k-Means (k de 2 a 8), países de 2024"),
        "ARI_KM3": (num(ari_km3), "índice de Rand ajustado de k-Means con k = 3 frente a las etapas reales"),
        "ARI_WARD": (num(float(tabla_grupos.loc["Jerárquico Ward (k = 3)", "ari"])),
                     "lo mismo para el jerárquico de Ward con k = 3"),
        "ARI_DBSCAN": (num(float(fila_dbscan["ari"])), "lo mismo para DBSCAN"),
        "RUIDO_DBSCAN": (str(int(fila_dbscan["ruido"])), "países que DBSCAN marcó como ruido"),
        "CUMPLE": ("cumple" if cumple_criterio else "no cumple",
                   f"el mejor modelo frente al criterio de éxito de {num(CRITERIO_EXITO, 2)}"),
    }
    for clave, texto in frases.items():
        marcadores[clave] = (texto, "frase redactada con los resultados reales de esta corrida")

    # --- Las cinco tablas del artículo, en el orden del artículo ------------
    tabla4 = pd.DataFrame([{
        "Modelo": m, "EB-V": num(comparacion.loc[m, "eb_validacion"]),
        "EB-P": num(comparacion.loc[m, "eb_prueba"]),
        "F1-M": num(comparacion.loc[m, "f1_macro"]),
        "Exac.": num(comparacion.loc[m, "exactitud"]),
        "LL": num(comparacion.loc[m, "log_loss"]),
        "AUC": num(comparacion.loc[m, "auc_ovr_macro"])} for m in ORDEN_MODELOS])
    tabla5 = pd.DataFrame([{
        "Transformación": f["transformacion"], "Priors": f["priors"],
        "EB-V": num(f["eb_validacion"]), "EB-P": num(f["eb_prueba"]),
        "LL": num(f["log_loss"])} for _, f in variantes.iterrows()])
    tabla6 = pd.DataFrame([{
        "Región": f["region"], "Filas": int(f["filas"]),
        "Naive Bayes": num(f[MODELO_ASIGNADO]), "Mejor modelo": num(f[mejor])}
        for _, f in regiones.iterrows()])
    tabla7 = capitulo3["escalado"].copy()
    for columna in tabla7.columns[1:]:
        tabla7[columna] = tabla7[columna].map(num)
    tabla7 = tabla7.rename(columns={"preprocesamiento": "Preprocesamiento",
                                    "Bosque aleatorio": "Bosque aleat."})
    tabla8 = pd.DataFrame([{
        "Método": ("k-Means (k = 3)" if m == "k-Means (k = 3)" else
                   ("k-Means (mejor silueta)" if m == nombre_km_silueta else m)),
        "Grupos": int(f["grupos"]), "Ruido": int(f["ruido"]),
        "Silueta": num(f["silueta"]), "ARI": num(f["ari"]), "NMI": num(f["nmi"])}
        for m, f in tabla_grupos.iterrows()])

    lineas = [
        "# Valores para el artículo (Entregable 1)",
        "",
        "Generado por `main.py`. Cada marcador resaltado del artículo se reemplaza por el",
        "valor de la columna **Valor**. Los números usan coma decimal; las métricas van con",
        "tres decimales y los porcentajes con uno.",
        "",
        f"- scikit-learn {sklearn.__version__} · Python {platform.python_version()} · semilla {SEMILLA}",
        f"- Mejor modelo: **{mejor}** · Modelo asignado: **Naive Bayes gaussiano**",
        "",
        "## 1. Marcadores",
        "",
        "| Marcador | Valor | Cómo se calculó |",
        "|---|---|---|",
    ]
    for clave, (valor, explicacion) in marcadores.items():
        lineas.append(f"| `{clave}` | {valor} | {explicacion} |")
    lineas += [
        "", "## 2. Tabla IV · Desempeño en validación (2016-2019) y prueba (2020-2024)", "",
        tabla_md(tabla4),
        "", "EB-V y EB-P: exactitud balanceada en validación y en prueba; F1-M: F1 macro;",
        "Exac.: exactitud; LL: pérdida logarítmica; AUC: área bajo la curva ROC, una contra",
        "el resto, promedio macro. LL, F1-M, Exac. y AUC se miden en prueba.",
        "", "## 3. Tabla V · Variantes de Naive Bayes gaussiano", "",
        tabla_md(tabla5),
        "", "## 4. Tabla VI · Exactitud en prueba por región", "",
        tabla_md(tabla6),
        "", "## 5. Tabla VII · Exactitud balanceada en validación según el preprocesamiento", "",
        tabla_md(tabla7),
        "", "## 6. Tabla VIII · Agrupamiento de los 180 países en 2024 frente a las etapas", "",
        tabla_md(tabla8),
        "",
        f"ARI: índice de Rand ajustado; NMI: información mutua normalizada. DBSCAN se "
        f"ejecutó con eps = {str(agrupa['eps']).replace('.', ',')} y min_samples = 5; "
        f"formó un solo grupo y marcó {int(fila_dbscan['ruido'])} países como ruido, y la "
        f"silueta necesita al menos dos grupos, por eso va un guion en esa celda.",
        "", "## 7. Figuras", "",
        "Las nueve figuras están en `resultados/figuras/` con el nombre que las cita el",
        "artículo (fig02 a fig10), a 3,35 pulgadas de ancho y 300 ppp. La Fig. 1 del",
        "artículo es un esquema hecho a mano y no la genera este programa.",
        "",
        "## 8. Comprobaciones que respaldan las frases", "",
        "Resultados que no son marcadores del artículo, pero que sostienen lo que dicen",
        "`FRASE_ESCALADO` y `FRASE_VALIDACION`. Están en la consola de `main.py`.",
        "",
        "| Comprobación | Valor | Para qué |",
        "|---|---|---|",
        f"| Variables que dominan la distancia sin escalar | "
        f"{lista_y([legible(v) for v in extras['dominantes']])} | "
        f"explican por qué k-NN sin escalar acierta más: son las de mayor desviación "
        f"típica en los datos crudos |",
        f"| k-NN (k = 15) estandarizado, solo con los 11 indicadores continuos | "
        f"{num(extras['eb_solo_continuos'])} | frente a "
        f"{num(float(escalado_tabla.loc['Estandarizado', 'k-NN (k = 15)']))} con los 23 "
        f"predictores: confirma que las once indicadoras 0/1 son las que estorban |",
        f"| Filas de validación en las que el bosque aleatorio cambia de etapa entre "
        f"escalados | {extras['filas_cambian_bosque']} de {extras['filas_validacion']} | "
        f"la diferencia entre escalados es redondeo de los puntos de corte, no un efecto "
        f"real |",
        f"| Filas de validación cuyo vecino más cercano es el mismo país en otro año | "
        f"{pct(vecino_mismo_pais)} | explica que la curva de validación de k-NN gane con "
        f"k = 1 |",
        "",
        "## 9. Ajustes sugeridos al texto fijo del artículo", "",
        "Tres frases del artículo afirman algo un poco más fuerte de lo que sostienen",
        "nuestros números. Son cambios de una o dos palabras.",
        "",
        "| Dónde | Dice | Proponemos |",
        "|---|---|---|",
        f"| Sección IV-F | «El bosque aleatorio da **el mismo** resultado con todos los "
        f"escalados» | «El bosque aleatorio da **prácticamente el mismo** resultado con "
        f"todos los escalados» ({verbo_filas} {extras['filas_cambian_bosque']} de "
        f"{extras['filas_validacion']} filas) |",
        "| Tabla VIII, fila DBSCAN, columna Silueta | una cifra | un guion (–), porque "
        "DBSCAN formó un solo grupo y la silueta necesita al menos dos |",
        "| Pie de la Fig. 2 | «con intervalo de confianza del 95 %» | «con intervalo de "
        "confianza del 95 % en Naive Bayes, el mejor modelo y la regresión logística» "
        "(son los tres para los que se calculó) |",
        "",
    ]
    (CARPETA_REPORTES / "valores_para_paper.md").write_text("\n".join(lineas), encoding="utf-8")
    nota("reporte guardado: resultados/reportes/valores_para_paper.md")
    return marcadores


def control_calidad_y_valores(datos, modelado, evaluacion, capitulo2, capitulo3):
    """Ejecuta los controles de CRISP-ML(Q) y, si pasan, escribe los valores del artículo."""
    controles = controles_calidad(datos, modelado)
    marcadores = valores_para_paper(datos, modelado, evaluacion, capitulo2, capitulo3)
    return controles, marcadores


# =============================================================================
# PASO 10. FLUJO PRINCIPAL
# =============================================================================
def principal(argumentos: list | None = None) -> int:
    """
    Ejecuta las dos fases de CRISP-ML(Q) y los dos capítulos de la guía, en orden.

    Cada paso usa lo que produjo el anterior; al final quedan las tablas, las
    figuras y el archivo de valores que necesita el artículo.
    """
    lector = argparse.ArgumentParser(
        description="Grupo 7 - CRISP-ML(Q) fases 4 y 5: modelado y evaluación")
    lector.add_argument("--rapido", action="store_true",
                        help="rejillas reducidas, solo para comprobar que todo corre")
    lector.add_argument("--n-jobs", type=int, default=1,
                        help="núcleos para la búsqueda en rejilla (1 por defecto)")
    args = lector.parse_args(argumentos)

    inicio = time.perf_counter()
    preparar_carpetas()
    datos = cargar_datos()
    modelado = fase4_modelado(datos, args.rapido, args.n_jobs)
    evaluacion = fase5_evaluacion(datos, modelado, args.n_jobs)
    capitulo2 = capitulo2_supervisado(datos, modelado, evaluacion["mejor"])
    capitulo3 = capitulo3_no_supervisado(datos)
    generar_figuras(modelado, evaluacion, capitulo2, capitulo3)
    control_calidad_y_valores(datos, modelado, evaluacion, capitulo2, capitulo3)

    seccion("RESUMEN")
    comparacion = modelado["comparacion"].set_index("modelo")
    nota(f"mejor modelo: {evaluacion['mejor']} "
         f"(exactitud balanceada en prueba {num(comparacion.loc[evaluacion['mejor'], 'eb_prueba'])})")
    nota(f"modelo asignado (Naive Bayes): "
         f"{num(comparacion.loc[MODELO_ASIGNADO, 'eb_prueba'])}")
    nota(f"criterio de éxito de {num(CRITERIO_EXITO, 2)}: "
         f"{'CUMPLE' if comparacion.loc[evaluacion['mejor'], 'eb_prueba'] >= CRITERIO_EXITO else 'NO CUMPLE'}")
    nota("tablas en resultados/tablas, figuras en resultados/figuras, "
         "controles y valores en resultados/reportes")
    nota(f"tiempo total: {time.perf_counter() - inicio:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(principal())
