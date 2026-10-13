# Grupo 7 · Modelado de la transición energética (Naive Bayes frente a doce modelos)

**Curso:** Machine Learning · Práctica 2 · Entregable 3 (guía de código en Python)
**Universidad Nacional de Ingeniería**, Facultad de Ingeniería Industrial y de Sistemas, ciclo 2026-II
**Profesor:** Dr. Ing. Hilario Aradiel Castañeda · **Área:** Energía · **Modelo asignado:** Naive Bayes

| Integrante | Correo |
|---|---|
| Crisostomo Cahuas, Gabriel Andre | gabriel.crisostomo.c@uni.pe |
| Curto Meneses, Alessandro del Piero | alessandro.curto.m@uni.pe |
| Rubio Zeta, Luis Miguel | luis.rubio.z@uni.pe |

---

## 1. Descripción del proyecto

¿Se puede saber en qué etapa de la transición energética está un país **antes** de conocer su matriz eléctrica? La matriz eléctrica se publica con retraso, así que clasificamos cada país y año en una de tres etapas usando solo indicadores del **año anterior** (ingreso, urbanización, energía, emisiones, región y grupo de ingreso).

| Etapa | Condición | Lectura |
|---|---|---|
| 0 · Inicial | b̄ < 30 % | La electricidad viene sobre todo de combustibles fósiles |
| 1 · Intermedia | b̄ ≥ 30 % y no cumple la condición de avanzada | Matriz mixta, o limpia pero con acceso incompleto |
| 2 · Avanzada | b̄ ≥ 60 % **y** acceso a la electricidad ≥ 90 % | Electricidad mayoritariamente limpia y casi universal |

donde **b̄** es el promedio de tres años de la electricidad baja en carbono (renovable + nuclear). Las dos condiciones siguen las metas 7.1 (acceso) y 7.2 (energía limpia) del ODS 7.

Seguimos la metodología **CRISP-ML(Q)** ([Studer et al., 2021](https://doi.org/10.48550/arXiv.2003.05155)) y trabajamos **sus dos últimas fases antes del despliegue: modelado (fase 4) y evaluación (fase 5)**. El conjunto de datos ya viene limpio y etiquetado en `datos/`: **este programa no lo vuelve a limpiar y no cambia la etiqueta**.

Usamos **Naive Bayes gaussiano**, el modelo asignado a nuestro grupo, como modelo base, y lo comparamos con **doce algoritmos supervisados más**, además de un clasificador trivial que sirve de línea base. También respondemos, con nuestros propios datos, las preguntas de los **capítulos 2 y 3** de la guía del curso.

**Datos de entrada** (no se modifican):

| Archivo | Qué es |
|---|---|
| `datos/etapas_transicion_preparado.csv` | 3 780 filas × 39 columnas, 180 países, 2004–2024, sin faltantes |
| `datos/etapas_auditoria.csv` | insumos de la etiqueta; solo se usan para revisar los errores cerca de los umbrales |
| `datos/diccionario_datos.csv` | descripción de cada columna |

**Predictores (23):** once indicadores continuos del año anterior (`pib_pc_log_ant`, `poblacion_log_ant`, `urbano_pct_ant`, `industria_pct_ant`, `renovable_pct_ant`, `co2_pc_log_ant`, `energia_pc_log_ant`, `consumo_elec_pc_log_ant`, `var_pib_pc_pct_ant`, `var_urbano_pp_ant`, `var3_renovable_pp_ant`), el año relativo, siete indicadoras de región y cuatro de grupo de ingreso. **No usamos** la composición de la generación eléctrica, porque revelaría la etiqueta.

**Partición temporal:** entrenamiento 2004–2015 (2 160 filas), validación 2016–2019 (720) y prueba 2020–2024 (900). La etapa avanzada pesa 12,9 % en entrenamiento, 19,0 % en validación y 22,9 % en prueba, así que el modelo también tiene que generalizar a un mundo que cambia.

> **Por qué no usamos 2025:** el Banco Mundial todavía no publica el acceso a la electricidad de ese año, y sin ese dato no se puede construir la etiqueta. Cuando aparezca, basta con volver a correr el programa.

---

## 2. Fases trabajadas

### Fase 4 · Modelado (`fase4_modelado`)

Comparamos **catorce modelos**: Naive Bayes, doce rivales y el clasificador trivial.

| Modelo | Hiperparámetros que buscamos |
|---|---|
| Naive Bayes (base) | transformación (ninguna o Yeo-Johnson), priors (de los datos o iguales), `var_smoothing` de 10⁻¹¹ a 10⁻¹ |
| Regresión logística | `C`, pesos de clase |
| k-NN | número de vecinos, pesos uniformes o por distancia |
| Árbol de decisión | profundidad, hojas mínimas, pesos de clase |
| Bosque aleatorio · Extra Trees | 300 árboles; profundidad, hojas mínimas, variables por corte, pesos |
| Gradient Boosting · Hist. Gradient Boosting · AdaBoost | número de árboles, tasa de aprendizaje, profundidad, regularización |
| SVM RBF · SVM lineal | `C`, `gamma`, pesos de clase |
| LDA | solucionador SVD o mínimos cuadrados con contracción |
| Red neuronal MLP | capas ocultas, `alpha`, parada temprana |
| Clasificador trivial | ninguno (predice siempre la etapa más frecuente) |

Con cada modelo hacemos siempre los mismos cuatro pasos:

1. **Búsqueda en rejilla** (`GridSearchCV` con `PredefinedSplit`): entrena con 2004–2015 y elige los hiperparámetros por exactitud balanceada en 2016–2019. Los años nunca se mezclan.
2. **Métricas de validación:** con esos hiperparámetros entrenamos otra vez solo con 2004–2015 y medimos en 2016–2019.
3. **Modelo final:** con los mismos hiperparámetros entrenamos con 2004–2019 y cronometramos el entrenamiento.
4. **Una sola medición** en 2020–2024.

Los modelos que dependen de distancias o de márgenes llevan el escalador **dentro** del pipeline, así que este aprende solo de los años de entrenamiento y no hay fuga de información.

### Fase 5 · Evaluación (`fase5_evaluacion`)

La métrica principal es la **exactitud balanceada** (el promedio del recall de las tres etapas), porque la etapa avanzada es la más rara y la exactitud a secas premiaría al modelo que la ignora. Además reportamos F1 macro, exactitud, recall de cada etapa, pérdida logarítmica, AUC multiclase y tiempo de entrenamiento.

| Qué hacemos | Para qué |
|---|---|
| Bootstrap de 1 000 remuestreos | intervalo de confianza del 95 % de la exactitud balanceada |
| Prueba exacta de McNemar | saber si la diferencia con Naive Bayes es real o es ruido |
| Matrices de confusión normalizadas por fila | ver **en qué** se equivoca cada modelo |
| Exactitud por región | ver **dónde** se equivoca |
| Errores cerca de los umbrales (30 %, 60 %, 90 %) | ver si falla donde la propia regla es frágil |
| Importancia por permutación | saber qué variables usa cada modelo |

### Capítulo 2 de la guía · Aprendizaje supervisado (`capitulo2_supervisado`)

Curvas de validación del árbol, de k-NN y de la SVM RBF (sobreajuste y subajuste), curvas de aprendizaje de Naive Bayes y del mejor modelo (¿ayudarían más datos?) y calibración de las probabilidades (pérdida logarítmica, curvas de calibración e histograma de la probabilidad máxima).

### Capítulo 3 de la guía · No supervisado y preprocesamiento (`capitulo3_no_supervisado`)

Efecto de cinco preprocesamientos (sin escalar, estandarizado, mínimo-máximo, robusto y Yeo-Johnson) en tres modelos; reducción de dimensionalidad con **PCA**, **NMF** y **t-SNE**; y agrupamiento de los 180 países de 2024 con **k-Means**, **jerárquico de Ward** y **DBSCAN**, comparado con las etapas reales mediante silueta, ARI y NMI.

### Control de calidad (`controles_calidad`)

Lo que CRISP-ML(Q) añade a CRISP-DM. Son siete comprobaciones automáticas; si falla una crítica, el programa **se detiene** en vez de publicar resultados que no se sostienen.

| Control | Qué verifica |
|---|---|
| C1 | Hay al menos 11 modelos además del trivial |
| C2 | Ningún predictor prohibido (no hay fuga de la etiqueta) |
| C3 | Los años de entrenamiento, validación y prueba no se mezclan |
| C4 | Cada modelo calculó sus métricas de prueba una sola vez |
| C5 | Semillas fijas en modelos, remuestreos y agrupamiento |
| C6 | Todos los modelos superan al clasificador trivial (no crítico: si falla, se informa) |
| C7 | Dos corridas dan exactamente las mismas métricas |

> **Sobre la reproducibilidad:** con la semilla 42 y las mismas versiones de las librerías, dos corridas dan `comparacion_modelos.csv` idéntico en **todas** las métricas. La única columna que cambia es `segundos_entrenamiento`, porque es tiempo medido con el reloj y depende de la computadora.

---

## 3. Estructura del repositorio

```
grupo7-modelado-transicion-energetica/
├── main.py                 # todo el código, en un único archivo (pasos 0 a 10)
├── main.ipynb              # el mismo código en Jupyter, ya ejecutado y con salidas
├── requirements.txt        # librerías necesarias
├── README.md
├── datos/                  # entradas, no se modifican
│   ├── etapas_transicion_preparado.csv
│   ├── etapas_auditoria.csv
│   └── diccionario_datos.csv
└── resultados/
    ├── tablas/             # comparacion_modelos.csv, variantes_naive_bayes.csv, etc.
    ├── figuras/            # fig02 a fig10 (3,35 pulgadas de ancho, 300 ppp)
    ├── modelos/            # naive_bayes_final.joblib y mejor_modelo.joblib
    └── reportes/           # controles_calidad.csv y valores_para_paper.md
```

`main.py` está dividido en pasos numerados:

| Paso | Contenido |
|---|---|
| 0 | Importación de librerías |
| 1 | Parámetros (rutas, semilla, predictores, criterio de éxito, estilo de las figuras) |
| 2 | Funciones de apoyo (mensajes, guardado, formato de números con coma decimal) |
| 3 | Carga y verificación de los datos |
| 4 | Fase 4 de CRISP-ML(Q): modelado |
| 5 | Fase 5 de CRISP-ML(Q): evaluación |
| 6 | Capítulo 2 de la guía: aprendizaje supervisado |
| 7 | Capítulo 3 de la guía: no supervisado y preprocesamiento |
| 8 | Figuras del artículo (fig02 a fig10) |
| 9 | Control de calidad y `valores_para_paper.md` |
| 10 | Flujo principal (`principal()`) |

---

## 4. Instrucciones de ejecución

### Opción A · Desde la terminal (`main.py`)

1. Instalar **Python 3.10 o superior** (https://www.python.org).
2. Descargar el proyecto: botón verde **Code → Download ZIP** en GitHub (y descomprimir), o bien
   ```bash
   git clone https://github.com/AlessandroCurto/grupo7-modelado-transicion-energetica.git
   cd grupo7-modelado-transicion-energetica
   ```
3. (Opcional) Crear y activar un entorno virtual:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   source .venv/bin/activate       # Linux / macOS
   ```
4. Instalar las librerías:
   ```bash
   pip install -r requirements.txt
   ```
5. Ejecutar:
   ```bash
   python main.py                  # protocolo completo
   python main.py --rapido         # rejillas reducidas, solo para comprobar que todo corre
   python main.py --n-jobs 4       # usa 4 núcleos en la búsqueda de hiperparámetros
   ```
   Tarda unos **7 minutos** con un núcleo (casi todo en la búsqueda de hiperparámetros del paso 4). En la consola se ve el avance de cada paso y el resultado de los siete controles de calidad.

> **Importante:** hace falta **scikit-learn 1.6 o superior**, porque el comportamiento de AdaBoost cambió en esa versión. Las cifras de este README se obtuvieron con scikit-learn 1.9.1 y Python 3.12.

### Opción B · Con Jupyter (`main.ipynb`) usando Anaconda

1. Abrir **Anaconda Navigator** y lanzar **Jupyter Notebook** (o escribir `jupyter notebook` en *Anaconda Prompt*).
2. En el navegador, entrar a la carpeta del proyecto y abrir `main.ipynb`. El notebook debe estar en la misma carpeta que la carpeta `datos/`.
3. Menú **Kernel → Restart Kernel and Run All Cells** (o ejecutar celda por celda con **Shift + Enter**).
4. Para cambiar las opciones, editar la celda de opciones del paso 1: `RAPIDO = True` para una prueba rápida y `N_JOBS = 4` para usar varios núcleos.

Anaconda ya trae pandas, numpy, matplotlib, scipy, joblib y scikit-learn; si la versión de scikit-learn fuera anterior a la 1.6, ejecutar `pip install -r requirements.txt` en *Anaconda Prompt*.

El notebook tiene **el mismo código que `main.py`** y da los mismos resultados; además muestra las tablas y las nueve figuras en pantalla.

---

## 5. Resultados esperados

### El mejor modelo y el modelo asignado

| | Exactitud balanceada en prueba | IC 95 % | F1 macro | Pérdida logarítmica |
|---|---|---|---|---|
| **Extra Trees** (mejor) | **0,895** | 0,875–0,916 | 0,903 | 0,337 |
| **Naive Bayes** (asignado) | 0,611 | 0,579–0,643 | 0,603 | 1,068 |
| Clasificador trivial | 0,333 | — | 0,211 | 19,343 |

**Ocho de los trece modelos** superan el criterio de éxito de 0,75 y **todos** superan al clasificador trivial. La diferencia entre Extra Trees y Naive Bayes **es significativa** según la prueba exacta de McNemar (p < 0,001).

### Los catorce modelos

| Modelo | EB validación | EB prueba | F1 macro | Exactitud | Log loss | AUC |
|---|---|---|---|---|---|---|
| Naive Bayes (base) | 0,660 | 0,611 | 0,603 | 0,618 | 1,068 | 0,765 |
| Regresión logística | 0,724 | 0,717 | 0,691 | 0,708 | 0,690 | 0,862 |
| k-NN | 0,812 | 0,790 | 0,800 | 0,813 | 2,822 | 0,921 |
| Árbol de decisión | 0,795 | 0,773 | 0,783 | 0,799 | 7,249 | 0,833 |
| Bosque aleatorio | 0,861 | 0,875 | 0,879 | 0,886 | 0,367 | 0,966 |
| **Extra Trees** | **0,877** | **0,895** | **0,903** | **0,909** | **0,337** | **0,972** |
| Gradient Boosting | 0,808 | 0,813 | 0,821 | 0,830 | 0,434 | 0,946 |
| Hist. Gradient Boosting | 0,872 | 0,872 | 0,875 | 0,882 | 0,363 | 0,966 |
| AdaBoost | 0,663 | 0,666 | 0,675 | 0,689 | 1,066 | 0,856 |
| SVM RBF | 0,845 | 0,829 | 0,838 | 0,850 | 0,467 | 0,937 |
| SVM lineal | 0,754 | 0,729 | 0,716 | 0,729 | 0,618 | 0,880 |
| LDA | 0,693 | 0,662 | 0,664 | 0,691 | 0,690 | 0,858 |
| Red neuronal MLP | 0,771 | 0,798 | 0,805 | 0,813 | 0,524 | 0,932 |
| Clasificador trivial | 0,333 | 0,333 | 0,211 | 0,463 | 19,343 | 0,500 |

### Lo que aprendimos de cada parte

| Pregunta | Lo que salió |
|---|---|
| ¿Ayudan las variantes de Naive Bayes? | No. La mejor en validación siguió siendo la versión simple (sin transformación y con los priors de los datos, 0,660); Yeo-Johnson y los priors iguales se quedaron entre 0,620 y 0,630 |
| ¿Dónde se equivoca el mejor modelo? | Su confusión más frecuente es clasificar como intermedia a un país avanzado (12,6 % de los avanzados). El **54,9 %** de sus 82 errores cae a menos de 5 puntos de los umbrales de la regla, frente al 16,1 % de todas las filas de prueba |
| ¿Qué variables pesan más? | En Extra Trees, la energía renovable (%) y las indicadoras de Europa y Asia central y de África subsahariana; en Naive Bayes, la energía renovable (%), la industria (% del PIB) y la población urbana (%) |
| ¿Se puede creer en las probabilidades? | Naive Bayes queda peor calibrado que la regresión logística (log loss 1,068 frente a 0,690) y se declara casi seguro (> 0,99) en el 6,8 % de las filas, frente al 2,1 % de la regresión logística |
| ¿Sobreajuste o subajuste? | El árbol alcanza su máximo en validación con profundidad 11; la brecha de la curva de aprendizaje es de −2,8 puntos en Naive Bayes y de +12,3 puntos en Extra Trees. k-NN gana con k = 1 porque el vecino más cercano de un país es él mismo en otro año en el 82,9 % de las filas de validación |
| ¿Importa el escalado? | A la SVM RBF mucho (+16,6 puntos al estandarizar); a k-NN con k = 15 le sienta mal (−3,8 puntos), porque sin escalar la distancia la dominan la energía renovable (%) y la población urbana (%), que son de las más informativas, y al estandarizar las once indicadoras 0/1 pasan a pesar igual: con solo los once indicadores continuos estandarizados sube a 0,727. Al bosque aleatorio no le afecta: **1** de las 720 filas de validación cambia de etapa entre un escalado y otro |
| ¿Aparecen las etapas sin la etiqueta? | No. Dos componentes de PCA explican el 60,7 % de la varianza y hacen falta 5 para llegar al 80 %; el mejor índice de Rand ajustado del agrupamiento es 0,044, así que los grupos no reproducen las etapas |

### Archivos que deja el programa

| Carpeta | Contenido |
|---|---|
| `resultados/tablas/` | `comparacion_modelos.csv`, `mejores_parametros.json`, `variantes_naive_bayes.csv`, `intervalos_bootstrap.csv`, `matrices_confusion.csv`, `errores_por_region.csv`, `importancia_permutacion.csv`, `curvas_validacion.csv`, `curvas_aprendizaje.csv`, `calibracion.csv`, `efecto_escalado.csv`, `nmf_componentes.csv`, `kmeans_barrido.csv`, `agrupamiento.csv` |
| `resultados/figuras/` | `fig02` a `fig10`, las nueve figuras del artículo |
| `resultados/modelos/` | `naive_bayes_final.joblib` y `mejor_modelo.joblib` |
| `resultados/reportes/` | `controles_calidad.csv` (los siete controles en **CUMPLE**) y `valores_para_paper.md` |

`valores_para_paper.md` es la salida pensada para redactar el artículo: trae cada cifra con su explicación y las cinco tablas del artículo con las filas en el mismo orden.
