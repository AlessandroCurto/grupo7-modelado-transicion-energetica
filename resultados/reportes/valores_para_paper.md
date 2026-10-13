# Valores para el artículo (Entregable 1)

Generado por `main.py`. Cada marcador resaltado del artículo se reemplaza por el
valor de la columna **Valor**. Los números usan coma decimal; las métricas van con
tres decimales y los porcentajes con uno.

- scikit-learn 1.9.1 · Python 3.12.2 · semilla 42
- Mejor modelo: **Extra Trees** · Modelo asignado: **Naive Bayes gaussiano**

## 1. Marcadores

| Marcador | Valor | Cómo se calculó |
|---|---|---|
| `MEJOR` | Extra Trees | modelo con mayor exactitud balanceada en prueba, sin contar el trivial |
| `EN_BEST` | Extra Trees | el mismo nombre en inglés, para la portada |
| `BA_MEJOR` | 0,895 | exactitud balanceada del mejor modelo en 2020-2024 |
| `IC_MEJOR` | 0,875–0,916 | percentiles 2,5 y 97,5 de 1000 remuestreos bootstrap de las 900 filas de prueba |
| `F1_MEJOR` | 0,903 | F1 macro del mejor modelo en prueba |
| `BA_NB` | 0,611 | exactitud balanceada de Naive Bayes final en prueba |
| `IC_NB` | 0,579–0,643 | bootstrap de 1 000 remuestreos, igual que el mejor modelo |
| `BA_NB_VAL` | 0,660 | exactitud balanceada de Naive Bayes en validación (2016-2019) |
| `VARIANTE_NB` | sin transformación con priors de los datos (var_smoothing = 1e-02) | combinación de la rejilla de Naive Bayes con mayor exactitud balanceada en validación |
| `N_075` | Ocho | modelos (de 13, sin contar el trivial) con exactitud balanceada en prueba >= 0,75 |
| `BA_TRIVIAL` | 0,333 | exactitud balanceada en prueba del clasificador que siempre predice la etapa más frecuente |
| `MCNEMAR` | es significativa | prueba exacta de McNemar con alfa = 0,05 sobre los aciertos discordantes (b01 = 290, b10 = 28) |
| `P_MCNEMAR` | < 0,001 | p-valor de la prueba exacta de McNemar |
| `REC_AVA_NB` | 65,5 % | recall de la etapa avanzada en prueba, Naive Bayes |
| `REC_AVA_MEJOR` | 86,4 % | recall de la etapa avanzada en prueba, mejor modelo |
| `CONFUSION_PRINCIPAL` | avanzada predicha como intermedia (12,6 % de las avanzadas) | celda fuera de la diagonal más grande de la matriz de confusión normalizada por fila |
| `PROF_OPT` | 11 | profundidad del árbol con mayor exactitud balanceada en validación |
| `K_OPT` | 1 | número de vecinos con mayor EB en validación |
| `C_OPT` | 100 | C de la SVM RBF con mayor EB en validación |
| `BRECHA_NB` | -2,8 | (EB en entrenamiento - EB en validación) x 100 con todo el entrenamiento, Naive Bayes |
| `BRECHA_MEJOR` | +12,3 | lo mismo para el mejor modelo |
| `LL_NB` | 1,068 | pérdida logarítmica en prueba de Naive Bayes |
| `LL_RL` | 0,690 | pérdida logarítmica en prueba de la regresión logística |
| `LL_MEJOR` | 0,337 | pérdida logarítmica en prueba del mejor modelo |
| `EXT_NB` | 6,8 % | filas de prueba con probabilidad máxima > 0,99, Naive Bayes |
| `EXT_RL` | 2,1 % | lo mismo para la regresión logística |
| `TOP3_MEJOR` | Energía renovable (%), Región: Europa y Asia central y Región: África subsahariana | tres variables con mayor caída de la EB al permutarlas (10 repeticiones) en el mejor modelo |
| `TOP3_NB` | Energía renovable (%), Industria (% del PIB) y Población urbana (%) | lo mismo para Naive Bayes |
| `ERR_UMBRAL` | 54,9 % | errores del mejor modelo a menos de 5 puntos de los umbrales 30 %, 60 % o 90 % (sobre 82 errores) |
| `PRUEBA_UMBRAL` | 16,1 % | mismo criterio, calculado sobre las 900 filas de prueba |
| `REGION_PEOR` | Asia oriental y el Pacífico (0,852) | región con menor exactitud del mejor modelo en prueba |
| `GAN_KNN` | -3,8 | (EB de k-NN estandarizado - EB de k-NN sin escalar) x 100, en validación |
| `PCA_2` | 60,7 % | varianza explicada por las dos primeras componentes de los 11 indicadores continuos |
| `PCA_80` | 5 | componentes necesarias para llegar al 80 % de varianza |
| `NMF_C1` | CO2 per cápita (log) y Energía per cápita (log) | las dos variables con mayor peso en el primer componente de NMF |
| `NMF_C2` | Energía renovable (%) y Variación renovable de 3 años (p.p.) | lo mismo para el segundo componente |
| `K_SIL` | 2 | k con la mejor silueta en k-Means (k de 2 a 8), países de 2024 |
| `ARI_KM3` | 0,028 | índice de Rand ajustado de k-Means con k = 3 frente a las etapas reales |
| `ARI_WARD` | 0,036 | lo mismo para el jerárquico de Ward con k = 3 |
| `ARI_DBSCAN` | -0,002 | lo mismo para DBSCAN |
| `RUIDO_DBSCAN` | 11 | países que DBSCAN marcó como ruido |
| `CUMPLE` | cumple | el mejor modelo frente al criterio de éxito de 0,75 |
| `FRASE_COMPARACION` | Los tres mejores fueron Extra Trees, Bosque aleatorio y Hist. Gradient Boosting, los tres ensambles de árboles; el más bajo de los trece fue Naive Bayes, con 0,611. | frase redactada con los resultados reales de esta corrida |
| `FRASE_VALIDACION` | En nuestros datos el árbol llega al máximo de validación con profundidad 11 y después sigue mejorando en entrenamiento mientras la validación baja, que es sobreajuste; con profundidad 1 o 2 los dos aciertos son bajos, que es subajuste. En k-NN no hay punto intermedio: gana k = 1, porque el vecino más cercano de un país es él mismo en otro año en el 82,9 % de las filas de validación, de modo que un solo vecino equivale a repetir su última etapa conocida. | frase redactada con los resultados reales de esta corrida |
| `FRASE_APRENDIZAJE` | En Naive Bayes la brecha es incluso negativa: no llega a sobreajustar, le falta capacidad, mientras que Extra Trees memoriza el entrenamiento y, aun así, generaliza mucho mejor. En los dos, la curva de validación casi no sube en la última mitad, así que más años del mismo tipo no cambiarían mucho el resultado. | frase redactada con los resultados reales de esta corrida |
| `FRASE_CALIBRACION` | Naive Bayes quedó peor calibrado que la regresión logística: su pérdida logarítmica es 1,068 frente a 0,690, y se declara casi seguro (6,8 % de las filas por encima de 0,99) muchas más veces que ella (2,1 %). | frase redactada con los resultados reales de esta corrida |
| `FRASE_ESCALADO` | A la SVM RBF el escalado le cambia la vida (+16,6 puntos al estandarizar); a k-NN con k = 15, en cambio, le sienta mal (-3,8 puntos): sin escalar, la distancia la deciden las dos variables de mayor escala, energía renovable (%) y población urbana (%), que además están entre las más informativas, y al estandarizar las once indicadoras 0/1 de región y de ingreso pasan a pesar igual que ellas. Lo comprobamos repitiendo k-NN estandarizado solo con los once indicadores continuos: sube a 0,727. En el bosque aleatorio la diferencia es de redondeo: entre un preprocesamiento y otro cambia de etapa como mucho 1 de las 720 filas de validación. | frase redactada con los resultados reales de esta corrida |
| `FRASE_PCA_TSNE` | las tres etapas se superponen: se ve un gradiente de desarrollo económico, pero los países avanzados no forman una zona propia, porque la etiqueta depende de la matriz eléctrica y esa variable no está entre los once indicadores. | frase redactada con los resultados reales de esta corrida |
| `FRASE_NB_DESEMPENO` | Con esa sencillez llegó a 0,611 de exactitud balanceada en prueba, 28,5 puntos por debajo de Extra Trees, y entrenarlo tomó menos de una centésima de segundo. | frase redactada con los resultados reales de esta corrida |
| `EFECTO_VARIANTES_NB` | no ayudaron: la variante sin transformación y con los priors de los datos siguió siendo la mejor en validación (0,660), y las otras tres se quedaron entre 0,620 y 0,630 | frase redactada con los resultados reales de esta corrida |
| `FRASE_VENTAJAS` | En nuestros datos el orden quedó claro: los ensambles de árboles ganan en exactitud y además dan las probabilidades más creíbles (Extra Trees tiene la menor pérdida logarítmica), mientras que k-NN es el más barato de entrenar. | frase redactada con los resultados reales de esta corrida |
| `FRASE_CRITERIO` | Con 0,895 de exactitud balanceada, Extra Trees cumple el criterio de 0,75 que nos fijamos, y reconoce el 86,4 % de los países avanzados. | frase redactada con los resultados reales de esta corrida |
| `FRASE_CLUSTER` | el agrupamiento apenas reproduce las etapas (el mejor índice de Rand ajustado es 0,044) y DBSCAN, con el radio que calculamos, juntó a casi todos los países en un solo grupo y dejó 11 como ruido, de modo que ni siquiera tiene silueta que informar; la etiqueta no se puede reemplazar por grupos encontrados sin ella. | frase redactada con los resultados reales de esta corrida |
| `FRASE_CONCLUSION_NB` | Es un resultado inferior al de Extra Trees, con un entrenamiento de menos de una centésima de segundo; su límite está en el supuesto de independencia y en unas probabilidades mal calibradas. | frase redactada con los resultados reales de esta corrida |

## 2. Tabla IV · Desempeño en validación (2016-2019) y prueba (2020-2024)

| Modelo | EB-V | EB-P | F1-M | Exac. | LL | AUC |
|---|---|---|---|---|---|---|
| Naive Bayes (base) | 0,660 | 0,611 | 0,603 | 0,618 | 1,068 | 0,765 |
| Regresión logística | 0,724 | 0,717 | 0,691 | 0,708 | 0,690 | 0,862 |
| k-NN | 0,812 | 0,790 | 0,800 | 0,813 | 2,822 | 0,921 |
| Árbol de decisión | 0,795 | 0,773 | 0,783 | 0,799 | 7,249 | 0,833 |
| Bosque aleatorio | 0,861 | 0,875 | 0,879 | 0,886 | 0,367 | 0,966 |
| Extra Trees | 0,877 | 0,895 | 0,903 | 0,909 | 0,337 | 0,972 |
| Gradient Boosting | 0,808 | 0,813 | 0,821 | 0,830 | 0,434 | 0,946 |
| Hist. Gradient Boosting | 0,872 | 0,872 | 0,875 | 0,882 | 0,363 | 0,966 |
| AdaBoost | 0,663 | 0,666 | 0,675 | 0,689 | 1,066 | 0,856 |
| SVM RBF | 0,845 | 0,829 | 0,838 | 0,850 | 0,467 | 0,937 |
| SVM lineal | 0,754 | 0,729 | 0,716 | 0,729 | 0,618 | 0,880 |
| LDA | 0,693 | 0,662 | 0,664 | 0,691 | 0,690 | 0,858 |
| Red neuronal MLP | 0,771 | 0,798 | 0,805 | 0,813 | 0,524 | 0,932 |
| Clasificador trivial | 0,333 | 0,333 | 0,211 | 0,463 | 19,343 | 0,500 |

EB-V y EB-P: exactitud balanceada en validación y en prueba; F1-M: F1 macro;
Exac.: exactitud; LL: pérdida logarítmica; AUC: área bajo la curva ROC, una contra
el resto, promedio macro. LL, F1-M, Exac. y AUC se miden en prueba.

## 3. Tabla V · Variantes de Naive Bayes gaussiano

| Transformación | Priors | EB-V | EB-P | LL |
|---|---|---|---|---|
| Ninguna | De los datos | 0,660 | 0,611 | 1,068 |
| Ninguna | Iguales | 0,626 | 0,633 | 2,298 |
| Yeo-Johnson | De los datos | 0,630 | 0,659 | 2,664 |
| Yeo-Johnson | Iguales | 0,620 | 0,628 | 2,847 |

## 4. Tabla VI · Exactitud en prueba por región

| Región | Filas | Naive Bayes | Mejor modelo |
|---|---|---|---|
| África subsahariana | 220 | 0,641 | 0,936 |
| América Latina y el Caribe | 170 | 0,629 | 0,953 |
| América del Norte | 15 | 0,667 | 1,000 |
| Asia oriental y el Pacífico | 135 | 0,541 | 0,852 |
| Asia meridional | 30 | 0,400 | 0,900 |
| Europa y Asia central | 230 | 0,565 | 0,857 |
| Medio Oriente y norte de África | 100 | 0,830 | 0,960 |

## 5. Tabla VII · Exactitud balanceada en validación según el preprocesamiento

| Preprocesamiento | k-NN (k = 15) | SVM RBF (C = 10) | Bosque aleat. |
|---|---|---|---|
| Sin escalar | 0,719 | 0,652 | 0,836 |
| Estandarizado | 0,681 | 0,818 | 0,839 |
| Mínimo-máximo | 0,684 | 0,804 | 0,836 |
| Robusto | 0,654 | 0,798 | 0,836 |
| Yeo-Johnson | 0,711 | 0,823 | 0,839 |

## 6. Tabla VIII · Agrupamiento de los 180 países en 2024 frente a las etapas

| Método | Grupos | Ruido | Silueta | ARI | NMI |
|---|---|---|---|---|---|
| k-Means (k = 3) | 3 | 0 | 0,242 | 0,028 | 0,059 |
| k-Means (mejor silueta) | 2 | 0 | 0,328 | 0,044 | 0,055 |
| Jerárquico Ward (k = 3) | 3 | 0 | 0,218 | 0,036 | 0,062 |
| DBSCAN | 1 | 11 | - | -0,002 | 0,033 |

ARI: índice de Rand ajustado; NMI: información mutua normalizada. DBSCAN se ejecutó con eps = 2,75 y min_samples = 5; formó un solo grupo y marcó 11 países como ruido, y la silueta necesita al menos dos grupos, por eso va un guion en esa celda.

## 7. Figuras

Las nueve figuras están en `resultados/figuras/` con el nombre que las cita el
artículo (fig02 a fig10), a 3,35 pulgadas de ancho y 300 ppp. La Fig. 1 del
artículo es un esquema hecho a mano y no la genera este programa.

## 8. Comprobaciones que respaldan las frases

Resultados que no son marcadores del artículo, pero que sostienen lo que dicen
`FRASE_ESCALADO` y `FRASE_VALIDACION`. Están en la consola de `main.py`.

| Comprobación | Valor | Para qué |
|---|---|---|
| Variables que dominan la distancia sin escalar | Energía renovable (%) y Población urbana (%) | explican por qué k-NN sin escalar acierta más: son las de mayor desviación típica en los datos crudos |
| k-NN (k = 15) estandarizado, solo con los 11 indicadores continuos | 0,727 | frente a 0,681 con los 23 predictores: confirma que las once indicadoras 0/1 son las que estorban |
| Filas de validación en las que el bosque aleatorio cambia de etapa entre escalados | 1 de 720 | la diferencia entre escalados es redondeo de los puntos de corte, no un efecto real |
| Filas de validación cuyo vecino más cercano es el mismo país en otro año | 82,9 % | explica que la curva de validación de k-NN gane con k = 1 |

## 9. Ajustes sugeridos al texto fijo del artículo

Tres frases del artículo afirman algo un poco más fuerte de lo que sostienen
nuestros números. Son cambios de una o dos palabras.

| Dónde | Dice | Proponemos |
|---|---|---|
| Sección IV-F | «El bosque aleatorio da **el mismo** resultado con todos los escalados» | «El bosque aleatorio da **prácticamente el mismo** resultado con todos los escalados» (cambia 1 de 720 filas) |
| Tabla VIII, fila DBSCAN, columna Silueta | una cifra | un guion (–), porque DBSCAN formó un solo grupo y la silueta necesita al menos dos |
| Pie de la Fig. 2 | «con intervalo de confianza del 95 %» | «con intervalo de confianza del 95 % en Naive Bayes, el mejor modelo y la regresión logística» (son los tres para los que se calculó) |
