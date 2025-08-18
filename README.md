# TelecomX_Parte2_Latam_Predicci-n_Modelos

## INTROCUCCIÓN
La intención de realizar el proyecto es establecer modelos de predicción de **Churn** de clientes de la compañia **TelecomX_Latam** aplicando una estadistica descriptiva y aprendizaje automatico (Machine Learning). Por ende,  la compañia quiere conocer que tipo de clientes podrían permanecer o retirar los diferentes servicios que se ofrecen, para ello debemos establecer unas variables explicativas y la variable objetivo (respuesta) en nuestro modelo de Machine Learning. 

## Tabla de Contenido
* Descripción del proyecto
* Creación del repositorio en Github
* Listado de Bibliotecas a utilizar para el desarrollo del proyecto
* Extracción del archivo tratado (CSV)
* Eliminación de columnas irrelevantes
* Realizar el Encoding
* Verificación de la Proporción de Cancelación (Churn)
* Balanceo de Clases (opcional)
* Normalización o Estandarización (si es necesario)
* Correlación y Selección de Variables
* Análisis de Correlación
* Análisis Dirigido
* Modelado Predictivo
* Separación de Datos
* Creación de Modelos
* Evaluación de los Modelos
* Interpretación y Conclusiones
* Análisis de la Importancia de las Variables
* Conclusión

1. **Descripción del Proyecto**
La compañia quiere anticiparse a la problematica de cancelación de servicios, para ello debemos analizar los datos historicos sobre nuestros clientes y servicios, posteriormente debemos establecer modelos de machine learning  para comparar su efectividad en la predicción de clientes (churn). Para ello vamos a establecer modelos como: **Baseline**, **Arbol de Decisiones**, **Random Forest** y comparar su rendimiento.

2. **Creación del repositorio en Github**
Para el desarrollo del proyecto realizamos la creación del repositorio en Github. donde se encontrará disponible el proyecto y ser usado libremente por cualquier persona que tenga interes. https://github.com/Starleen1996/TelecomX_Parte2_Latam_Predicci-n_Modelos.git

3. ** Listado de Bibliotecas a utilizar para el desarrollo del proyecto**
Para el desarrollo del proyecto realizamos el uso de diferentes bibliotecas Python entre ellas:
* import requests
* import json
* import pandas as pd
* import sklearn
* import pickle
* import numpy as np
* import seaborn as sns
* import matplotlib.pyplot as plt
* import warnings
* warnings.filterwarnings('ignore')
* import plotly.express as px
* import yellowbrick
* import DecisionTreeClassifier
* import train_test_split
* from sklearn.compose import make_column_transformer  # Este modulo ayuda a realizar una transformación de columnas
* from sklearn.preprocessing import OneHotEncoder # Ayuda a realizar transformación de 1 y 0
* from sklearn.model_selection import StratifiedKFold # Aseguramos que la proporción de cada clase se mantenga
* from sklearn.preprocessing import StandardScaler
* from sklearn.linear_model import LogisticRegression
* from sklearn.pipeline import Pipeline
* from sklearn.metrics import classification_report, confusion_matrix
* from sklearn.metrics import accuracy_score, confusion_matrix
* from imblearn.over_sampling import SMOTE
* from imblearn.pipeline import Pipeline as ImbPipeline


4. **Extracción del archivo tratado (CSV)**
Realizamos la extracción del archivo tratado (CSV), este ya se encuentra limpiado y normalizado con el fin de extraer las columnas que se encontraban en diccionarios, eliminar datos nullos y vacios, cambio del tipo de columnas entre otros.
url = 'https://raw.githubusercontent.com/Starleen1996/TelecomX_Parte2_Latam_Predicci-n_Modelos/refs/heads/main/df_clientes_LATAM.csv'

5. **Eliminación de Columnas irrelevantes**
Vamos a realizar la eliminación de la columna **ID_Cliente** ya que no aporta valor al análisis o al modelo predictivo

6. **Realizar el Encoding**
En este paso realizamos la transformación de las variables explicativas, excluyendo la variable de respuesta. para este paso primero hacemos la separación de las variables explicativas y la variable de respuesta.

X = datos.drop('cliente_vigente', axis=1)
y= datos['cliente_vigente']

posteriormente realizamos la transformación de los datos (Encodig) utilizando la biblioteca sklearn.compose y el modulo make_columns_transformer, sklearn.preprocessing import OneHotEncoder.
En la transformación de datos, logramos que las variables categoricas queden de forma binaria (0 y 1), esto se hace con el procesamientos de los datos en nuestros modelos de machine learning no presenten errores o sesgos.

7. **Verificación de la Proporción de Cancelación (Churn)**
 Verificando la proporción de cancelación vemos un desbalance en la variable (clientes vigentes), para ello utilizamos el metodo **datos.value_counts('cliente_vigente')**
0 = Clientes que siguen en la compañia  (5174 Clientes)
1 = Clientes que se han retirado en la compañia (1869 Clientes)
Con el fin de que ambas clases  queden con igual proporción de los datos vamos a realizar una estratificación (balanceo) en nuestra variable objetivo o de respuesta.
8. **Balanceo de Clases (opcional)**
9. **Normalización o Estandarización (si es necesario)**
10. **Correlación y Selección de Variables**
Realizamos la correlación entre las variables númericas, entre ellas la variable objetivo (respuesta) **Churn** para verificar el resultado respecto a las variables explicativas. Para ello realizamos un filtro de las variables tipo (int64', 'float64) y posteriormente calculamos la correlación con el metodo (corr()), finalmente para tener una mejor visualización de las correlaciones, realizamos un grafico (Heatmap) con laa biblioteca Seaborn.
11. **Análisis Dirijido de Correlación:**

Número de meses_contrato vs total_pagado_cliente: 0.825
Muy fuerte correlación positiva.
Significa que cuanto más meses tiene el cliente en contrato, mayor es el total pagado.
Es esperable: más tiempo = más facturación.
    
Número de meses_contrato vs cliente_vigente: -0.352
Correlación negativa moderada.
Sugiere que a mayor número de meses en contrato, hay cierta tendencia a que el cliente ya no esté vigente (abandone el servicio).
No es muy fuerte, pero hay relación.

total_pagado_cliente vs cliente_vigente: -0.199
Correlación negativa débil.
Indica que los clientes que han pagado más, tienden ligeramente a no estar vigentes, pero la relación es débil (casi cercana a 0).

**En Resumen:**

La variable más fuerte es:
#_meses_contrato ↔ total_pagado_cliente (0.825).
Existe un patrón de rotación de clientes:
mientras más tiempo y más pagan, hay una ligera probabilidad de que ya no estén vigentes (correlaciones negativas con cliente_vigente).
Pero esas correlaciones negativas no son lo suficientemente fuertes como para sacar conclusiones absolutas; se debería complementar con otros análisis.

12. **Modelado Predictivo:**
    
Modelos recomendados para este caso

* Regresión Logística:

Punto de partida clásico para problemas de clasificación binaria.

Te da interpretabilidad (puedes ver qué variable aumenta o disminuye la probabilidad de estar vigente).

Puede verse afectada por la multicolinealidad entre meses_contrato y total_pagado_cliente.

* Árboles de Decisión:

No tienen problema con colinealidad.

Fácil de interpretar en forma de reglas (ej: "si meses_contrato > X entonces...").

* Random Forest o Gradient Boosting (XGBoost, LightGBM, CatBoost):

Muy recomendados cuando quieres mayor precisión.

Manejan bien relaciones no lineales y colinealidad.

Random Forest → bueno para empezar.

XGBoost/LightGBM → mejor rendimiento si tienes más datos.

* Redes Neuronales (opcional, si tienes muchos datos):

Podrían aplicarse, pero en un dataset pequeño o con pocas variables no aporta mucho más que un modelo de boosting.+

13. **Separación de Datos**
Realizamos la separación de los datos para entrenamiento y prueba, lo recomendable es usar el 70/80 porciento para entrenamiento y 30/20 porciento para probar el modelo, para ellos utilizamos el modulo import train_test_split de la biblioteca sklearn.modelselection.

14 - 15 **Creación y Evaluación de modelos**

**Modelo Regresión Logístico**
Uno de los modelos utilizados fue el de regresión logística, donde se evaluaron las variables explicativas y variables de respuesta.

**Resultados principales:**

Exactitud (Accuracy):

Entrenamiento: 0.95

Prueba: 0.77

➝ El modelo generaliza relativamente bien, aunque hay una caída de 0.95 → 0.77, lo que indica cierto sobreajuste (el modelo aprende muy bien los datos de entrenamiento, pero pierde rendimiento con los de prueba).

Reporte de Clasificación:

**Clase 0 (No Vigente):**

Precisión: 0.87

Recall: 0.81

F1-score: 0.84

➝ El modelo identifica bastante bien a los clientes No Vigentes, con buena precisión y recall.

**Clase 1 (Vigente):**

Precisión: 0.56

Recall: 0.66

F1-score: 0.61

➝ El desempeño es más bajo en la clase Vigente, aunque el recall de 0.66 muestra que el modelo logra recuperar 2 de cada 3 clientes vigentes. La precisión baja (0.56) indica que se generan falsos positivos (se predicen vigentes clientes que no lo son).

**Matriz de Confusión:**

1258 clientes No Vigentes bien clasificados.

294 No Vigentes mal clasificados como Vigentes.

188 Vigentes mal clasificados como No Vigentes.

373 Vigentes bien clasificados.

➝ El modelo tiende a estar más inclinado hacia predecir No Vigentes, aunque SMOTE ayudó a balancear un poco (sin SMOTE seguramente la clase Vigente habría tenido un recall aún más bajo).

🔎 Conclusiones sobre el modelo:

El balanceo con SMOTE ayudó a mejorar el recall de la clase minoritaria (Vigente), aunque todavía el rendimiento es desigual entre clases.

El accuracy general (0.77) es aceptable, pero se debe analizar con cuidado dado el desbalance original: el modelo sigue siendo mejor prediciendo la clase mayoritaria.

La Regresión Logística funciona como un modelo base que da buena interpretabilidad, pero puede que no capture relaciones complejas en tus datos.

El hecho de que el recall en "Vigente" sea mayor que la precisión significa que el modelo prefiere arriesgarse a clasificar clientes como Vigentes (aunque se equivoque), lo cual puede ser bueno si tu interés es detectar clientes que se mantendrán activos y no perderlos.
