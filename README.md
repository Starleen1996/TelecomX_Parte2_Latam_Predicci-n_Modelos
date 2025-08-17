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

4. **Extracción del archivo tratado (CSV)**
Realizamos la extracción del archivo tratado (CSV), este ya se encuentra limpiado y normalizado con el fin de extraer las columnas que se encontraban en diccionarios, eliminar datos nullos y vacios, cambio del tipo de columnas entre otros.
url = 'https://raw.githubusercontent.com/Starleen1996/TelecomX_Parte2_Latam_Predicci-n_Modelos/refs/heads/main/df_clientes_LATAM.csv'
