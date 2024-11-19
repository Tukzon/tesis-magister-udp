# Predicción de tendencias bursátiles mediante el uso de indicadores técnicos y análisis de sentimiento en noticias financieras en español:
## Aplicación de Algoritmos de Clasificación de Machine Learning

### Resúmen Algoritmos

Cada carpeta en modelos contiene las tres diferentes fases de la investigación, así como también el baseline de referencia. Dentro de cada una de estas carpetas se encuentra un archivo main.py el cual implementa funcionalidades iniciales al conjunto de datos, los cuales son almacenados en /input , en la carpeta /code se encuentran las diferentes implementaciones de los siete modelos para las 5 diferentes empresas (con excepción de fase 3 que están todas juntas), en algunas de estas fases los resultados fueron exportados a /output mediante la exportación del notebook o fueron exportados como archivo directamente de los prints realizados en el algoritmo.

### Descripción de Carpetas y Archivos

- **modelos/**: Contiene los diferentes modelos y fases del proyecto.
  - **baseline/**: Fase inicial con datos de tendencia.
    - **data/**: Carpeta para almacenar datos.
      - **code/**: Código fuente específico de esta fase.
      - **Input/**: Datos de entrada.
      - **output/**: Datos de salida.
    - **main.py**: Script principal para la fase baseline.
  - **fase1/**: Fase de indicadores técnicos.
    - **data/**: Carpeta para almacenar datos.
      - **code/**: Código fuente específico de esta fase.
      - **Input/**: Datos de entrada.
      - **output/**: Datos de salida.
    - **main.py**: Script principal para la fase 1.
  - **fase2/**: Fase de análisis de sentimiento.
    - **data/**: Carpeta para almacenar datos.
    - **main.py**: Script principal para la fase 2.
    - **merger.py**: Script para combinar datos de diferentes fuentes.
  - **fase3/**: Fase de predicción.
    - **data/**: Carpeta para almacenar datos.
    - **main.py**: Script principal para la fase 3.
- **scraper/**: Contiene scripts para la recolección y procesamiento de datos.
  - **__init__.py**: Archivo de inicialización del módulo para invocar datos desde otros archivos.
  - **data/**: Carpeta para almacenar datos recolectados.
    - **AGUAS-A.SN.csv**, **ANDINA-B.SN.csv**, etc.: Archivos CSV con datos de diferentes empresas.
  - **df.py**: Script para scrapear del diario financier (No se usa).
  - **noticias/**: Carpeta para almacenar noticias y su procesamiento.
    - **dejar_noticias_en_esta_carpeta.txt**: Indicaciones para el uso de la carpeta.
    - **limpieza_data.ipynb**: Notebook para limpieza de datos.
    - **noticias_filtradas.csv**: Noticias filtradas.
    - **noticias.csv**: Noticias originales.
  - **yahoo.py**: Script para la recolección de datos desde Yahoo Finance.

#### Nota:
  Para la correcta ejecución de la fase 2 (para obtener los sentimientos) es necesario dejar manualmente la base de datos de noticias en la carpeta, no se puede subir al git por el tamaño.

### Scripts Principales

- **modelos/baseline/main.py**: Script principal para la fase baseline.
- **modelos/fase1/main.py**: Script principal para la fase 1 (inserción de indicadores técnicos).
- **modelos/fase2/main.py**: Script principal para la fase 2 (inserción de carácterísticas para sentimientos).
- **modelos/fase3/main.py**: Script principal para la fase 3 (merge entre f1 y f2).
- **scraper/yahoo.py**: Script para la recolección de datos desde Yahoo Finance.

### Investigación basada en:
Wang, Z., Hu, Z., Li, F. et al. Learning-Based Stock Trending Prediction by Incorporating Technical Indicators and Social Media Sentiment. Cogn Comput 15, 1092–1102 (2023). https://doi.org/10.1007/s12559-023-10125-8