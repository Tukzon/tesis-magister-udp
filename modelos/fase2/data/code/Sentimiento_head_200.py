# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:53:54 2024

@author: benja
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:36:39 2024

@author: benja
"""

import pandas as pd
from pysentimiento import create_analyzer
import time

# Función para calcular el sentimiento de un texto
def calculate_sentiment(analyzer, text):
    '''
    Calcula el sentimiento de un texto utilizando un analizador de sentimientos.
    
    Parámetros:
    analyzer (SentimentAnalyzer): El analizador de sentimientos.
    text (str): El texto a analizar.
    '''
    resultado = analyzer.predict(text)  # No es necesario preprocess_tweet si ya has limpiado el texto
    return resultado.output  # Devuelve el sentimiento (positivo, negativo, neutro)

# Función para calcular el sentimiento de un DataFrame
def calculate_sentiment_df(df, analyzer):
    '''
    Calcula el sentimiento de un DataFrame de Pandas utilizando un analizador de sentimientos.
    
    Parámetros:
    df (DataFrame): El DataFrame de Pandas que contiene los datos.
    analyzer (SentimentAnalyzer): El analizador de sentimientos.
    '''
    total_rows = len(df)
    df['Sentimiento'] = None  # Inicializa la columna de sentimiento

    start_time = time.time()  # Inicia el temporizador
    
    # Calcula el sentimiento de cada noticia
    for index in range(total_rows):
        df.at[index, 'Sentimiento'] = calculate_sentiment(analyzer, df.at[index, 'cuerpo_limpio'])

        # Calcula el tiempo transcurrido
        elapsed_time = time.time() - start_time
        # Calcula el tiempo promedio por fila
        time_per_row = elapsed_time / (index + 1)
        
        # Imprime el progreso
        print(f"Procesando fila {index + 1}/{total_rows} - Tiempo total: {elapsed_time:.2f}s - Tiempo por fila: {time_per_row:.2f}s")

    return df

# Mapeo de sentimientos a valores numéricos
def map_sentiment_to_numeric(sentiment):
    if sentiment == 'POS':
        return 1
    elif sentiment == 'NEU':
        return 0
    elif sentiment == 'NEG':
        return -1
    else:
        print(f"Valor inesperado encontrado: {sentiment}")  # Agrega una línea para depuración
        return None  # Maneja cualquier otro valor inesperado

# Función para calcular el sentimiento promedio diario
def sentimiento_promedio_diario(df):
    '''
    Calcula el sentimiento promedio diario de un DataFrame de Pandas.
    
    Parámetros:
    df (DataFrame): El DataFrame de Pandas que contiene los datos.
    '''
    return df.groupby('fecha')['Sentimiento'].mean()

# Función para calcular el sentimiento máximo diario
def sentimiento_maximo_diario(df):
    '''
    Calcula el sentimiento máximo diario de un DataFrame de Pandas.
    
    Parámetros:
    df (DataFrame): El DataFrame de Pandas que contiene los datos.
    '''
    return df.groupby('fecha')['Sentimiento'].max()

# Función para calcular el sentimiento mínimo diario
def sentimiento_minimo_diario(df):
    '''
    Calcula el sentimiento mínimo diario de un DataFrame de Pandas.
    
    Parámetros:
    df (DataFrame): El DataFrame de Pandas que contiene los datos.
    '''
    return df.groupby('fecha')['Sentimiento'].min()

# Función principal
if __name__ == "__main__":
    print("*" * 40)
    print("Analizando sentimiento")
    
    # Cargar el archivo CSV de noticias
    try:
        df_noticias = pd.read_csv('C:\\Users\\benja\\Desktop\\para tesis\\output\\noticias_filtradas.csv')
    except FileNotFoundError:
        print("Error: El archivo CSV no fue encontrado.")
        exit()

    # Crear el analizador de sentimiento para español
    analyzer = create_analyzer(task="sentiment", lang="es")

    # Tomar las primeras 200 filas
    df_noticias_sample = df_noticias.head(200)

    # Analizar el sentimiento de las noticias
    df_sentimiento = calculate_sentiment_df(df_noticias_sample, analyzer)
    
    print("Sentimiento analizado")
    print("*" * 40)

    # Verificar valores únicos en la columna Sentimiento
    print("Valores únicos en la columna Sentimiento antes de la conversión:")
    print(df_sentimiento['Sentimiento'].unique())

    # Convertir sentimientos a valores numéricos
    df_sentimiento['Sentimiento'] = df_sentimiento['Sentimiento'].apply(map_sentiment_to_numeric)

    # Calcular el sentimiento promedio, máximo y mínimo diarios
    print("Calculando sentimiento promedio diario")
    sentimiento_promedio = sentimiento_promedio_diario(df_sentimiento)
    
    print("Calculando sentimiento máximo diario")
    sentimiento_maximo = sentimiento_maximo_diario(df_sentimiento)
    
    print("Calculando sentimiento mínimo diario")
    sentimiento_minimo = sentimiento_minimo_diario(df_sentimiento)
    
    # Crear un DataFrame para guardar los resultados
    resultados_sentimiento = pd.DataFrame({
        'Sentimiento Promedio': sentimiento_promedio,
        'Sentimiento Máximo': sentimiento_maximo,
        'Sentimiento Mínimo': sentimiento_minimo
    }).reset_index()

    # Guardar el DataFrame como CSV en la ruta especificada
    output_path = "C:\\Users\\benja\\Desktop\\Metodologia_tesis\\tesis-magister-udp\\modelos\\fase2\\data\\output\\resultados_sentimiento.csv"
    resultados_sentimiento.to_csv(output_path, index=False)
    
    # Mostrar los resultados
    print("*" * 40)
    print("Resultados del sentimiento promedio diario:")
    print(sentimiento_promedio.head())
    
    print("Resultados del sentimiento máximo diario:")
    print(sentimiento_maximo.head())
    
    print("Resultados del sentimiento mínimo diario:")
    print(sentimiento_minimo.head())

    print(f"Resultados guardados en: {output_path}")
