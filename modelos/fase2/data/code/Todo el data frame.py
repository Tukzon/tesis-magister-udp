# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:18:46 2024

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
    resultado = analyzer.predict(text)
    
    # Acceder a las probabilidades
    try:
        pos = resultado.probas['POS'] if 'POS' in resultado.probas else 0  # Probabilidad de positivo
        neg = resultado.probas['NEG'] if 'NEG' in resultado.probas else 0  # Probabilidad de negativo
    except TypeError:
        print(f"Error al acceder a resultado.probas: {resultado.probas}")  # Muestra el contenido de probas
        pos = 0
        neg = 0

    # Calcular la polaridad
    polaridad = pos - neg
    
    return polaridad

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

    # Analizar el sentimiento de las noticias
    print("Cargando los datos y comenzando el análisis...")
    df_sentimiento = calculate_sentiment_df(df_noticias, analyzer)
    
    print("Sentimiento analizado")
    print("*" * 40)

    # Calcular el sentimiento promedio, máximo y mínimo diarios
    print("Calculando sentimiento promedio diario")
    sentimiento_promedio = df_sentimiento.groupby('fecha')['Sentimiento'].mean()
    
    print("Calculando sentimiento máximo diario")
    sentimiento_maximo = df_sentimiento.groupby('fecha')['Sentimiento'].max()
    
    print("Calculando sentimiento mínimo diario")
    sentimiento_minimo = df_sentimiento.groupby('fecha')['Sentimiento'].min()

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
    
    
    
