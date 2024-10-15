'''

Fase 2: Análisis de sentimiento

'''
import os
import sys
import pandas as pd
import numpy as np
import pysentimiento
from pysentimiento.preprocessing import preprocess_tweet
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from scraper.yahoo import empresas

def load_data(csv_path):
    '''
    Carga los datos de un archivo CSV y los retorna en un DataFrame de Pandas.
    '''
    return pd.read_csv(csv_path)

def export_data(df, csv_path):
    '''
    Exporta los datos de un DataFrame de Pandas a un archivo CSV.
    '''
    df.to_csv(csv_path)

def calculate_trend(df, delta=0.01):
    '''
    Calcula la tendencia de los precios de cierre de un DataFrame de Pandas.
    La tendencia será:
    -1 si el precio actual es menor que el del periodo anterior fuera del delta,
    0 si es igual o dentro del delta porcentual,
    1 si es mayor fuera del delta.
    
    Parámetros:
    df (DataFrame): El DataFrame de Pandas que contiene los datos.
    delta (float): El delta porcentual para considerar los precios como iguales. Por defecto es 1% (0.01).
    '''
    # Calcula la diferencia porcentual entre el precio actual y el del periodo anterior
    df['Tendencia'] = df['Adj Close'].pct_change().apply(lambda x: 1 if x > delta else (-1 if x < -delta else 0))

    return df

def calculate_sentiment(analyzer, text):
    '''
    Calcula el sentimiento de un texto utilizando un analizador de sentimientos.
    
    Parámetros:
    analyzer (SentimentAnalyzer): El analizador de sentimientos.
    text (str): El texto a analizar.
    '''
    return analyzer.predict(preprocess_tweet(text, lang='es'))

def data_to_df(data):
    '''
    TODO: Convierte la fuente de datos de las noticias (cuando se tengan a futuro) a un DataFrame de Pandas.
    '''
    return pd.DataFrame(data)

def calculate_sentiment_df(df, analyzer):
    '''
    Calcula el sentimiento de un DataFrame de Pandas utilizando un analizador de sentimientos.
    
    Parámetros:
    df (DataFrame): El DataFrame de Pandas que contiene los datos.
    analyzer (SentimentAnalyzer): El analizador de sentimientos.
    '''
    # Calcula el sentimiento de cada noticia
    df['Sentimiento'] = df['cuerpo_limpio'].apply(lambda x: calculate_sentiment(analyzer, x))

    return df

def sentimiento_promedio_diario(df):
    '''
    Calcula el sentimiento promedio diario de un DataFrame de Pandas.
    
    Parámetros:
    df (DataFrame): El DataFrame de Pandas que contiene los datos.
    '''
    # Agrupa las noticias por fecha y calcula el promedio del sentimiento
    return df.groupby('fecha')['Sentimiento'].mean()

def sentimiento_maximo_diario(df):
    '''
    Calcula el sentimiento máximo diario de un DataFrame de Pandas.
    
    Parámetros:
    df (DataFrame): El DataFrame de Pandas que contiene los datos.
    '''
    # Agrupa las noticias por fecha y calcula el sentimiento máximo
    return df.groupby('fecha')['Sentimiento'].max()

def sentimiento_minimo_diario(df):
    '''
    Calcula el sentimiento mínimo diario de un DataFrame de Pandas.
    
    Parámetros:
    df (DataFrame): El DataFrame de Pandas que contiene los datos.
    '''
    # Agrupa las noticias por fecha y calcula el sentimiento mínimo
    return df.groupby('fecha')['Sentimiento'].min()

if __name__ == "__main__":
    print("*" * 40)
    print("Analizando sentimiento")
    df_noticias = load_data("../../scraper/noticias/noticias_filtradas.csv")
    analyzer = pysentimiento.SentimentAnalyzer(lang="es")
    df_sentimiento = calculate_sentiment_df(df_noticias, analyzer)
    print("Sentimiento analizado")
    print("*" * 40)
    print("-" * 40)
    print("Calculando sentimiento promedio diario")
    sentimiento_promedio = sentimiento_promedio_diario(df_sentimiento)
    print("Calculando sentimiento máximo diario")
    sentimiento_maximo = sentimiento_maximo_diario(df_sentimiento)
    print("Calculando sentimiento mínimo diario")
    sentimiento_minimo = sentimiento_minimo_diario(df_sentimiento)
    print("*" * 40)
    print("Cargando data")
    for empresa in empresas:
        df = load_data(f"../../scraper/data/{empresa}.csv")
        print("Data cargada - "+empresa)
        print("-" * 40)
        if 'Unnamed: 0' in df.columns:
            print("Eliminando la columna 'Unnamed: 0'")
            df.drop(columns=['Unnamed: 0'], inplace=True)
        print("Calculando tendencia")
        df = calculate_trend(df, delta=0.005)  
        print("Tendencia calculada")
        print("Uniendo dataframes")
        '''
        TODO: Hacer función que maneje el merge de los DataFrames de sentimientos. sin errores.
        Esto debería fallar si no hay coincidencias en las fechas, pero no se maneja el error.
        df = df.merge(sentimiento_promedio, on='date', how='left')
        df = df.merge(sentimiento_maximo, on='date', how='left')
        df = df.merge(sentimiento_minimo, on='date', how='left')
        '''
        print("-" * 40)
        print("Limpiando data")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        print("-" * 40)
        print("Exportando data")
        export_data(df, f"./data/Input/{empresa}.csv")
        print("Data exportada")
        print("*" * 40)
