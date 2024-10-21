import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from scraper.yahoo import empresas

def load_data(csv_path):
    '''
    Carga los datos de un archivo CSV y los retorna en un DataFrame de Pandas.
    '''
    return pd.read_csv(csv_path)

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
    
def export_data(df, csv_path):
    '''
    Exporta los datos de un DataFrame de Pandas a un archivo CSV.
    '''
    df.to_csv(csv_path, index=False)
    
def complete_timeserie(df, desde, hasta, columna_fecha):
    '''
    Agrega las fechas faltantes en un DataFrame de Pandas.
    '''
    df = df.set_index(columna_fecha)
    for fecha in pd.date_range(desde, hasta):
        # Si la fecha no está en el DataFrame, se agrega con la fecha y todas las columnas en 0
        if fecha not in df.index:
            df.loc[fecha] = 0
    df = df.sort_index().reset_index()
    return df
            
def merge_df_by_fecha(df1, df2, columna_fecha_df1, columna_fecha_df2):
    '''
    Une dos DataFrames de Pandas por la columna de fecha.
    '''
    merged_df = df1.merge(df2, left_on=columna_fecha_df1, right_on=columna_fecha_df2, how='left')
    merged_df.drop(columns=[columna_fecha_df2], inplace=True)
    return merged_df
    
    
if __name__ == '__main__':
    df_sentimiento = load_data('./data/Input/resultados_sentimiento_tendencia.csv')
    df_sentimiento['fecha'] = pd.to_datetime(df_sentimiento['fecha'])
    df_sentimiento = complete_timeserie(df_sentimiento, '2015-01-02', '2023-12-29', 'fecha')
    for empresa in empresas:
        df = load_data(f"../../scraper/data/{empresa}.csv")
        print("Data cargada - "+empresa)
        print("-" * 40)
        if 'Unnamed: 0' in df.columns:
            print("Eliminando la columna 'Unnamed: 0'")
            df.drop(columns=['Unnamed: 0'], inplace=True)
        print("Calculando tendencia")
        calculate_trend(df, delta=0.005)  
        print("Tendencia calculada")
        print("-" * 40)
        print("Uniendo dataframes")
        df['Date'] = pd.to_datetime(df['Date'])
        df = merge_df_by_fecha(df, df_sentimiento, 'Date', 'fecha')
        print("-" * 40)
        print("Limpiando data")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        print("-" * 40)
        print("Exportando data")
        export_data(df, f"./data/Input/{empresa}.csv")
        print("Data exportada")
        print("*" * 40)