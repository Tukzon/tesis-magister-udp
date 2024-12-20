'''

Fase 0: Base solo con tendencia

'''
import os
import sys
import pandas as pd
import numpy as np
import pandas_ta as ta
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

if __name__ == "__main__":
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
        print("-" * 40)
        print("Exportando data")
        export_data(df, f"./data/Input/{empresa}.csv")
        print("Data exportada")
        print("*" * 40)
