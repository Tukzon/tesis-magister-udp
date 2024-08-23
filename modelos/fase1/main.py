'''

Fase 1: Indicadores técnicos

'''
import os
import sys
import pandas as pd
import pandas_ta as ta

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

def calculate_trend(df):
    '''
    Calcula la tendencia de los precios de cierre de un DataFrame de Pandas.
    La tendencia será:
    -1 si el precio actual es menor que el del periodo anterior,
    0 si es igual,
    1 si es mayor.
    '''
    # Calcula la diferencia entre el precio actual y el precio del periodo anterior
    df['Tendencia'] = df['Close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    return df

if __name__ == "__main__":
    print("*" * 40)
    print("Cargando data")
    for empresa in empresas:
        df = load_data(f"../../scraper/data/{empresa}.csv")
        print("Data cargada - "+empresa)
        print("-" * 40)
        print("Calculando tendencia")
        df = calculate_trend(df)
        
        macd_df = ta.macd(df['Close'])
        
        df['MACD'] = macd_df['MACD_12_26_9']
        df['Signal'] = macd_df['MACDs_12_26_9']
        df['Histogram'] = macd_df['MACDh_12_26_9']     
        
        print("Tendencia calculada")
        print("-" * 40)
        print("Exportando data")
        export_data(df, f"./data/{empresa}.csv")
        print("Data exportada")
        print("*" * 40)
