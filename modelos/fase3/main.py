'''

Fase 3: Predicción

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

def attach_indicators(df_f1, df_empresa_modelo, indicators_list):
    '''
    Obtiene los indicadores técnicos del Dataframe de la fase 1 y los adjunta al Dataframe final. (USAR LOS DEL RFE)
    Verificar esta lógica, ya que cada empresa tendrá diferentes indicadores según modelo. Diferentes DF a procesar por modelo?
    '''
    for indicator in indicators_list:
        df_empresa_modelo[indicator] = df_f1[indicator]
    return df_empresa_modelo

if __name__ == "__main__":
    
    #VERIFICAR CON LOS OUTPUTS DE LA FASE 1, varios estaban de 5 y complementé
    indicadores_agua = {
        "LR": ['RSI_14','BBANDS_lower','Histogram','ATR_14','WILLR_14','TSI_13_25_13','CMO'],
        "DT": ['BBANDS_lower','Histogram','CCI_14','WILLR_14','COPP','CMO','RSI_14'],
        "XGB": ['CCI_14','WILLR_14','TSI_13_25_13','CMO','COPP','ATR_14','RSI_14'],
        "RF": ['BBANDS_lower','CCI_14','OBV','WILLR_14','COPP','CMO','RSI_14'],
        "NB": ['BBANDS_lower','CCI_14','OBV','WILLR_14','COPP','CMO','RSI_14'],
        "MLP": ['BBANDS_lower','CCI_14','OBV','WILLR_14','CMO','COPP','STCmacd_10_12_26_0.5'],
        "LSTM": []
    }
    indicadores_bci = {
        "LR": ['RSI_14','BBANDS_lower','Histogram','ATR_14','WILLR_14','TSI_13_25_13','CMO'],
        "DT": ['BBANDS_lower','Histogram','CCI_14','WILLR_14','COPP','CMO','RSI_14'],
        "XGB": ['CCI_14','WILLR_14','TSI_13_25_13','CMO','COPP','ATR_14','RSI_14'],
        "RF": ['BBANDS_lower','CCI_14','OBV','WILLR_14','COPP','CMO','RSI_14'],
        "NB": ['BBANDS_lower','CCI_14','OBV','WILLR_14','COPP','CMO','RSI_14'],
        "MLP": ['BBANDS_lower','CCI_14','OBV','WILLR_14','CMO','COPP','STCmacd_10_12_26_0.5'],
        "LSTM": []
    }
    indicadores_cap = {
        "LR": ['RSI_14','BBANDS_lower','Histogram','ATR_14','WILLR_14','TSI_13_25_13','CMO'],
        "DT": ['BBANDS_lower','Histogram','CCI_14','WILLR_14','COPP','CMO','RSI_14'],
        "XGB": ['CCI_14','WILLR_14','TSI_13_25_13','CMO','COPP','ATR_14','RSI_14'],
        "RF": ['BBANDS_lower','CCI_14','OBV','WILLR_14','COPP','CMO','RSI_14'],
        "NB": ['BBANDS_lower','CCI_14','OBV','WILLR_14','COPP','CMO','RSI_14'],
        "MLP": ['BBANDS_lower','CCI_14','OBV','WILLR_14','CMO','COPP','STCmacd_10_12_26_0.5'],
        "LSTM": []
    }
    indicadores_ccu = {
        "LR": ['RSI_14','BBANDS_lower','Histogram','ATR_14','WILLR_14','TSI_13_25_13','CMO'],
        "DT": ['BBANDS_lower','Histogram','CCI_14','WILLR_14','COPP','CMO','RSI_14'],
        "XGB": ['CCI_14','WILLR_14','TSI_13_25_13','CMO','COPP','ATR_14','RSI_14'],
        "RF": ['BBANDS_lower','CCI_14','OBV','WILLR_14','COPP','CMO','RSI_14'],
        "NB": ['BBANDS_lower','CCI_14','OBV','WILLR_14','COPP','CMO','RSI_14'],
        "MLP": ['BBANDS_lower','CCI_14','OBV','WILLR_14','CMO','COPP','STCmacd_10_12_26_0.5'],
        "LSTM": []
    }
    
    print("*" * 40)
    print("Cargando data")
    for empresa in empresas:
        df_fase1 = load_data(f"../fase1/data/Input/{empresa}.csv")
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
        print("Aplicando indicadores técnicos")
        print("-" * 40)
        print("Limpiando data")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        print("-" * 40)
        print("Exportando data")
        export_data(df, f"./data/input/{empresa}.csv")
        print("Data exportada")
        print("*" * 40)
