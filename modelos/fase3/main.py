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

#VERIFICAR CON LOS OUTPUTS DE LA FASE 1, varios estaban de 5 y complementé
indicadores = {
    "AGUAS-A.SN": {
        "LR": ['RSI_14','BBANDS_lower','Histogram','ATR_14','WILLR_14','TSI_13_25_13','CMO'],
        "DT": ['BBANDS_lower','Histogram','CCI_14','WILLR_14','COPP','CMO','RSI_14'],
        "XGB": ['CCI_14','WILLR_14','TSI_13_25_13','CMO','COPP','ATR_14','RSI_14'],
        "RF": ['BBANDS_lower','CCI_14','OBV','WILLR_14','COPP','CMO','RSI_14'],
        "NB": ['BBANDS_lower','CCI_14','OBV','WILLR_14','COPP','CMO','RSI_14'],
        "MLP": ['BBANDS_lower','CCI_14','OBV','WILLR_14','CMO','COPP','STCmacd_10_12_26_0.5'],
        "LSTM": []
        },
    "BCI.SN": {
        "LR": ['RSI_14','BBANDS_lower','Histogram','ATR_14','WILLR_14','TSI_13_25_13','CMO'],
        "DT": ['BBANDS_lower','Histogram','CCI_14','WILLR_14','COPP','CMO','RSI_14'],
        "XGB": ['CCI_14','WILLR_14','TSI_13_25_13','CMO','COPP','ATR_14','RSI_14'],
        "RF": ['BBANDS_lower','CCI_14','OBV','WILLR_14','COPP','CMO','RSI_14'],
        "NB": ['BBANDS_lower','CCI_14','OBV','WILLR_14','COPP','CMO','RSI_14'],
        "MLP": ['BBANDS_lower','CCI_14','OBV','WILLR_14','CMO','COPP','STCmacd_10_12_26_0.5'],
        "LSTM": []
        },
    "CAP.SN": {
        "LR": ['RSI_14','BBANDS_lower','Histogram','ATR_14','WILLR_14','TSI_13_25_13','CMO'],
        "DT": ['BBANDS_lower','Histogram','CCI_14','WILLR_14','COPP','CMO','RSI_14'],
        "XGB": ['CCI_14','WILLR_14','TSI_13_25_13','CMO','COPP','ATR_14','RSI_14'],
        "RF": ['BBANDS_lower','CCI_14','OBV','WILLR_14','COPP','CMO','RSI_14'],
        "NB": ['BBANDS_lower','CCI_14','OBV','WILLR_14','COPP','CMO','RSI_14'],
        "MLP": ['BBANDS_lower','CCI_14','OBV','WILLR_14','CMO','COPP','STCmacd_10_12_26_0.5'],
        "LSTM": []
        },
    "CCU.SN": {
        "LR": ['RSI_14','BBANDS_lower','Histogram','ATR_14','WILLR_14','TSI_13_25_13','CMO'],
        "DT": ['BBANDS_lower','Histogram','CCI_14','WILLR_14','COPP','CMO','RSI_14'],
        "XGB": ['CCI_14','WILLR_14','TSI_13_25_13','CMO','COPP','ATR_14','RSI_14'],
        "RF": ['BBANDS_lower','CCI_14','OBV','WILLR_14','COPP','CMO','RSI_14'],
        "NB": ['BBANDS_lower','CCI_14','OBV','WILLR_14','COPP','CMO','RSI_14'],
        "MLP": ['BBANDS_lower','CCI_14','OBV','WILLR_14','CMO','COPP','STCmacd_10_12_26_0.5'],
        "LSTM": []
        },
    "CENCOSUD.SN": {
        "LR": [],
        "DT": [],
        "XGB": [],
        "RF": [],
        "NB": [],
        "MLP": [],
        "LSTM": []
        }
}

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
    print("*" * 40)
    print("Iniciando procesamiento")

    # Iterar sobre cada empresa y algoritmo
    for empresa, algoritmos in indicadores.items():
        df_fase1 = load_data(f"../fase1/data/Input/{empresa}.csv")
        df_fase2 = load_data(f"../fase2/data/input/{empresa}.csv")
        df = load_data(f"../../scraper/data/{empresa}.csv")

        # Procesar y exportar cada combinación de empresa y algoritmo
        for algoritmo, indicadores_list in algoritmos.items():
            print(f"Procesando {empresa} - {algoritmo}")

            # F1 Calcular tendencia
            df_trend = calculate_trend(df.copy(), delta=0.005)
            df_final = attach_indicators(df_fase1, df_trend, indicadores_list)
            
            # F2 analisis de sentimiento
            df_final = df_final.merge(df_fase2[['Date', 'Sentimiento Promedio', 'Sentimiento Máximo', 'Sentimiento Mínimo']],
                                    on='Date', how='left')

            # Limpiar data
            df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_final.fillna(0, inplace=True)

            # Exportar data por empresa y algoritmo
            output_path = f"./data/input/{empresa}_{algoritmo}.csv"
            export_data(df_final, output_path)
            print(f"Data exportada para {empresa} - {algoritmo} en {output_path}")

    print("Proceso completado")
    print("*" * 40)
