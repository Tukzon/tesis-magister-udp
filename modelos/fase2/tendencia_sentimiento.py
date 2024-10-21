import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def export_data(df, csv_path):
    '''
    Exporta los datos de un DataFrame de Pandas a un archivo CSV.
    '''
    df.to_csv(csv_path)
    
def load_data(csv_path):
    '''
    Carga los datos de un archivo CSV y los retorna en un DataFrame de Pandas.
    '''
    return pd.read_csv(csv_path)

if __name__ == '__main__':
    df_sentimientos = load_data('./data/Input/resultados_sentimiento.csv')
    df_sentimientos['Tendencia Sentimiento'] = df_sentimientos['Sentimiento Promedio'].apply(lambda x: 1 if x > 0 else 0)
    print("Total de noticias positivas:", df_sentimientos['Tendencia Sentimiento'].sum())
    print("Total de noticias negativas:", len(df_sentimientos) - df_sentimientos['Tendencia Sentimiento'].sum())
    print("Porcentaje de noticias positivas:", df_sentimientos['Tendencia Sentimiento'].mean())
        
    export_data(df_sentimientos, './data/Input/resultados_sentimiento_tendencia.csv')