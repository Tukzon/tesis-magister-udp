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
    df['Tendencia'] = df['Close'].pct_change().apply(lambda x: 1 if x > delta else (-1 if x < -delta else 0))

    return df

if __name__ == "__main__":
    print("*" * 40)
    print("Cargando data")
    for empresa in empresas:
        df = load_data(f"../../scraper/data/{empresa}.csv")
        print("Data cargada - "+empresa)
        print("-" * 40)
        print("Calculando tendencia")
        df = calculate_trend(df, delta=0.005)
        
        # Calcular múltiples indicadores técnicos
        df['SMA_10'] = ta.sma(df['Close'], length=10)
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['EMA_10'] = ta.ema(df['Close'], length=10)
        df['EMA_50'] = ta.ema(df['Close'], length=50)
        df['RSI_14'] = ta.rsi(df['Close'], length=14)
        bbands = ta.bbands(df['Close'])
        df['BBANDS_upper'] = bbands[f"BBU_5_2.0"]
        df['BBANDS_middle'] = bbands[f"BBM_5_2.0"]
        df['BBANDS_lower'] = bbands[f"BBL_5_2.0"]
        #df['STOCH_k'], df['STOCH_d'] = ta.stoch(df['High'], df['Low'], df['Close'])
        macd_df = ta.macd(df['Close'])
        df['MACD'] = macd_df['MACD_12_26_9']
        df['Signal'] = macd_df['MACDs_12_26_9']
        df['Histogram'] = macd_df['MACDh_12_26_9']
        df['ADX_14'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']
        df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['CCI_14'] = ta.cci(df['High'], df['Low'], df['Close'], length=14)
        df['OBV'] = ta.obv(df['Close'], df['Volume'])
        df['WILLR_14'] = ta.willr(df['High'], df['Low'], df['Close'], length=14)
        df['ROC_10'] = ta.roc(df['Close'], length=10)
        df['MFI_14'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14).astype('float64')
        ppo_df = ta.ppo(df['Close'])
        df[ppo_df.columns] = ppo_df
        tsi_df = ta.tsi(df['Close'])
        df[tsi_df.columns] = tsi_df
        df['UO'] = ta.uo(df['High'], df['Low'], df['Close'])
        #df['VORTEX_pos'], df['VORTEX_neg'] = ta.vortex(df['High'], df['Low'], df['Close'])
        df['KAMA'] = ta.kama(df['Close'])
        trix_df = ta.trix(df['Close'])
        df[trix_df.columns] = trix_df
        df['DPO'] = ta.dpo(df['Close'])
        df['CMO'] = ta.cmo(df['Close'])
        df['COPP'] = ta.coppock(df['Close'])
        kst_df = ta.kst(df['Close'])
        df[kst_df.columns] = kst_df
        stc_df = ta.stc(df['Close'])
        df[stc_df.columns] = stc_df
        df['PVI'] = ta.pvi(df['Close'], df['Volume'])
        df['NVI'] = ta.nvi(df['Close'], df['Volume'])
        df['ZLEMA'] = ta.zlma(df['Close'])
        df['DEMA'] = ta.dema(df['Close'])
        df['TEMA'] = ta.tema(df['Close'])
        df['WMA'] = ta.wma(df['Close'])
        df['HMA'] = ta.hma(df['Close'])
        df['VWMA'] = ta.vwma(df['Close'], df['Volume'])
        df['VWMA'] = ta.vwma(df['Close'], df['Volume'])
        df['ZLEMA'] = ta.zlma(df['Close'])  
        
        df.fillna(0, inplace=True) 
        
        print("Tendencia calculada")
        print("-" * 40)
        print("Exportando data")
        export_data(df, f"./data/Input/{empresa}.csv")
        print("Data exportada")
        print("*" * 40)
