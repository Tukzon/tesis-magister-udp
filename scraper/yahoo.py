'''

En este archivo se encuentran las funciones necesarias para obtener la data de las empresas objetivo desde Yahoo Finance.

'''

import yfinance as yf

empresas = [
    'FALABELLA.SN', # Falabella - Discrecional
    'BCH', # Banco de Chile - Financiero
    'ENELAM.SN', # Enel Américas - Servicios Públicos
    'SQM', # SQM - Materiales Básicos
    'ANTARCHILE.SN' # Antarchile - Industrial
    #'LTM' # LATAM Airlines - Industrial [Poca data]
    ]

if __name__ == "__main__":
    for empresa in empresas:
        print("*" * 40)
        print("Obteniendo data de Yahoo Finance - "+empresa)
        data = yf.download(empresa, period="3y", interval="1d")
        print("Datos obtenidos")
        print("-" * 40)
        print("Guardando data en CSV")
        data.to_csv(f"./data/{empresa}.csv")
