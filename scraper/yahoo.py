'''

En este archivo se encuentran las funciones necesarias para obtener la data de las empresas objetivo desde Yahoo Finance.

'''

import yfinance as yf

#Empresas removidas
#'ANDINA-B.SN',
#'BSANTANDER.SN',

empresas = [
    'AGUAS-A.SN',
    'BCI.SN',
    'CAP.SN',
    'CCU.SN',
    'CENCOSUD.SN'
    ]

if __name__ == "__main__":
    for empresa in empresas:
        print("*" * 40)
        print("Obteniendo data de Yahoo Finance - "+empresa)
        data = yf.download(empresa, start="2015-01-01", end="2023-12-31", interval="1d")
        print("Datos obtenidos")
        print("-" * 40)
        print("Guardando data en CSV")
        data.to_csv(f"./data/{empresa}.csv")
