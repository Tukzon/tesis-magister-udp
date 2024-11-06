# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 18:26:15 2024

@author: benja
"""

#%%

import pandas as pd
from pysentimiento import create_analyzer
#%%
# Suponiendo que df_filtrado ya está cargado
df_filtrado = pd.read_csv('C:\\Users\\benja\\Desktop\\para tesis\\output\\noticias_filtradas.csv')

# Crea un analizador de sentimiento para español (modifica 'es' por 'en' si es en inglés)
analyzer = create_analyzer(task='sentiment', lang='es')

# Función para analizar el sentimiento
def obtener_sentimiento(texto):
    resultado = analyzer.predict(texto)
    return resultado.output  # Devuelve el resultado del sentimiento (positivo, negativo, neutro)

# Aplica la función al DataFrame
df_filtrado['sentimiento'] = df_filtrado['cuerpo_limpio'].apply(obtener_sentimiento)

# Muestra el DataFrame con los resultados de sentimiento
print(df_filtrado[['cuerpo_limpio', 'sentimiento']].head(10))
#%%