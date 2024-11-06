# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:21:44 2024

@author: benja
"""

import pandas as pd
import matplotlib.pyplot as plt



# Cargar el archivo CSV con los resultados de sentimiento
output_path = "C:\\Users\\benja\\Desktop\\Metodologia_tesis\\tesis-magister-udp\\modelos\\fase2\\data\\output\\resultados_sentimiento.csv"
df_resultados = pd.read_csv(output_path)

# Mostrar las primeras filas del DataFrame
print("Datos cargados:")
print(df_resultados.head())

# Calcular estadísticas básicas
print("*" * 40)
print("Estadísticas básicas del Sentimiento Promedio:")
print(df_resultados['Sentimiento Promedio'].describe())

print("*" * 40)
print("Estadísticas básicas del Sentimiento Máximo:")
print(df_resultados['Sentimiento Máximo'].describe())

print("*" * 40)
print("Estadísticas básicas del Sentimiento Mínimo:")
print(df_resultados['Sentimiento Mínimo'].describe())

# Calcular promedios por columna
promedio_sentimiento_promedio = df_resultados['Sentimiento Promedio'].mean()
promedio_sentimiento_maximo = df_resultados['Sentimiento Máximo'].mean()
promedio_sentimiento_minimo = df_resultados['Sentimiento Mínimo'].mean()

print("*" * 40)
print(f"Promedio del Sentimiento Promedio: {promedio_sentimiento_promedio:.2f}")
print(f"Promedio del Sentimiento Máximo: {promedio_sentimiento_maximo:.2f}")
print(f"Promedio del Sentimiento Mínimo: {promedio_sentimiento_minimo:.2f}")


# Cargar los datos desde el CSV
data = pd.read_csv(r"C:\Users\benja\Desktop\Metodologia_tesis\tesis-magister-udp\modelos\fase2\data\output\resultados_sentimiento.csv")

# Configurar la figura para las subgráficas
plt.figure(figsize=(15, 5))

# Graficar la distribución del Sentimiento Promedio
plt.subplot(1, 3, 1)
plt.hist(data['Sentimiento Promedio'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribución del Sentimiento Promedio')
plt.xlabel('Sentimiento Promedio')
plt.ylabel('Frecuencia')
plt.axvline(data['Sentimiento Promedio'].mean(), color='red', linestyle='dashed', linewidth=1, label='Promedio')
plt.legend()

# Graficar la distribución del Sentimiento Máximo
plt.subplot(1, 3, 2)
plt.hist(data['Sentimiento Máximo'], bins=30, color='lightgreen', edgecolor='black')
plt.title('Distribución del Sentimiento Máximo')
plt.xlabel('Sentimiento Máximo')
plt.ylabel('Frecuencia')
plt.axvline(data['Sentimiento Máximo'].mean(), color='red', linestyle='dashed', linewidth=1, label='Promedio')
plt.legend()

# Graficar la distribución del Sentimiento Mínimo
plt.subplot(1, 3, 3)
plt.hist(data['Sentimiento Mínimo'], bins=30, color='salmon', edgecolor='black')
plt.title('Distribución del Sentimiento Mínimo')
plt.xlabel('Sentimiento Mínimo')
plt.ylabel('Frecuencia')
plt.axvline(data['Sentimiento Mínimo'].mean(), color='red', linestyle='dashed', linewidth=1, label='Promedio')
plt.legend()

# Ajustar el espacio entre las subgráficas
plt.tight_layout()

# Mostrar las gráficas
plt.show()


