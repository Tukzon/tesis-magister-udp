# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 22:25:14 2024

@author: benja
"""

#%%
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
pd.set_option('display.float_format', lambda x: '%.2f' % x)

#%%


# Cargar el archivo CSV
ruta_archivo = r"C:\Users\benja\Desktop\Metodologia_tesis\tesis-magister-udp\modelos\fase1\data\Input\ANDINA-B.SN.csv"
df = pd.read_csv(ruta_archivo)

# Seleccionar solo las columnas especificadas
columnas_requeridas = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Tendencia']
df = df[columnas_requeridas]

# Calcular el MACD
macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
df = df.join(macd)

# Calcular el RSI
df['RSI'] = ta.rsi(df['Close'], length=14)

# Reemplazar NaN con 0, esto es unicamente para trabajar la estadistica descriptiva
df.fillna(0, inplace=True)

# Ruta para guardar el archivo con el nombre deseado
nueva_ruta_archivo = r'C:\Users\benja\Desktop\Metodologia_tesis\tesis-magister-udp\modelos\fase1\data\output\ANDINA-B.SN.csv'

# Guardar el DataFrame modificado en un nuevo archivo CSV
df.to_csv(nueva_ruta_archivo, index=False)

print(f"Archivo guardado en: {nueva_ruta_archivo}")

#%%


ruta_archivo = r"C:\Users\benja\Desktop\Metodologia_tesis\tesis-magister-udp\modelos\fase1\data\output\ANDINA-B.SN.csv"

data_Andina_BSN = pd.read_csv(ruta_archivo)

data_Andina_BSN.head()
data_Andina_BSN.columns
#Ahora a este archivo le haremos la estadistica descriptiva: 
    
# Estadísticas descriptivas por columnas
open_stats = data_Andina_BSN['Open'].describe()
high_stats = data_Andina_BSN['High'].describe()
low_stats = data_Andina_BSN['Low'].describe()
close_stats = data_Andina_BSN['Close'].describe()
adj_close_stats = data_Andina_BSN['Adj Close'].describe()
volume_stats = data_Andina_BSN['Volume'].describe()
tendencia_stats = data_Andina_BSN['Tendencia'].describe()
macd_stats = data_Andina_BSN['MACD_12_26_9'].describe()
macdh_stats = data_Andina_BSN['MACDh_12_26_9'].describe()
macds_stats = data_Andina_BSN['MACDs_12_26_9'].describe()
rsi_stats = data_Andina_BSN['RSI'].describe()

# Mostramos las estadísticas descriptivas para cada columna
print("Estadísticas descriptivas para 'Open':\n", open_stats, "\n")
print("Estadísticas descriptivas para 'High':\n", high_stats, "\n")
print("Estadísticas descriptivas para 'Low':\n", low_stats, "\n")
print("Estadísticas descriptivas para 'Close':\n", close_stats, "\n")
print("Estadísticas descriptivas para 'Adj Close':\n", adj_close_stats, "\n")
print("Estadísticas descriptivas para 'Volume':\n", volume_stats, "\n")
print("Estadísticas descriptivas para 'Tendencia':\n", tendencia_stats, "\n")
print("Estadísticas descriptivas para 'MACD_12_26_9':\n", macd_stats, "\n")
print("Estadísticas descriptivas para 'MACDh_12_26_9':\n", macdh_stats, "\n")
print("Estadísticas descriptivas para 'MACDs_12_26_9':\n", macds_stats, "\n")
print("Estadísticas descriptivas para 'RSI':\n", rsi_stats, "\n")


#%%



# Libraries to help with data visualization

num_cols = data_Andina_BSN.select_dtypes(include=np.number).columns.tolist()

plt.figure(figsize=(15, 15))

for i, variable in enumerate(num_cols):
    plt.subplot(4, 3, i + 1)
    sns.boxplot(data=data_Andina_BSN, x=variable)
    plt.tight_layout(pad=2)

plt.show()

#Es importante saber como vamos a manejar los datos atipicos existentes en los indicadores tecnicos y en el volumen


#%%

#Lo que haremos ahora será ejecutar los distintos algoritmos de clasificación, sin hiperparametros y sin validación cruzada de los datos para 
#tener una visión apriori de lo que tenemos


#Regresión logistica


# Selección de características y etiqueta
# Aquí seleccionamos las características relevantes y la etiqueta 'Tendencia'
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'RSI']
X = data_Andina_BSN[features]
y = data_Andina_BSN['Tendencia']

# Normalización de los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División de los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Construcción del modelo de regresión logística (multiclase)
model = LogisticRegression(multi_class='ovr', max_iter=1000)

# Entrenamiento del modelo
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Evaluación del modelo
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Baja', 'Mantiene', 'Sube'])

print(f"Precisión del modelo: {accuracy:.2f}")
print("Reporte de clasificación para regresión logistica:\n", report)



#%%


#Arbol de decisión
# Selección de características y etiqueta
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'RSI']
X = data_Andina_BSN[features]
y = data_Andina_BSN['Tendencia']

# Normalización de los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División de los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Construcción del modelo de árbol de decisión
model = DecisionTreeClassifier(random_state=42)

# Entrenamiento del modelo
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Evaluación del modelo
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Baja', 'Mantiene', 'Sube'])

print(f"Precisión del modelo: {accuracy:.2f}")
print("Reporte de clasificación para arbol de decisión:\n", report)














#%%


# XGBoost

#Generalmente, XGBoost espera que las etiquetas sean enteros no negativos, por lo que [-1, 0, 1] debe ser transformado a [0, 1, 2].

# Selección de características y etiqueta
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'RSI']
X = data_Andina_BSN[features]
y = data_Andina_BSN['Tendencia']

# Codificación de las etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Verifica los valores únicos en las etiquetas transformadas
print("Valores únicos en 'y_encoded':", np.unique(y_encoded))
print("Nombres de las clases:", label_encoder.classes_)

# Normalización de los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División de los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Verificar tamaños de las divisiones
print("Tamaño de X_train:", X_train.shape)
print("Tamaño de y_train:", y_train.shape)
print("Tamaño de X_test:", X_test.shape)
print("Tamaño de y_test:", y_test.shape)

# Construcción del modelo XGBoost
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

# Entrenamiento del modelo
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Verifica los primeros valores de y_test y y_pred
print("Primeros valores de y_test:", y_test[:10])
print("Primeros valores de y_pred:", y_pred[:10])

# Evaluación del modelo
accuracy = accuracy_score(y_test, y_pred)

# Obtener nombres de clases desde el LabelEncoder
class_names = label_encoder.classes_

# Generar el reporte de clasificación
report = classification_report(y_test, y_pred, target_names=[str(cls) for cls in class_names])

print(f"Precisión del modelo: {accuracy:.2f}")
print("Reporte de clasificación:\n", report)



#%%





