# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 22:42:06 2024

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.optimizers import Adam
pd.set_option('display.float_format', lambda x: '%.2f' % x)

#%%


# Cargar el archivo CSV
ruta_archivo = r".\..\Input\CAP.SN.csv"
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
nueva_ruta_archivo = r'.\..\output\CAP.SN.csv'

# Guardar el DataFrame modificado en un nuevo archivo CSV
df.to_csv(nueva_ruta_archivo, index=False)

print(f"Archivo guardado en: {nueva_ruta_archivo}")

#%%


ruta_archivo = r".\..\output\CAP.SN.csv"

data_CAP_SN = pd.read_csv(ruta_archivo)

data_CAP_SN.head()
data_CAP_SN.columns
#Ahora a este archivo le haremos la estadistica descriptiva: 
    
# Estadísticas descriptivas por columnas
open_stats = data_CAP_SN['Open'].describe()
high_stats = data_CAP_SN['High'].describe()
low_stats = data_CAP_SN['Low'].describe()
close_stats = data_CAP_SN['Close'].describe()
adj_close_stats = data_CAP_SN['Adj Close'].describe()
volume_stats = data_CAP_SN['Volume'].describe()
tendencia_stats = data_CAP_SN['Tendencia'].describe()
macd_stats = data_CAP_SN['MACD_12_26_9'].describe()
macdh_stats = data_CAP_SN['MACDh_12_26_9'].describe()
macds_stats = data_CAP_SN['MACDs_12_26_9'].describe()
rsi_stats = data_CAP_SN['RSI'].describe()

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

num_cols = data_CAP_SN.select_dtypes(include=np.number).columns.tolist()

plt.figure(figsize=(15, 15))

for i, variable in enumerate(num_cols):
    plt.subplot(4, 3, i + 1)
    sns.boxplot(data=data_CAP_SN, x=variable)
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
X = data_CAP_SN[features]
y = data_CAP_SN['Tendencia']

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
X = data_CAP_SN[features]
y = data_CAP_SN['Tendencia']

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
X = data_CAP_SN[features]
y = data_CAP_SN['Tendencia']

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

# Random Forest

# Selección de características y etiqueta
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'RSI']
X = data_CAP_SN[features]
y = data_CAP_SN['Tendencia']

# Codificación de las etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Normalización de los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División de los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Construcción del modelo Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenamiento del modelo
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Evaluación del modelo
accuracy = accuracy_score(y_test, y_pred)

# Asegúrate de que los nombres de las clases sean cadenas
class_names = [str(cls) for cls in label_encoder.classes_]

# Generar el reporte de clasificación
report = classification_report(y_test, y_pred, target_names=class_names)

print(f"Precisión del modelo: {accuracy:.2f}")
print("Reporte de clasificación para Random Forest:\n", report)

#%%

# Naive Bayes

# Selección de características y etiqueta
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'RSI']
X = data_CAP_SN[features]
y = data_CAP_SN['Tendencia']

# Codificación de las etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Normalización de los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División de los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Construcción del modelo Naive Bayes
model = GaussianNB()

# Entrenamiento del modelo
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Evaluación del modelo
accuracy = accuracy_score(y_test, y_pred)

# Asegúrate de que los nombres de las clases sean cadenas
class_names = [str(cls) for cls in label_encoder.classes_]

# Generar el reporte de clasificación
report = classification_report(y_test, y_pred, target_names=class_names)

print(f"Precisión del modelo: {accuracy:.2f}")
print("Reporte de clasificación para Naive Bayes:\n", report)

#%%

# MLP (Multi-Layer Perceptron)

# Selección de características y etiqueta
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'RSI']
X = data_CAP_SN[features]
y = data_CAP_SN['Tendencia']

# Codificación de las etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Normalización de los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División de los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Construcción del modelo MLP
model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)

# Entrenamiento del modelo
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Evaluación del modelo
accuracy = accuracy_score(y_test, y_pred)

# Asegúrate de que los nombres de las clases sean cadenas
class_names = [str(cls) for cls in label_encoder.classes_]

# Generar el reporte de clasificación
report = classification_report(y_test, y_pred, target_names=class_names)

print(f"Precisión del modelo: {accuracy:.2f}")
print("Reporte de clasificación para MLP:\n", report)


#%%

# LSTM (Long Short-Term Memory)

# Selección de características y etiqueta
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'RSI']
X = data_CAP_SN[features]
y = data_CAP_SN['Tendencia']

# Codificación de las etiquetas para un problema multicategoría
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_encoded = np.eye(len(label_encoder.classes_))[y_encoded]  # One-hot encoding

# Normalización de los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Conversión de datos a formato secuencial para LSTM
n_input = 20  # Aumentamos los pasos de tiempo a 20
n_features = X.shape[1]

# Crear el generador de secuencias para el entrenamiento
train_generator = TimeseriesGenerator(X_scaled, y_encoded, length=n_input, batch_size=32)

# Construcción del modelo LSTM mejorado
model = Sequential()
model.add(LSTM(100, activation='tanh', return_sequences=True, input_shape=(n_input, n_features)))
model.add(Dropout(0.2))
model.add(LSTM(50, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))  # Cambiado a softmax para clasificación multicategoría
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento del modelo
model.fit(train_generator, epochs=50)

# Crear el generador de secuencias para la prueba
test_generator = TimeseriesGenerator(X_scaled[-(len(X_test) + n_input):], y_encoded[-(len(X_test) + n_input):], length=n_input, batch_size=1)

# Predicción
y_pred = model.predict(test_generator)

# Convertir las predicciones a clases
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_aligned = np.argmax(y_encoded[-len(y_pred_classes):], axis=1)

# Convertir las clases a strings
target_names = [str(cls) for cls in label_encoder.classes_]

# Evaluación del modelo
accuracy = accuracy_score(y_test_aligned, y_pred_classes)
report = classification_report(y_test_aligned, y_pred_classes, target_names=target_names)
conf_matrix = confusion_matrix(y_test_aligned, y_pred_classes)

print(f"Precisión del modelo: {accuracy:.2f}")
print("Reporte de clasificación para LSTM:\n", report)
print("Matriz de confusión:\n", conf_matrix)
# %%




