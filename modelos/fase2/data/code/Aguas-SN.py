# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 21:14:25 2024

@author: benja
"""

#%% Importar librerías
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

#%% Cargar los datos
ruta_archivo = r'.\..\Input\AGUAS-A.SN.csv'
df = pd.read_csv(ruta_archivo)

# Copiar el dataframe
data_aguas_SNN = df.copy()

#%% Preprocesamiento
# Asegúrate de eliminar cualquier columna no deseada
if 'Unnamed: 0' in data_aguas_SNN.columns:
    data_aguas_SNN.drop(columns=['Unnamed: 0'], inplace=True)

# Definir características (excluyendo 'Date' y 'Tendencia')
features = [col for col in data_aguas_SNN.columns if col not in ['Date', 'Tendencia']]
X = data_aguas_SNN[features]
y = data_aguas_SNN['Tendencia']

# Normalizar los datos
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

#%% Selección de características con RFECV
# Definir modelo base para RFECV
model_rfecv = LogisticRegression(multi_class='ovr', max_iter=1000)

# Usar RFECV para seleccionar las mejores características con validación cruzada
cv = StratifiedKFold(n_splits=5)
rfecv = RFECV(estimator=model_rfecv, step=1, cv=cv, scoring='accuracy')
X_rfecv = rfecv.fit_transform(X_scaled, y)

# Mostrar las características seleccionadas
print("Características seleccionadas por RFECV:", np.array(features)[rfecv.support_])

#%% Implementar la ventana rodante
window_size = 3  # Ajusta el tamaño de la ventana
dates = pd.to_datetime(data_aguas_SNN['Date'])
train_size = int(len(dates) * 0.8)  # 80% para entrenamiento

# Acumular todas las predicciones y etiquetas verdaderas
all_y_true = []
all_y_pred = []

hora_de_inicio = datetime.now()
iteration_count = 0  # Contador de iteraciones

# Usar el 80% inicial como conjunto de entrenamiento
X_train_initial = X_rfecv[:train_size]
y_train_initial = y[:train_size]

# Iterar sobre el 20% restante usando la ventana rodante
for start in range(train_size, len(dates) - window_size):
    test_indices = (dates >= dates.iloc[start]) & (dates < dates.iloc[start + window_size])
    train_indices = dates < dates.iloc[start]

    X_train = X_rfecv[train_indices]
    y_train = y[train_indices]
    X_test = X_rfecv[test_indices]
    y_test = y[test_indices]

    # Verificar que haya suficientes clases en el conjunto de entrenamiento
    if len(np.unique(y_train)) < 2:
        continue  # Si solo hay una clase, pasa a la siguiente iteración

    if len(X_train) == 0 or len(X_test) == 0:
        continue

    # Entrenar el modelo
    model = LogisticRegression(multi_class='ovr', max_iter=1000)
    model.fit(X_train, y_train)

    # Predecir
    y_pred = model.predict(X_test)

    # Acumular las predicciones y etiquetas verdaderas
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)
    
    iteration_count += 1    

hora_de_fin = datetime.now()

#%% Evaluación
# Convertir las listas acumuladas a arrays de NumPy
all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)

# Calcular las métricas globales
overall_accuracy = accuracy_score(all_y_true, all_y_pred)
overall_precision = precision_score(all_y_true, all_y_pred, average='weighted')
overall_recall = recall_score(all_y_true, all_y_pred, average='weighted')
overall_f1 = f1_score(all_y_true, all_y_pred, average='weighted')
overall_report = classification_report(all_y_true, all_y_pred)

#%% Resultados
print(f"Accuracy general del modelo: {overall_accuracy:.2f}")
print(f"Precision global (ponderada): {overall_precision:.2f}")
print(f"Recall global (ponderado): {overall_recall:.2f}")
print(f"F1-score global (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación general:\n {overall_report}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Número total de iteraciones realizadas: {iteration_count}")

#%%
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam

# Supongamos que ya tienes tus datos cargados en `data_aguas_SNN`

# Selección de características y etiqueta
features = [col for col in data_aguas_SNN.columns if col not in ['Date', 'Tendencia']]
X = data_aguas_SNN[features].values  # Convertir a array
y = data_aguas_SNN['Tendencia'].values  # Convertir a array

# Codificación de las etiquetas para un problema multicategoría
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_encoded = np.eye(len(label_encoder.classes_))[y_encoded]  # One-hot encoding

# Normalización de los datos
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Reshape de los datos para LSTM (n_samples, time_steps, n_features)
n_samples, n_features = X_scaled.shape
X_scaled = X_scaled.reshape((n_samples, 1, n_features))  # Solo un paso de tiempo

# Construcción del modelo LSTM
model = Sequential()
model.add(LSTM(150, activation='tanh', return_sequences=True, input_shape=(1, n_features)))
model.add(Dropout(0.3))
model.add(LSTM(100, activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento del modelo
model.fit(X_scaled, y_encoded, epochs=50, batch_size=32)

# Predicción
y_pred = model.predict(X_scaled)

# Convertir las predicciones a clases
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_aligned = np.argmax(y_encoded, axis=1)

# Decodificar las predicciones para interpretarlas en términos de las clases originales
y_pred_classes_decoded = label_encoder.inverse_transform(y_pred_classes)
y_test_aligned_decoded = label_encoder.inverse_transform(y_test_aligned)

# Evaluación del modelo
accuracy = accuracy_score(y_test_aligned_decoded, y_pred_classes_decoded)
report = classification_report(y_test_aligned_decoded, y_pred_classes_decoded, target_names=label_encoder.classes_)
conf_matrix = confusion_matrix(y_test_aligned_decoded, y_pred_classes_decoded)

print(f"Precisión del modelo: {accuracy:.2f}")
print("Reporte de clasificación para LSTM:\n", report)
print("Matriz de confusión:\n", conf_matrix)
