#%%
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.optimizers import Adam
from collections import Counter
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.float_format', lambda x: '%.2f' % x)

#%%


# Cargar el archivo CSV
ruta_archivo = r".\..\Input\BCI.SN.csv"
df = pd.read_csv(ruta_archivo)
data_BCI_SN = df.copy()
if 'Unnamed: 0' in df.columns:
    data_BCI_SN.drop(columns=['Unnamed: 0'], inplace=True)
print("columnas totales:", len(data_BCI_SN.columns))

#%%

data_BCI_SN.head()
data_BCI_SN.columns
#Ahora a este archivo le haremos la estadistica descriptiva: 
    
# Estadísticas descriptivas por columnas
open_stats = data_BCI_SN['Open'].describe()
high_stats = data_BCI_SN['High'].describe()
low_stats = data_BCI_SN['Low'].describe()
close_stats = data_BCI_SN['Close'].describe()
adj_close_stats = data_BCI_SN['Adj Close'].describe()
volume_stats = data_BCI_SN['Volume'].describe()
tendencia_stats = data_BCI_SN['Tendencia'].describe()
macd_stats = data_BCI_SN['MACD_12_26_9'].describe()
macdh_stats = data_BCI_SN['MACDh_12_26_9'].describe()
macds_stats = data_BCI_SN['MACDs_12_26_9'].describe()
rsi_stats = data_BCI_SN['RSI'].describe()

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

num_cols = df.select_dtypes(include=np.number).columns.tolist()

plt.figure(figsize=(15, 15))

for i, variable in enumerate(num_cols):
    plt.subplot(4, 3, i + 1)
    sns.boxplot(data=df, x=variable)
    plt.tight_layout(pad=2)

plt.show()

#Es importante saber como vamos a manejar los datos atipicos existentes en los indicadores tecnicos y en el volumen


#%%

#Regresión logistica


# Selección de todas las características excepto 'Date' y 'Tendencia'
features = [col for col in data_BCI_SN.columns if col not in ['Date', 'Tendencia']]
X = data_BCI_SN[features]
y = data_BCI_SN['Tendencia']

# Normalización de los datos
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar RFE para seleccionar las mejores características
model_rfe = LogisticRegression(multi_class='ovr', max_iter=1000)
rfe = RFE(model_rfe, n_features_to_select=5)
X_rfe = rfe.fit_transform(X_scaled, y)

print("Características seleccionadas por RFE:", np.array(features)[rfe.support_])

# Implementación de la ventana rodante con la columna 'Date'
window_size = 3  # Ajusta el tamaño de la ventana según tus necesidades

dates = pd.to_datetime(data_BCI_SN['Date'])
results = []

# Acumular todas las predicciones y etiquetas verdaderas
all_y_true = []
all_y_pred = []

train_size = int(len(dates) * 0.8)

hora_de_inicio = datetime.now()

# Usar el 80% inicial como conjunto de entrenamiento
X_train_initial = X_rfe[:train_size]
y_train_initial = y[:train_size]

# Contador de iteraciones
iteration_count = 0

# Iterar sobre el 20% restante usando la ventana rodante
for start in range(train_size, len(dates) - window_size):
    test_indices = (dates >= dates.iloc[start]) & (dates < dates.iloc[start + window_size])
    train_indices = dates < dates.iloc[start]

    X_train = X_rfe[train_indices]
    y_train = y[train_indices]
    X_test = X_rfe[test_indices]
    y_test = y[test_indices]

    # Verificar que haya suficientes clases en el conjunto de entrenamiento
    if len(np.unique(y_train)) < 2:
        continue  # Si solo hay una clase, pasa a la siguiente iteración

    if len(X_train) == 0 or len(X_test) == 0:
        continue

    model = LogisticRegression(multi_class='ovr', max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Acumular las predicciones y etiquetas verdaderas
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)
    
    iteration_count += 1

hora_de_fin = datetime.now()

# Convertir las listas acumuladas a arrays de NumPy
all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)

# Calcular las métricas globales
overall_accuracy = accuracy_score(all_y_true, all_y_pred)
overall_precision = precision_score(all_y_true, all_y_pred, average='weighted')
overall_recall = recall_score(all_y_true, all_y_pred, average='weighted')
overall_f1 = f1_score(all_y_true, all_y_pred, average='weighted')
overall_report = classification_report(all_y_true, all_y_pred)

print(f"Accuracy general del modelo: {overall_accuracy:.2f}")
print(f"Precision global (ponderada): {overall_precision:.2f}")
print(f"Recall global (ponderado): {overall_recall:.2f}")
print(f"F1-score global (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación general:\n {overall_report}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Número total de iteraciones realizadas: {iteration_count}")








#%%


# Árbol de decisión

# Selección de todas las características excepto 'Date' y 'Tendencia'
features = [col for col in data_BCI_SN.columns if col not in ['Date', 'Tendencia']]
X = data_BCI_SN[features]
y = data_BCI_SN['Tendencia']

# Normalización de los datos
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar RFE para seleccionar las mejores características utilizando un Árbol de Decisión
model_rfe = DecisionTreeClassifier(random_state=42)
rfe = RFE(model_rfe, n_features_to_select=5)  # Ajusta n_features_to_select según lo desees
X_rfe = rfe.fit_transform(X_scaled, y)

print("Características seleccionadas por RFE:", np.array(features)[rfe.support_])

# Implementación de la ventana rodante con la columna 'Date'
window_size = 3  # Ajusta el tamaño de la ventana según tus necesidades

dates = pd.to_datetime(data_BCI_SN['Date'])
results = []

# Acumular todas las predicciones y etiquetas verdaderas
all_y_true = []
all_y_pred = []

# Contador de iteraciones
iteration_count = 0

# Calcular el índice para el 80% de los datos
train_size = int(len(dates) * 0.8)

hora_de_inicio = datetime.now()

# Iterar sobre el 20% restante usando la ventana rodante
for start in range(train_size, len(dates) - window_size):
    # Definir índices de entrenamiento y prueba correctamente
    train_indices = (dates < dates.iloc[start])
    test_indices = (dates >= dates.iloc[start]) & (dates < dates.iloc[start + window_size])

    # Asegurar que no haya intersección entre los conjuntos de entrenamiento y prueba
    X_train, X_test = X_rfe[train_indices], X_rfe[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    # Verificar que haya suficientes clases en el conjunto de entrenamiento
    if len(np.unique(y_train)) < 2:
        continue  # Si solo hay una clase, pasa a la siguiente iteración

    if len(X_train) == 0 or len(X_test) == 0:
        continue

    # Crear y ajustar el modelo con limitación de profundidad para evitar sobreajuste
    model = DecisionTreeClassifier(random_state=42, max_depth=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Acumular las predicciones y etiquetas verdaderas
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)
    
    # Incrementar el contador de iteraciones
    iteration_count += 1

hora_de_fin = datetime.now()

# Convertir las listas acumuladas a arrays de NumPy
all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)

# Calcular las métricas globales
overall_accuracy = accuracy_score(all_y_true, all_y_pred)
overall_precision = precision_score(all_y_true, all_y_pred, average='weighted')
overall_recall = recall_score(all_y_true, all_y_pred, average='weighted')
overall_f1 = f1_score(all_y_true, all_y_pred, average='weighted')
overall_report = classification_report(all_y_true, all_y_pred)

print(f"Accuracy general del modelo: {overall_accuracy:.2f}")
print(f"Precision global (ponderada): {overall_precision:.2f}")
print(f"Recall global (ponderado): {overall_recall:.2f}")
print(f"F1-score global (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación general:\n {overall_report}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Número total de iteraciones realizadas: {iteration_count}")


#%%


# XGBoost

# Selección de todas las características excepto 'Date' y 'Tendencia'
features = [col for col in data_BCI_SN.columns if col not in ['Date', 'Tendencia']]
X = data_BCI_SN[features]
y = data_BCI_SN['Tendencia']

# Codificación de las etiquetas (transformación de [-1, 0, 1] a [0, 1, 2])
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Normalización de los datos
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar RFE para seleccionar las mejores características utilizando XGBoost
model_rfe = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
rfe = RFE(model_rfe, n_features_to_select=5)  # Ajusta n_features_to_select según lo desees
X_rfe = rfe.fit_transform(X_scaled, y_encoded)

print("Características seleccionadas por RFE:", np.array(features)[rfe.support_])

# Implementación de la ventana rodante con la columna 'Date'
window_size = 3  # Ajusta el tamaño de la ventana según tus necesidades

dates = pd.to_datetime(data_BCI_SN['Date'])
results = []

# Acumular todas las predicciones y etiquetas verdaderas
all_y_true = []
all_y_pred = []

# Contador de iteraciones
iteration_count = 0

# Calcular el índice para el 80% de los datos
train_size = int(len(dates) * 0.8)

hora_de_inicio = datetime.now()

# Iterar sobre el 20% restante usando la ventana rodante
for start in range(train_size, len(dates) - window_size):
    test_indices = (dates >= dates.iloc[start]) & (dates < dates.iloc[start + window_size])
    train_indices = dates < dates.iloc[start]

    X_train, X_test = X_rfe[train_indices], X_rfe[test_indices]
    y_train, y_test = y_encoded[train_indices], y_encoded[test_indices]

    # Verificar que haya suficientes clases en el conjunto de entrenamiento
    if len(np.unique(y_train)) < 2:
        continue  # Si solo hay una clase, pasa a la siguiente iteración

    if len(X_train) == 0 or len(X_test) == 0:
        continue

    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Acumular las predicciones y etiquetas verdaderas
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)
    
    # Incrementar el contador de iteraciones
    iteration_count += 1

hora_de_fin = datetime.now()

# Convertir las listas acumuladas a arrays de NumPy
all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)

# Calcular las métricas globales
overall_accuracy = accuracy_score(all_y_true, all_y_pred)
overall_precision = precision_score(all_y_true, all_y_pred, average='weighted')
overall_recall = recall_score(all_y_true, all_y_pred, average='weighted')
overall_f1 = f1_score(all_y_true, all_y_pred, average='weighted')
overall_report = classification_report(all_y_true, all_y_pred, target_names=[str(cls) for cls in label_encoder.classes_])

print(f"Precisión general del modelo: {overall_accuracy:.2f}")
print(f"Precision global (ponderada): {overall_precision:.2f}")
print(f"Recall global (ponderado): {overall_recall:.2f}")
print(f"F1-score global (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación general:\n {overall_report}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Número total de iteraciones realizadas: {iteration_count}")



#%%


# Random Forest

# Selección de todas las características excepto 'Date' y 'Tendencia'
features = [col for col in data_BCI_SN.columns if col not in ['Date', 'Tendencia']]
X = data_BCI_SN[features]
y = data_BCI_SN['Tendencia']

# Codificación de las etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Normalización de los datos
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar RFE para seleccionar las mejores características utilizando Random Forest
model_rfe = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(model_rfe, n_features_to_select=5)  # Ajusta n_features_to_select según lo desees
X_rfe = rfe.fit_transform(X_scaled, y_encoded)

print("Características seleccionadas por RFE:", np.array(features)[rfe.support_])

# Implementación de la ventana rodante con la columna 'Date'
window_size = 3  # Ajusta el tamaño de la ventana según tus necesidades

dates = pd.to_datetime(data_BCI_SN['Date'])
results = []

# Acumular todas las predicciones y etiquetas verdaderas
all_y_true = []
all_y_pred = []

# Contador de iteraciones
iteration_count = 0

# Calcular el índice para el 80% de los datos
train_size = int(len(dates) * 0.8)

hora_de_inicio = datetime.now()

# Iterar sobre el 20% restante usando la ventana rodante
for start in range(train_size, len(dates) - window_size):
    test_indices = (dates >= dates.iloc[start]) & (dates < dates.iloc[start + window_size])
    train_indices = dates < dates.iloc[start]

    X_train, X_test = X_rfe[train_indices], X_rfe[test_indices]
    y_train, y_test = y_encoded[train_indices], y_encoded[test_indices]

    # Verificar que haya suficientes clases en el conjunto de entrenamiento
    if len(np.unique(y_train)) < 2:
        continue  # Si solo hay una clase, pasa a la siguiente iteración

    if len(X_train) == 0 or len(X_test) == 0:
        continue

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Acumular las predicciones y etiquetas verdaderas
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)
    
    # Incrementar el contador de iteraciones
    iteration_count += 1

hora_de_fin = datetime.now()

# Convertir las listas acumuladas a arrays de NumPy
all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)

# Calcular las métricas globales
overall_accuracy = accuracy_score(all_y_true, all_y_pred)
overall_precision = precision_score(all_y_true, all_y_pred, average='weighted')
overall_recall = recall_score(all_y_true, all_y_pred, average='weighted')
overall_f1 = f1_score(all_y_true, all_y_pred, average='weighted')
overall_report = classification_report(all_y_true, all_y_pred, target_names=[str(cls) for cls in label_encoder.classes_])

print(f"Precisión general del modelo: {overall_accuracy:.2f}")
print(f"Precision global (ponderada): {overall_precision:.2f}")
print(f"Recall global (ponderado): {overall_recall:.2f}")
print(f"F1-score global (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación general:\n {overall_report}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Número total de iteraciones realizadas: {iteration_count}")

#%%

# Naive Bayes

# Selección de todas las características excepto 'Date' y 'Tendencia'
features = [col for col in data_BCI_SN.columns if col not in ['Date', 'Tendencia']]
X = data_BCI_SN[features]
y = data_BCI_SN['Tendencia']

# Codificación de las etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Normalización de los datos
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X_scaled, y_encoded)
selected_features = np.array(features)[selector.get_support()]
print("Características seleccionadas:", selected_features)
# Implementación de la ventana rodante con la columna 'Date'
window_size = 3  # Ajusta el tamaño de la ventana según tus necesidades

dates = pd.to_datetime(data_BCI_SN['Date'])
results = []

# Acumular todas las predicciones y etiquetas verdaderas
all_y_true = []
all_y_pred = []

# Contador de iteraciones
iteration_count = 0

# Calcular el índice para el 80% de los datos
train_size = int(len(dates) * 0.8)

hora_de_inicio = datetime.now()

# Iterar sobre el 20% restante usando la ventana rodante
for start in range(train_size, len(dates) - window_size):
    test_indices = (dates >= dates.iloc[start]) & (dates < dates.iloc[start + window_size])
    train_indices = dates < dates.iloc[start]

    X_train, X_test = X_selected[train_indices], X_selected[test_indices]
    y_train, y_test = y_encoded[train_indices], y_encoded[test_indices]

    # Verificar que haya suficientes clases en el conjunto de entrenamiento
    if len(np.unique(y_train)) < 2:
        continue  # Si solo hay una clase, pasa a la siguiente iteración

    if len(X_train) == 0 or len(X_test) == 0:
        continue

    # Ahora usamos Naive Bayes para el entrenamiento y predicción
    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Acumular las predicciones y etiquetas verdaderas
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)
    
    # Incrementar el contador de iteraciones
    iteration_count += 1

hora_de_fin = datetime.now()

# Convertir las listas acumuladas a arrays de NumPy
all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)

# Calcular las métricas globales
overall_accuracy = accuracy_score(all_y_true, all_y_pred)
overall_precision = precision_score(all_y_true, all_y_pred, average='weighted')
overall_recall = recall_score(all_y_true, all_y_pred, average='weighted')
overall_f1 = f1_score(all_y_true, all_y_pred, average='weighted')
overall_report = classification_report(all_y_true, all_y_pred, target_names=[str(cls) for cls in label_encoder.classes_])

print(f"Precisión general del modelo: {overall_accuracy:.2f}")
print(f"Precision global (ponderada): {overall_precision:.2f}")
print(f"Recall global (ponderado): {overall_recall:.2f}")
print(f"F1-score global (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación general:\n {overall_report}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Número total de iteraciones realizadas: {iteration_count}")

#%%

# MLP (Multi-Layer Perceptron)

# Selección de todas las características excepto 'Date' y 'Tendencia'
features = [col for col in data_BCI_SN.columns if col not in ['Date', 'Tendencia']]
X = data_BCI_SN[features]
y = data_BCI_SN['Tendencia']

# Codificación de las etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Normalización de los datos
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X_scaled, y_encoded)
selected_features = np.array(features)[selector.get_support()]
print("Características seleccionadas:", selected_features)
# Implementación de la ventana rodante con la columna 'Date'
window_size = 3  # Ajusta el tamaño de la ventana según tus necesidades

dates = pd.to_datetime(data_BCI_SN['Date'])
results = []

# Acumular todas las predicciones y etiquetas verdaderas
all_y_true = []
all_y_pred = []

# Contador de iteraciones
iteration_count = 0

# Calcular el índice para el 80% de los datos
train_size = int(len(dates) * 0.8)

hora_de_inicio = datetime.now()

# Iterar sobre el 20% restante usando la ventana rodante
for start in range(train_size, len(dates) - window_size):
    test_indices = (dates >= dates.iloc[start]) & (dates < dates.iloc[start + window_size])
    train_indices = dates < dates.iloc[start]

    X_train, X_test = X_selected[train_indices], X_selected[test_indices]
    y_train, y_test = y_encoded[train_indices], y_encoded[test_indices]

    # Verificar que haya suficientes clases en el conjunto de entrenamiento
    if len(np.unique(y_train)) < 2:
        continue  # Si solo hay una clase, pasa a la siguiente iteración

    if len(X_train) == 0 or len(X_test) == 0:
        continue

    # Ahora usamos MLP para el entrenamiento y predicción
    model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Acumular las predicciones y etiquetas verdaderas
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)
    
    # Incrementar el contador de iteraciones
    iteration_count += 1

hora_de_fin = datetime.now()

# Convertir las listas acumuladas a arrays de NumPy
all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)

# Calcular las métricas globales
overall_accuracy = accuracy_score(all_y_true, all_y_pred)
overall_precision = precision_score(all_y_true, all_y_pred, average='weighted')
overall_recall = recall_score(all_y_true, all_y_pred, average='weighted')
overall_f1 = f1_score(all_y_true, all_y_pred, average='weighted')
overall_report = classification_report(all_y_true, all_y_pred, target_names=[str(cls) for cls in label_encoder.classes_])

print(f"Precisión general del modelo: {overall_accuracy:.2f}")
print(f"Precision global (ponderada): {overall_precision:.2f}")
print(f"Recall global (ponderado): {overall_recall:.2f}")
print(f"F1-score global (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación general:\n {overall_report}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Número total de iteraciones realizadas: {iteration_count}")


#%%

# LSTM (Long Short-Term Memory)

# Selección de características y etiqueta
features = [col for col in data_BCI_SN.columns if col not in ['Date', 'Tendencia']]
X = data_BCI_SN[features]
y = data_BCI_SN['Tendencia']

# Codificación de las etiquetas para un problema multicategoría
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_encoded = np.eye(len(label_encoder.classes_))[y_encoded]  # One-hot encoding

print("Clases originales:", label_encoder.classes_)
print("Valores codificados:", list(label_encoder.transform(label_encoder.classes_)))

print("Distribución de clases en el conjunto de entrenamiento:", Counter(np.argmax(y_encoded, axis=1)))

# Normalización de los datos
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Conversión de datos a formato secuencial para LSTM
n_input = 20  # Aumentamos los pasos de tiempo a 20
n_features = X.shape[1]

# Crear el generador de secuencias para el entrenamiento
train_generator = TimeseriesGenerator(X_scaled, y_encoded, length=n_input, batch_size=32)

# Construcción del modelo LSTM
model = Sequential()
model.add(LSTM(150, activation='tanh', return_sequences=True, input_shape=(n_input, n_features)))
model.add(Dropout(0.3))
model.add(LSTM(100, activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento del modelo
model.fit(train_generator, epochs=50)

# Crear el generador de secuencias para la prueba
test_generator = TimeseriesGenerator(X_scaled[-(len(X_scaled) + n_input):], y_encoded[-(len(X_scaled) + n_input):], length=n_input, batch_size=1)

# Predicción
y_pred = model.predict(test_generator)

# Convertir las predicciones a clases
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_aligned = np.argmax(y_encoded[-len(y_pred_classes):], axis=1)

# Decodificar las predicciones para interpretarlas en términos de las clases originales
y_pred_classes_decoded = label_encoder.inverse_transform(y_pred_classes)
y_test_aligned_decoded = label_encoder.inverse_transform(y_test_aligned)

print("Clases decodificadas en las predicciones:", np.unique(y_pred_classes_decoded))
print("Clases decodificadas en y_test_aligned:", np.unique(y_test_aligned_decoded))

# Verificar las clases únicas en las predicciones
unique_pred_classes = np.unique(y_pred_classes_decoded)
print("Clases únicas en las predicciones:", unique_pred_classes)

# Verificar las clases únicas en y_test_aligned
unique_test_classes = np.unique(y_test_aligned_decoded)
print("Clases únicas en y_test_aligned:", unique_test_classes)

# Convertir las clases a strings
target_names = [str(cls) for cls in label_encoder.classes_]

# Evaluación del modelo
accuracy = accuracy_score(y_test_aligned_decoded, y_pred_classes_decoded)
report = classification_report(y_test_aligned_decoded, y_pred_classes_decoded, target_names=target_names)
conf_matrix = confusion_matrix(y_test_aligned_decoded, y_pred_classes_decoded)

print(f"Precisión del modelo: {accuracy:.2f}")
print("Reporte de clasificación para LSTM:\n", report)
print("Matriz de confusión:\n", conf_matrix)

# %%





