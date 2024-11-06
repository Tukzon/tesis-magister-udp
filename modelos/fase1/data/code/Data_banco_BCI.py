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

# Selección de carácterísticas
features = [col for col in data_BCI_SN.columns if col not in ['Date', 'Tendencia']]
X = data_BCI_SN[features]
y = data_BCI_SN['Tendencia']

# Normalización de los datos
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

window_size = 3
dates = pd.to_datetime(data_BCI_SN['Date'])
train_size = int(len(dates) * 0.8)


#%%

#Regresión logistica


# Aplicar RFE para seleccionar las mejores características
model_rfe = LogisticRegression(multi_class='ovr', max_iter=1000)
rfe = RFE(model_rfe, n_features_to_select=7)
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

# Guardar los resultados en un archivo de texto
output_file_path = r'.\..\output\LR_banco_BCI.txt'
with open(output_file_path, 'w') as f:
    f.write(f"Características seleccionadas por RFE: {np.array(features)[rfe.support_]}\n\n")
    f.write(f"Accuracy general del modelo: {overall_accuracy:.2f}\n")
    f.write(f"Precision global (ponderada): {overall_precision:.2f}\n")
    f.write(f"Recall global (ponderado): {overall_recall:.2f}\n")
    f.write(f"F1-score global (ponderado): {overall_f1:.2f}\n")
    f.write(f"Reporte de clasificación general:\n{overall_report}\n")
    f.write(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}\n")
    f.write(f"Número total de iteraciones realizadas: {iteration_count}\n")

print(f"Accuracy general del modelo: {overall_accuracy:.2f}")
print(f"Precision global (ponderada): {overall_precision:.2f}")
print(f"Recall global (ponderado): {overall_recall:.2f}")
print(f"F1-score global (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación general:\n {overall_report}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Número total de iteraciones realizadas: {iteration_count}")




#%%
# Regresión logística con C=100, se probo con varios valores de C, siento este el mejor y solver=lbfgs

# Aplicar RFE para seleccionar las mejores características
model_rfe = LogisticRegression(multi_class='ovr', C=100, solver='lbfgs', max_iter=1000)
rfe = RFE(model_rfe, n_features_to_select=7)
X_rfe = rfe.fit_transform(X_scaled, y)

print("Características seleccionadas por RFE:", np.array(features)[rfe.support_])

results = []

# Acumular todas las predicciones y etiquetas verdaderas
all_y_true = []
all_y_pred = []

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

    model = LogisticRegression(multi_class='ovr', C=100, solver='lbfgs', max_iter=1000)
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

# Guardar los resultados en un archivo de texto
output_file_path = r'.\..\output\LR_banco_bci.txt'
with open(output_file_path, 'w') as f:
    f.write(f"Características seleccionadas por RFE: {np.array(features)[rfe.support_]}\n\n")
    f.write(f"Accuracy general del modelo: {overall_accuracy:.2f}\n")
    f.write(f"Precision global (ponderada): {overall_precision:.2f}\n")
    f.write(f"Recall global (ponderado): {overall_recall:.2f}\n")
    f.write(f"F1-score global (ponderado): {overall_f1:.2f}\n")
    f.write(f"Reporte de clasificación general:\n{overall_report}\n")
    f.write(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}\n")
    f.write(f"Número total de iteraciones realizadas: {iteration_count}\n")

# Imprimir resultados en consola (opcional)
print(f"Características seleccionadas por RFE: {np.array(features)[rfe.support_]}")
print(f"Accuracy general del modelo: {overall_accuracy:.2f}")
print(f"Precision global (ponderada): {overall_precision:.2f}")
print(f"Recall global (ponderado): {overall_recall:.2f}")
print(f"F1-score global (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación general:\n {overall_report}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Número total de iteraciones realizadas: {iteration_count}")



#%%


# Árbol de decisión

# Aplicar RFE para seleccionar las mejores características utilizando un Árbol de Decisión
model_rfe = DecisionTreeClassifier(random_state=42)
rfe = RFE(model_rfe, n_features_to_select=7)  # Ajusta n_features_to_select según lo desees
X_rfe = rfe.fit_transform(X_scaled, y)

print("Características seleccionadas por RFE:", np.array(features)[rfe.support_])

results = []

# Acumular todas las predicciones y etiquetas verdaderas
all_y_true = []
all_y_pred = []

# Contador de iteraciones
iteration_count = 0

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

# Guardar resultados en archivo .text
output_path = r'.\..\output\DT_banco_bci.text'
with open(output_path, 'w') as f:
    f.write(f"Resultados para el Árbol de Decisión:\n")
    f.write(f"Accuracy general del modelo: {overall_accuracy:.2f}\n")
    f.write(f"Precision global (ponderada): {overall_precision:.2f}\n")
    f.write(f"Recall global (ponderado): {overall_recall:.2f}\n")
    f.write(f"F1-score global (ponderado): {overall_f1:.2f}\n")
    f.write(f"Reporte de clasificación general:\n{overall_report}\n")
    f.write(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}\n")
    f.write(f"Número total de iteraciones realizadas: {iteration_count}\n")

# También imprimir los resultados en consola
print(f"Resultados para el Árbol de Decisión:\n")
print(f"Accuracy general del modelo: {overall_accuracy:.2f}")
print(f"Precision global (ponderada): {overall_precision:.2f}")
print(f"Recall global (ponderado): {overall_recall:.2f}")
print(f"F1-score global (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación general:\n{overall_report}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Número total de iteraciones realizadas: {iteration_count}")

#%%


# Arbol segun paper
# Aplicar RFE para seleccionar las mejores características utilizando un Árbol de Decisión
model_rfe = DecisionTreeClassifier(random_state=42)
rfe = RFE(model_rfe, n_features_to_select=10)  # Ajusta n_features_to_select según lo desees
X_rfe = rfe.fit_transform(X_scaled, y)

print("Características seleccionadas por RFE:", np.array(features)[rfe.support_])

results = []

# Acumular todas las predicciones y etiquetas verdaderas
all_y_true = []
all_y_pred = []

# Contador de iteraciones
iteration_count = 0

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

    # Crear y ajustar el modelo con los hiperparámetros especificados en el paper
    model = DecisionTreeClassifier(random_state=42)  # Hiperparámetros no especificados, puedes añadirlos si es necesario
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
confusion_mat = confusion_matrix(all_y_true, all_y_pred)

# Guardar resultados en archivo .text
output_path = r'.\..\output\DT_paper_banco:bci.text'
with open(output_path, 'w') as f:
    f.write(f"Resultados para el Árbol de Decisión:\n")
    f.write(f"Accuracy general del modelo: {overall_accuracy:.2f}\n")
    f.write(f"Precision global (ponderada): {overall_precision:.2f}\n")
    f.write(f"Recall global (ponderado): {overall_recall:.2f}\n")
    f.write(f"F1-score global (ponderado): {overall_f1:.2f}\n")
    f.write(f"Reporte de clasificación general:\n{overall_report}\n")
    f.write(f"Matriz de confusión:\n{confusion_mat}\n")
    f.write(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}\n")
    f.write(f"Número total de iteraciones realizadas: {iteration_count}\n")

# También imprimir los resultados en consola
print(f"Resultados para el Árbol de Decisión:\n")
print(f"Accuracy general del modelo: {overall_accuracy:.2f}")
print(f"Precision global (ponderada): {overall_precision:.2f}")
print(f"Recall global (ponderado): {overall_recall:.2f}")
print(f"F1-score global (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación general:\n{overall_report}")
print(f"Matriz de confusión:\n{confusion_mat}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Número total de iteraciones realizadas: {iteration_count}")



#%%


# Aplicar RFE para seleccionar las mejores características utilizando un Árbol de Decisión
model_rfe = DecisionTreeClassifier(random_state=42)
rfe = RFE(model_rfe, n_features_to_select=7)  # Ajusta n_features_to_select según lo desees
X_rfe = rfe.fit_transform(X_scaled, y)

print("Características seleccionadas por RFE:", np.array(features)[rfe.support_])

results = []
all_y_true = []
all_y_pred = []
iteration_count = 0
hora_de_inicio = datetime.now()

# Iterar sobre el 20% restante usando la ventana rodante
for start in range(train_size, len(dates) - window_size):
    train_indices = (dates < dates.iloc[start])
    test_indices = (dates >= dates.iloc[start]) & (dates < dates.iloc[start + window_size])

    X_train, X_test = X_rfe[train_indices], X_rfe[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    if len(np.unique(y_train)) < 2 or len(X_train) == 0 or len(X_test) == 0:
        continue

    # Crear y ajustar el modelo con hiperparámetros especificados
    model = DecisionTreeClassifier(
        random_state=42,
        max_depth=10,               # Profundidad máxima del árbol
        min_samples_split=5,        # Mínimo de muestras para dividir un nodo
        min_samples_leaf=3,         # Mínimo de muestras en una hoja
        max_features='sqrt'         # Máximo de características consideradas en cada división
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)
    iteration_count += 1

hora_de_fin = datetime.now()

# Convertir las listas acumuladas a arrays de NumPy
all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)

# Calcular métricas globales
overall_accuracy = accuracy_score(all_y_true, all_y_pred)
overall_precision = precision_score(all_y_true, all_y_pred, average='weighted')
overall_recall = recall_score(all_y_true, all_y_pred, average='weighted')
overall_f1 = f1_score(all_y_true, all_y_pred, average='weighted')
overall_report = classification_report(all_y_true, all_y_pred)
confusion_mat = confusion_matrix(all_y_true, all_y_pred)

# Guardar resultados en archivo .text
output_path = r'.\..\output\DT_paper_banco:bci.text'
with open(output_path, 'w') as f:
    f.write(f"Resultados para el Árbol de Decisión con Hiperparámetros Ajustados:\n")
    f.write(f"Accuracy general del modelo: {overall_accuracy:.2f}\n")
    f.write(f"Precision global (ponderada): {overall_precision:.2f}\n")
    f.write(f"Recall global (ponderado): {overall_recall:.2f}\n")
    f.write(f"F1-score global (ponderado): {overall_f1:.2f}\n")
    f.write(f"Reporte de clasificación general:\n{overall_report}\n")
    f.write(f"Matriz de confusión:\n{confusion_mat}\n")
    f.write(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}\n")
    f.write(f"Número total de iteraciones realizadas: {iteration_count}\n")

# También imprimir los resultados en consola
print(f"Resultados para el Árbol de Decisión con Hiperparámetros Ajustados:\n")
print(f"Accuracy general del modelo: {overall_accuracy:.2f}")
print(f"Precision global (ponderada): {overall_precision:.2f}")
print(f"Recall global (ponderado): {overall_recall:.2f}")
print(f"F1-score global (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación general:\n{overall_report}")
print(f"Matriz de confusión:\n{confusion_mat}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Número total de iteraciones realizadas: {iteration_count}")


"""
#Probe bastantes hiper parametros y no aumenta más que esto " model = DecisionTreeClassifier(
    random_state=42,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features='sqrt',
    criterion='entropy',            # Ajusta el criterio de división
    max_leaf_nodes=50,              # Máximo número de nodos hoja
    min_weight_fraction_leaf=0.01,  # Fracción mínima de peso en cada hoja
    class_weight='balanced',        # Ponderación de clases para manejar desbalanceo
    splitter='best',                # Estrategia de selección de características en cada nodo
    ccp_alpha=0.001                 # Poda basada en el parámetro de complejidad
)""
    """
#%%

# XGBoost

# Codificación de las etiquetas (transformación de [-1, 0, 1] a [0, 1, 2])
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Aplicar RFE para seleccionar las mejores características utilizando XGBoost
model_rfe = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
rfe = RFE(model_rfe, n_features_to_select=5)  # Ajusta n_features_to_select según lo desees
X_rfe = rfe.fit_transform(X_scaled, y_encoded)

print("Características seleccionadas por RFE:", np.array(features)[rfe.support_])

results = []

# Acumular todas las predicciones y etiquetas verdaderas
all_y_true = []
all_y_pred = []

# Contador de iteraciones
iteration_count = 0

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
# Guardar resultados en archivo .text
output_path = r'.\..\output\XGBoost_banco_bci.text'
with open(output_path, 'w') as f:
    f.write(f"Resultados para el modelo XGBoost:\n")
    f.write(f"Precisión general del modelo: {overall_accuracy:.2f}\n")
    f.write(f"Precision global (ponderada): {overall_precision:.2f}\n")
    f.write(f"Recall global (ponderado): {overall_recall:.2f}\n")
    f.write(f"F1-score global (ponderado): {overall_f1:.2f}\n")
    f.write(f"Reporte de clasificación general:\n{overall_report}\n")
    f.write(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}\n")
    f.write(f"Número total de iteraciones realizadas: {iteration_count}\n")
    
print(f"Precisión general del modelo: {overall_accuracy:.2f}")
print(f"Precision global (ponderada): {overall_precision:.2f}")
print(f"Recall global (ponderado): {overall_recall:.2f}")
print(f"F1-score global (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación general:\n {overall_report}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Número total de iteraciones realizadas: {iteration_count}")

#%%

#XGBoost segun paper


# Codificación de las etiquetas (transformación de [-1, 0, 1] a [0, 1, 2])
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Aplicar RFE para seleccionar las mejores características utilizando XGBoost
model_rfe = XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    n_estimators=1000,  # Número de árboles (submodelos)
    max_depth=5,        # Profundidad máxima de los árboles
    learning_rate=0.1,  # Tasa de aprendizaje
    random_state=42
)
rfe = RFE(model_rfe, n_features_to_select=7)  # Ajusta n_features_to_select según lo desees
X_rfe = rfe.fit_transform(X_scaled, y_encoded)

print("Características seleccionadas por RFE:", np.array(features)[rfe.support_])

results = []

# Acumular todas las predicciones y etiquetas verdaderas
all_y_true = []
all_y_pred = []

# Contador de iteraciones
iteration_count = 0

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

    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
        n_estimators=1000,  # Número de árboles
        max_depth=5,        # Profundidad máxima
        learning_rate=0.1,  # Tasa de aprendizaje
        random_state=42
    )
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

# Guardar resultados en archivo .text
output_path = r'.\..\output\XGBoost_banco_bci_paper.text'
with open(output_path, 'w') as f:
    f.write(f"Resultados para el modelo XGBoost:\n")
    f.write(f"Precisión general del modelo: {overall_accuracy:.2f}\n")
    f.write(f"Precision global (ponderada): {overall_precision:.2f}\n")
    f.write(f"Recall global (ponderado): {overall_recall:.2f}\n")
    f.write(f"F1-score global (ponderado): {overall_f1:.2f}\n")
    f.write(f"Reporte de clasificación general:\n{overall_report}\n")
    f.write(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}\n")
    f.write(f"Número total de iteraciones realizadas: {iteration_count}\n")

# También imprimir los resultados en consola
print(f"Resultados para el modelo XGBoost:\n")
print(f"Precisión general del modelo: {overall_accuracy:.2f}")
print(f"Precision global (ponderada): {overall_precision:.2f}")
print(f"Recall global (ponderado): {overall_recall:.2f}")
print(f"F1-score global (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación general:\n{overall_report}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Número total de iteraciones realizadas: {iteration_count}")





#%%


# Random Forest

# Codificación de las etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Aplicar RFE para seleccionar las mejores características utilizando Random Forest
model_rfe = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(model_rfe, n_features_to_select=5)  # Ajusta n_features_to_select según lo desees
X_rfe = rfe.fit_transform(X_scaled, y_encoded)

print("Características seleccionadas por RFE:", np.array(features)[rfe.support_])

results = []

# Acumular todas las predicciones y etiquetas verdaderas
all_y_true = []
all_y_pred = []

# Contador de iteraciones
iteration_count = 0

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

# Guardar resultados en archivo .text
output_path = r'.\..\output\RF_banco_bci.text'
with open(output_path, 'w') as f:
    f.write(f"Resultados para el modelo de Random Forest:\n")
    f.write(f"Precisión general del modelo: {overall_accuracy:.2f}\n")
    f.write(f"Precision global (ponderada): {overall_precision:.2f}\n")
    f.write(f"Recall global (ponderado): {overall_recall:.2f}\n")
    f.write(f"F1-score global (ponderado): {overall_f1:.2f}\n")
    f.write(f"Reporte de clasificación general:\n{overall_report}\n")
    f.write(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}\n")
    f.write(f"Número total de iteraciones realizadas: {iteration_count}\n")
    
print(f"Precisión general del modelo random forest: {overall_accuracy:.2f}")
print(f"Precision global (ponderada): {overall_precision:.2f}")
print(f"Recall global (ponderado): {overall_recall:.2f}")
print(f"F1-score global (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación general:\n {overall_report}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Número total de iteraciones realizadas: {iteration_count}")



#%%


#Random forest con hiperparametros del paper

# Codificación de las etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Aplicar RFE para seleccionar las mejores características utilizando Random Forest
# Ajusta los hiperparámetros según el paper
model_rfe = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=2,
                                   min_samples_leaf=1, max_features='sqrt', criterion='gini',
                                   n_jobs=-1, random_state=42)

rfe = RFE(model_rfe, n_features_to_select=5)  # Ajusta n_features_to_select según lo desees
X_rfe = rfe.fit_transform(X_scaled, y_encoded)

print("Características seleccionadas por RFE:", np.array(features)[rfe.support_])

results = []

# Acumular todas las predicciones y etiquetas verdaderas
all_y_true = []
all_y_pred = []

# Contador de iteraciones
iteration_count = 0

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

    model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=2,
                                   min_samples_leaf=1, max_features='sqrt', criterion='gini',
                                   n_jobs=-1, random_state=42)
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

# Guardar resultados en archivo .text
output_path = r'.\..\output\RF_banco_bci_paper.text'
with open(output_path, 'w') as f:
    f.write(f"Resultados para el modelo de Random Forest:\n")
    f.write(f"Precisión general del modelo: {overall_accuracy:.2f}\n")
    f.write(f"Precision global (ponderada): {overall_precision:.2f}\n")
    f.write(f"Recall global (ponderado): {overall_recall:.2f}\n")
    f.write(f"F1-score global (ponderado): {overall_f1:.2f}\n")
    f.write(f"Reporte de clasificación general:\n{overall_report}\n")
    f.write(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}\n")
    f.write(f"Número total de iteraciones realizadas: {iteration_count}\n")

# También imprimir los resultados en consola
print(f"Resultados para el modelo de Random Forest:\n")
print(f"Precisión general del modelo: {overall_accuracy:.2f}")
print(f"Precision global (ponderada): {overall_precision:.2f}")
print(f"Recall global (ponderado): {overall_recall:.2f}")
print(f"F1-score global (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación general:\n {overall_report}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Número total de iteraciones realizadas: {iteration_count}")


#%%

# Naive Bayes

# Codificación de las etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X_scaled, y_encoded)
selected_features = np.array(features)[selector.get_support()]
print("Características seleccionadas:", selected_features)


results = []

# Acumular todas las predicciones y etiquetas verdaderas
all_y_true = []
all_y_pred = []

# Contador de iteraciones
iteration_count = 0

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

# Guardar resultados en archivo .text
output_path = r'.\..\output\NB_SelectKBest_banco_bci.text'
with open(output_path, 'w') as f:
    f.write(f"Resultados para el Naive Bayes:\n")
    f.write(f"Precisión general del modelo: {overall_accuracy:.2f}\n")
    f.write(f"Precision global (ponderada): {overall_precision:.2f}\n")
    f.write(f"Recall global (ponderado): {overall_recall:.2f}\n")
    f.write(f"F1-score global (ponderado): {overall_f1:.2f}\n")
    f.write(f"Reporte de clasificación general:\n{overall_report}\n")
    f.write(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}\n")
    f.write(f"Número total de iteraciones realizadas: {iteration_count}\n")


print(f"Precisión general del modelo naive bayes: {overall_accuracy:.2f}")
print(f"Precision global (ponderada): {overall_precision:.2f}")
print(f"Recall global (ponderado): {overall_recall:.2f}")
print(f"F1-score global (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación general:\n {overall_report}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Número total de iteraciones realizadas: {iteration_count}")


#%%

from sklearn.model_selection import GridSearchCV
# Codificación de las etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Selección de características
selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X_scaled, y_encoded)
selected_features = np.array(features)[selector.get_support()]
print("Características seleccionadas:", selected_features)

# Definir los posibles valores para el hiperparámetro var_smoothing
param_grid = {'var_smoothing': np.logspace(-12, -6, num=7)}

# Configurar GridSearchCV para Naive Bayes
grid_search = GridSearchCV(GaussianNB(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_selected, y_encoded)

# Obtener el mejor modelo con el valor óptimo de var_smoothing
best_model = grid_search.best_estimator_
print(f"Mejor valor de var_smoothing: {grid_search.best_params_['var_smoothing']}")

# Ahora podemos usar el mejor modelo para predicciones en la ventana rodante
results = []
all_y_true = []
all_y_pred = []
iteration_count = 0
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

    # Entrenar y predecir con el mejor modelo
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    # Acumular predicciones y etiquetas verdaderas
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)
    iteration_count += 1

hora_de_fin = datetime.now()

# Métricas finales
overall_accuracy = accuracy_score(all_y_true, all_y_pred)
overall_precision = precision_score(all_y_true, all_y_pred, average='weighted')
overall_recall = recall_score(all_y_true, all_y_pred, average='weighted')
overall_f1 = f1_score(all_y_true, all_y_pred, average='weighted')
overall_report = classification_report(all_y_true, all_y_pred, target_names=[str(cls) for cls in label_encoder.classes_])

# Guardar y mostrar resultados
output_path = r'.\..\output\NB_SelectKBest_banco_bci_opt_smoothing.text'
with open(output_path, 'w') as f:
    f.write(f"Resultados para Naive Bayes con ajuste de var_smoothing:\n")
    f.write(f"Precisión general: {overall_accuracy:.2f}\n")
    f.write(f"Precision (ponderada): {overall_precision:.2f}\n")
    f.write(f"Recall (ponderado): {overall_recall:.2f}\n")
    f.write(f"F1-score (ponderado): {overall_f1:.2f}\n")
    f.write(f"Reporte de clasificación:\n{overall_report}\n")
    f.write(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}\n")
    f.write(f"Iteraciones realizadas: {iteration_count}\n")

print(f"Precisión general del modelo Naive Bayes: {overall_accuracy:.2f}")
print(f"Precision (ponderada): {overall_precision:.2f}")
print(f"Recall (ponderado): {overall_recall:.2f}")
print(f"F1-score (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación:\n {overall_report}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Iteraciones realizadas: {iteration_count}")

#%%
# Naive Bayes con random forest

# Codificación de las etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Aplicar RFE utilizando Random Forest para seleccionar las mejores características
model_rfe = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(model_rfe, n_features_to_select=5)  # Ajusta n_features_to_select según lo desees
X_rfe = rfe.fit_transform(X_scaled, y_encoded)

print("Características seleccionadas por RFE:", np.array(features)[rfe.support_])

results = []

# Acumular todas las predicciones y etiquetas verdaderas
all_y_true = []
all_y_pred = []

# Contador de iteraciones
iteration_count = 0
    
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

    # Crear y ajustar el modelo Naive Bayes
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

# Guardar resultados en archivo .text
output_path = r'.\..\output\NB_randomforest_banco_bci.text'
with open(output_path, 'w') as f:
    f.write(f"Resultados para el Naive Bayes:\n")
    f.write(f"Precisión general del modelo: {overall_accuracy:.2f}\n")
    f.write(f"Precision global (ponderada): {overall_precision:.2f}\n")
    f.write(f"Recall global (ponderado): {overall_recall:.2f}\n")
    f.write(f"F1-score global (ponderado): {overall_f1:.2f}\n")
    f.write(f"Reporte de clasificación general:\n{overall_report}\n")
    f.write(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}\n")
    f.write(f"Número total de iteraciones realizadas: {iteration_count}\n")

# También imprimir los resultados en consola
print(f"Resultados para el Naive Bayes:\n")
print(f"Precisión general del modelo: {overall_accuracy:.2f}")
print(f"Precision global (ponderada): {overall_precision:.2f}")
print(f"Recall global (ponderado): {overall_recall:.2f}")
print(f"F1-score global (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación general:\n{overall_report}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Número total de iteraciones realizadas: {iteration_count}")








#%%

# MLP (Multi-Layer Perceptron)

# Codificación de las etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


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


# MLP (Multi-Layer Perceptron) con random forest sin hiperparametros del paper

# Codificación de las etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Aplicar RFE para seleccionar las mejores características utilizando Random Forest
model_rfe = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(model_rfe, n_features_to_select=7)  # Ajusta n_features_to_select según lo desees
X_rfe = rfe.fit_transform(X_scaled, y_encoded)
selected_features = np.array(features)[rfe.support_]
print("Características seleccionadas por RFE:", selected_features)

results = []

# Acumular todas las predicciones y etiquetas verdaderas
all_y_true = []
all_y_pred = []

# Contador de iteraciones
iteration_count = 0

# Calcular el índice para el 80% de los datos
train_size = int(len(dates) * 0.8)

hora_de_inicio = datetime.now()


#ojo con esto, esta bien esto que hacemos?, esta iterando para obtener las predicciones?, oh esta aplicando ventana
#sobre el 20% y no el 80 %? 
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

# Guardar resultados en archivo .text
output_path = r'.\..\output\MLP_results_no_paper.text'
with open(output_path, 'w') as f:
    f.write(f"Resultados para el MLP con random forest:\n")
    f.write(f"Precisión general del modelo: {overall_accuracy:.2f}\n")
    f.write(f"Precision global (ponderada): {overall_precision:.2f}\n")
    f.write(f"Recall global (ponderado): {overall_recall:.2f}\n")
    f.write(f"F1-score global (ponderado): {overall_f1:.2f}\n")
    f.write(f"Reporte de clasificación general:\n{overall_report}\n")
    f.write(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}\n")
    f.write(f"Número total de iteraciones realizadas: {iteration_count}\n")


print(f"Precisión general del modelo: {overall_accuracy:.2f}")
print(f"Precision global (ponderada): {overall_precision:.2f}")
print(f"Recall global (ponderado): {overall_recall:.2f}")
print(f"F1-score global (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación general:\n {overall_report}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Número total de iteraciones realizadas: {iteration_count}")

#%%



#MLP con random Forest y hiperaparametros del paper

# Codificación de las etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Aplicar RFE para seleccionar las mejores características utilizando Random Forest
model_rfe = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(model_rfe, n_features_to_select=7)  # Ajusta n_features_to_select según lo desees
X_rfe = rfe.fit_transform(X_scaled, y_encoded)
selected_features = np.array(features)[rfe.support_]
print("Características seleccionadas por RFE:", selected_features)

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

    # Ahora usamos MLP para el entrenamiento y predicción
    model = MLPClassifier(hidden_layer_sizes=(30,), max_iter=300, random_state=42)  # Tamaño de la capa oculta de 30
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

# Guardar resultados en archivo .text
output_path = r'.\..\output\MLP_banco_bci_paper.text'
with open(output_path, 'w') as f:
    f.write(f"Resultados para el MLP con random forest:\n")
    f.write(f"Precisión general del modelo: {overall_accuracy:.2f}\n")
    f.write(f"Precision global (ponderada): {overall_precision:.2f}\n")
    f.write(f"Recall global (ponderado): {overall_recall:.2f}\n")
    f.write(f"F1-score global (ponderado): {overall_f1:.2f}\n")
    f.write(f"Reporte de clasificación general:\n{overall_report}\n")
    f.write(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}\n")
    f.write(f"Número total de iteraciones realizadas: {iteration_count}\n")

# También imprimir los resultados en consola
print(f"Resultados para el MLP:\n")
print(f"Precisión general del modelo: {overall_accuracy:.2f}")
print(f"Precision global (ponderada): {overall_precision:.2f}")
print(f"Recall global (ponderado): {overall_recall:.2f}")
print(f"F1-score global (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación general:\n{overall_report}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Número total de iteraciones realizadas: {iteration_count}")
#%%

# LSTM (Long Short-Term Memory)

# Codificación de las etiquetas para un problema multicategoría
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_encoded = np.eye(len(label_encoder.classes_))[y_encoded]  # One-hot encoding

print("Clases originales:", label_encoder.classes_)
print("Valores codificados:", list(label_encoder.transform(label_encoder.classes_)))

print("Distribución de clases en el conjunto de entrenamiento:", Counter(np.argmax(y_encoded, axis=1)))

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





