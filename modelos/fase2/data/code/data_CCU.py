#%%
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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
ruta_archivo = r'.\..\Input\CCU.SN.csv'
df = pd.read_csv(ruta_archivo)
data_CCU_SNN = df.copy()
if 'Unnamed: 0' in df.columns:
    data_CCU_SNN.drop(columns=['Unnamed: 0'], inplace=True)
print("columnas totales:", len(data_CCU_SNN.columns))

#%%

# Selección de todas las características excepto 'Date' y 'Tendencia'
features = [col for col in data_CCU_SNN.columns if col not in ['Date', 'Tendencia', 'Sentimiento Máximo', 'Sentimiento Mínimo',"Tendencia Sentimiento"]]
X = data_CCU_SNN[features]
y = data_CCU_SNN['Tendencia']

# Normalización de los datos
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

window_size = 3
dates = pd.to_datetime(data_CCU_SNN['Date'])
train_size = int(len(dates) * 0.8)

#%%

#Regresión logistica


results = []

# Acumular todas las predicciones y etiquetas verdaderas
all_y_true = []
all_y_pred = []

hora_de_inicio = datetime.now()

# Usar el 80% inicial como conjunto de entrenamiento
X_train_initial = X[:train_size]
y_train_initial = y[:train_size]

# Contador de iteraciones
iteration_count = 0

# Iterar sobre el 20% restante usando la ventana rodante
for start in range(train_size, len(dates) - window_size):
    test_indices = (dates >= dates.iloc[start]) & (dates < dates.iloc[start + window_size])
    train_indices = dates < dates.iloc[start]

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    # Verificar que haya suficientes clases en el conjunto de entrenamiento
    if len(np.unique(y_train)) < 2:
        continue  # Si solo hay una clase, pasa a la siguiente iteración

    if len(X_train) == 0 or len(X_test) == 0:
        continue

    model = LogisticRegression(multi_class='ovr', max_iter=1000, penalty='l2')
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

print(f"Accuracy general del modelo Regresion logistica: {overall_accuracy:.2f}")
print(f"Precision global (ponderada): {overall_precision:.2f}")
print(f"Recall global (ponderado): {overall_recall:.2f}")
print(f"F1-score global (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación general:\n {overall_report}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Número total de iteraciones realizadas: {iteration_count}")


#%%

# Guardar los resultados en un archivo de texto
output_file_path = r'.\..\output\metricas_CCU\LR_segun_paper.txt'
with open(output_file_path, 'w') as f:
    f.write(f"Resultados para LR con Hiperparametros del paper:\n")
    f.write(f"Accuracy general del modelo: {overall_accuracy:.2f}\n")
    f.write(f"Precision global (ponderada): {overall_precision:.2f}\n")
    f.write(f"Recall global (ponderado): {overall_recall:.2f}\n")
    f.write(f"F1-score global (ponderado): {overall_f1:.2f}\n")
    f.write(f"Reporte de clasificación general:\n{overall_report}\n")
    f.write(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}\n")
    f.write(f"Número total de iteraciones realizadas: {iteration_count}\n")



#%%

#Regresión logistica


results = []

# Acumular todas las predicciones y etiquetas verdaderas
all_y_true = []
all_y_pred = []

hora_de_inicio = datetime.now()

# Usar el 80% inicial como conjunto de entrenamiento
X_train_initial = X[:train_size]
y_train_initial = y[:train_size]

# Contador de iteraciones
iteration_count = 0

# Iterar sobre el 20% restante usando la ventana rodante
for start in range(train_size, len(dates) - window_size):
    test_indices = (dates >= dates.iloc[start]) & (dates < dates.iloc[start + window_size])
    train_indices = dates < dates.iloc[start]

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
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

print(f"Accuracy general del modelo Regresion logistica: {overall_accuracy:.2f}")
print(f"Precision global (ponderada): {overall_precision:.2f}")
print(f"Recall global (ponderado): {overall_recall:.2f}")
print(f"F1-score global (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación general:\n {overall_report}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Número total de iteraciones realizadas: {iteration_count}")



#%%


# Guardar los resultados en un archivo de texto
output_file_path = r'.\..\output\metricas_CCU\LR_ajustado.txt'
with open(output_file_path, 'w') as f:
    f.write(f"Resultados para LR con Hiperparametros del paper:\n")
    f.write(f"Accuracy general del modelo: {overall_accuracy:.2f}\n")
    f.write(f"Precision global (ponderada): {overall_precision:.2f}\n")
    f.write(f"Recall global (ponderado): {overall_recall:.2f}\n")
    f.write(f"F1-score global (ponderado): {overall_f1:.2f}\n")
    f.write(f"Reporte de clasificación general:\n{overall_report}\n")
    f.write(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}\n")
    f.write(f"Número total de iteraciones realizadas: {iteration_count}\n")






#%%

# Árbol de decisión

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
    X_train, X_test = X_scaled[train_indices], X_scaled[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    # Verificar que haya suficientes clases en el conjunto de entrenamiento
    if len(np.unique(y_train)) < 2:
        continue  # Si solo hay una clase, pasa a la siguiente iteración

    if len(X_train) == 0 or len(X_test) == 0:
        continue

    # Crear y ajustar el modelo con limitación de profundidad para evitar sobreajuste
    model = DecisionTreeClassifier(random_state=42, max_depth=15) #Cambiando a 15 mejora las metricas #aplicar entropia
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

print(f"Accuracy general del modelo Arbol de decisión: {overall_accuracy:.2f}")
print(f"Precision global (ponderada): {overall_precision:.2f}")
print(f"Recall global (ponderado): {overall_recall:.2f}")
print(f"F1-score global (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación general:\n {overall_report}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Número total de iteraciones realizadas: {iteration_count}")

#%%



# Guardar los resultados en un archivo de texto
output_file_path = r'.\..\output\metricas_CCU\DT_paper.txt'
with open(output_file_path, 'w') as f:
    f.write(f"Resultados para el Árbol de Decisión segun el paper algoritmo Cart profundida 10, mejor que 5:\n")
    f.write(f"Accuracy general del modelo: {overall_accuracy:.2f}\n")
    f.write(f"Precision global (ponderada): {overall_precision:.2f}\n")
    f.write(f"Recall global (ponderado): {overall_recall:.2f}\n")
    f.write(f"F1-score global (ponderado): {overall_f1:.2f}\n")
    f.write(f"Reporte de clasificación general:\n{overall_report}\n")
    f.write(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}\n")
    f.write(f"Número total de iteraciones realizadas: {iteration_count}\n")


#%%
#Arbol ajustado



# Lista para almacenar resultados
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
    X_train, X_test = X_scaled[train_indices], X_scaled[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    # Verificar que haya suficientes clases en el conjunto de entrenamiento
    if len(np.unique(y_train)) < 2:
        continue  # Si solo hay una clase, pasa a la siguiente iteración

    if len(X_train) == 0 or len(X_test) == 0:
        continue

    # Crear y ajustar el modelo con diferentes hiperparámetros
    model = DecisionTreeClassifier(
        random_state=42,
        max_depth=20,  # Cambiar la profundidad máxima
        min_samples_split=20,  # Número mínimo de muestras para dividir un nodo
        min_samples_leaf=10,  # Número mínimo de muestras por hoja
        criterion='entropy',  # Usar entropía
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
overall_report = classification_report(all_y_true, all_y_pred)

# Mostrar las métricas
print(f"Accuracy general del modelo Árbol de Decisión: {overall_accuracy:.2f}")
print(f"Precision global (ponderada): {overall_precision:.2f}")
print(f"Recall global (ponderado): {overall_recall:.2f}")
print(f"F1-score global (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación general:\n{overall_report}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Número total de iteraciones realizadas: {iteration_count}")

#%%

# Guardar los resultados en un archivo de texto
output_file_path = r'.\..\output\metricas_CCU\DT_mas_ajustado.txt'
with open(output_file_path, 'w') as f:
    f.write(f"Resultados para el Árbol de Decisión ajustado:\n")
    f.write(f"Accuracy general del modelo: {overall_accuracy:.2f}\n")
    f.write(f"Precision global (ponderada): {overall_precision:.2f}\n")
    f.write(f"Recall global (ponderado): {overall_recall:.2f}\n")
    f.write(f"F1-score global (ponderado): {overall_f1:.2f}\n")
    f.write(f"Reporte de clasificación general:\n{overall_report}\n")
    f.write(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}\n")
    f.write(f"Número total de iteraciones realizadas: {iteration_count}\n")




#%%


# XGBoost Ajustado


# Codificación de las etiquetas (transformación de [-1, 0, 1] a [0, 1, 2])
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

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

    X_train, X_test = X_scaled[train_indices], X_scaled[test_indices]
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

print(f"Precisión general del modelo Xgboost ajustado: {overall_accuracy:.2f}")
print(f"Precision global (ponderada): {overall_precision:.2f}")
print(f"Recall global (ponderado): {overall_recall:.2f}")
print(f"F1-score global (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación general:\n {overall_report}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Número total de iteraciones realizadas: {iteration_count}")


#%%
# Guardar resultados en archivo .text
output_path = r'.\..\output\metricas_CCU\XGboost_con_100_submodels.txt'
with open(output_path, 'w') as f:
    f.write(f"Resultados para el modelo XGBoost con 100 submodelos en ves de 1000 como lo hace el paper:\n")
    f.write(f"Precisión general del modelo: {overall_accuracy:.2f}\n")
    f.write(f"Precision global (ponderada): {overall_precision:.2f}\n")
    f.write(f"Recall global (ponderado): {overall_recall:.2f}\n")
    f.write(f"F1-score global (ponderado): {overall_f1:.2f}\n")
    f.write(f"Reporte de clasificación general:\n{overall_report}\n")
    f.write(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}\n")
    f.write(f"Número total de iteraciones realizadas: {iteration_count}\n")

#%%

#Xgboost segun paper con 1000 sub modelos
# Codificación de las etiquetas (transformación de [-1, 0, 1] a [0, 1, 2])


#18 min aprox

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

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

    X_train, X_test = X_scaled[train_indices], X_scaled[test_indices]
    y_train, y_test = y_encoded[train_indices], y_encoded[test_indices]

    # Verificar que haya suficientes clases en el conjunto de entrenamiento
    if len(np.unique(y_train)) < 2:
        continue  # Si solo hay una clase, pasa a la siguiente iteración

    if len(X_train) == 0 or len(X_test) == 0:
        continue

    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_estimators=1000)
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

print(f"Precisión general del modelo Xgboost segun paper: {overall_accuracy:.2f}")
print(f"Precision global (ponderada): {overall_precision:.2f}")
print(f"Recall global (ponderado): {overall_recall:.2f}")
print(f"F1-score global (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación general:\n {overall_report}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Número total de iteraciones realizadas: {iteration_count}")

#%%

# Guardar resultados en archivo .text
output_path = r'.\..\output\metricas_CCU\XGboost_segun_paper.txt'
with open(output_path, 'w') as f:
    f.write(f"Resultados para el modelo XGBoost con 1000 submodelos:\n")
    f.write(f"Precisión general del modelo: {overall_accuracy:.2f}\n")
    f.write(f"Precision global (ponderada): {overall_precision:.2f}\n")
    f.write(f"Recall global (ponderado): {overall_recall:.2f}\n")
    f.write(f"F1-score global (ponderado): {overall_f1:.2f}\n")
    f.write(f"Reporte de clasificación general:\n{overall_report}\n")
    f.write(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}\n")
    f.write(f"Número total de iteraciones realizadas: {iteration_count}\n")




#%%

# Random Forest segun el paper

# 4 min aprox
# Codificación de las etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

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

    X_train, X_test = X_scaled[train_indices], X_scaled[test_indices]
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

print(f"Precisión general del modelo Random forest paper: {overall_accuracy:.2f}")
print(f"Precision global (ponderada): {overall_precision:.2f}")
print(f"Recall global (ponderado): {overall_recall:.2f}")
print(f"F1-score global (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación general:\n {overall_report}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Número total de iteraciones realizadas: {iteration_count}")


#%%

# Guardar resultados en archivo .text
output_path = r'.\..\output\metricas_CCU\RF_segun_paper.txt'
with open(output_path, 'w') as f:
    f.write(f"Resultados para el modelo RF con 100 submodelos segun paper:\n")
    f.write(f"Precisión general del modelo: {overall_accuracy:.2f}\n")
    f.write(f"Precision global (ponderada): {overall_precision:.2f}\n")
    f.write(f"Recall global (ponderado): {overall_recall:.2f}\n")
    f.write(f"F1-score global (ponderado): {overall_f1:.2f}\n")
    f.write(f"Reporte de clasificación general:\n{overall_report}\n")
    f.write(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}\n")
    f.write(f"Número total de iteraciones realizadas: {iteration_count}\n")

#%%

#Random forest con más hiperparametros para ver su rendimiento, se aumenta de 100 a 1000 submodelos y la profundidad
# de la hoja aumenta de 5 a 10, dado que anteriormente en arbol de decisión obtuvimos mejor rendimiento
# aumentando este parametro

# 16 minutos aprox

# Codificación de las etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

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

    X_train, X_test = X_scaled[train_indices], X_scaled[test_indices]
    y_train, y_test = y_encoded[train_indices], y_encoded[test_indices]

    # Verificar que haya suficientes clases en el conjunto de entrenamiento
    if len(np.unique(y_train)) < 2:
        continue  # Si solo hay una clase, pasa a la siguiente iteración

    if len(X_train) == 0 or len(X_test) == 0:
        continue

    model = RandomForestClassifier(n_estimators=1000, max_depth=10, min_samples_split=2,
                                   min_samples_leaf=1, max_features='sqrt', criterion='entropy', # ver aplicando entropy
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

print(f"Precisión general del modelo Random forest ajustado: {overall_accuracy:.2f}")
print(f"Precision global (ponderada): {overall_precision:.2f}")
print(f"Recall global (ponderado): {overall_recall:.2f}")
print(f"F1-score global (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación general:\n {overall_report}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Número total de iteraciones realizadas: {iteration_count}")


#%%

# Guardar resultados en archivo .text
output_path = r'.\..\output\metricas_CCU\RF_ajustado.txt'
with open(output_path, 'w') as f:
    f.write(f"Resultados para el modelo RF ajustado con más hiperparametros:\n")
    f.write(f"Precisión general del modelo: {overall_accuracy:.2f}\n")
    f.write(f"Precision global (ponderada): {overall_precision:.2f}\n")
    f.write(f"Recall global (ponderado): {overall_recall:.2f}\n")
    f.write(f"F1-score global (ponderado): {overall_f1:.2f}\n")
    f.write(f"Reporte de clasificación general:\n{overall_report}\n")
    f.write(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}\n")
    f.write(f"Número total de iteraciones realizadas: {iteration_count}\n")





#%%

# Naive Bayes

# nuevamente la clase 0 se va a abajo

# Codificación de las etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

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

    X_train, X_test = X_scaled[train_indices], X_scaled[test_indices]
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

print(f"Precisión general del modelo Naive bayes: {overall_accuracy:.2f}")
print(f"Precision global (ponderada): {overall_precision:.2f}")
print(f"Recall global (ponderado): {overall_recall:.2f}")
print(f"F1-score global (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación general:\n {overall_report}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Número total de iteraciones realizadas: {iteration_count}")


#%%

# Guardar resultados en archivo .text
output_path = r'.\..\output\metricas_CCU\NB_basico.txt'
with open(output_path, 'w') as f:
    f.write(f"Resultados para el modelo NB :\n")
    f.write(f"Precisión general del modelo: {overall_accuracy:.2f}\n")
    f.write(f"Precision global (ponderada): {overall_precision:.2f}\n")
    f.write(f"Recall global (ponderado): {overall_recall:.2f}\n")
    f.write(f"F1-score global (ponderado): {overall_f1:.2f}\n")
    f.write(f"Reporte de clasificación general:\n{overall_report}\n")
    f.write(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}\n")
    f.write(f"Número total de iteraciones realizadas: {iteration_count}\n")





#%%

# MLP (Multi-Layer Perceptron)


# Codificación de las etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

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

    X_train, X_test = X_scaled[train_indices], X_scaled[test_indices]
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

# Guardar resultados en archivo .text
output_path = r'.\..\output\metricas_CCU\MLP_basico.txt'
with open(output_path, 'w') as f:
    f.write(f"Resultados para el modelo MLP ajustado :\n")
    f.write(f"Precisión general del modelo: {overall_accuracy:.2f}\n")
    f.write(f"Precision global (ponderada): {overall_precision:.2f}\n")
    f.write(f"Recall global (ponderado): {overall_recall:.2f}\n")
    f.write(f"F1-score global (ponderado): {overall_f1:.2f}\n")
    f.write(f"Reporte de clasificación general:\n{overall_report}\n")
    f.write(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}\n")
    f.write(f"Número total de iteraciones realizadas: {iteration_count}\n")

#%%


# MLP (Multi-Layer Perceptron) segun paper


# Codificación de las etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

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

    X_train, X_test = X_scaled[train_indices], X_scaled[test_indices]
    y_train, y_test = y_encoded[train_indices], y_encoded[test_indices]

    # Verificar que haya suficientes clases en el conjunto de entrenamiento
    if len(np.unique(y_train)) < 2:
        continue  # Si solo hay una clase, pasa a la siguiente iteración

    if len(X_train) == 0 or len(X_test) == 0:
        continue

    # Ahora usamos MLP para el entrenamiento y predicción
    model = MLPClassifier(hidden_layer_sizes=(30,), max_iter=300, random_state=42)  # Tamaño de la capa oculta de 30 segun paper
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

# Guardar resultados en archivo .text
output_path = r'.\..\output\metricas_CCU\MLP_segun_paper.txt'
with open(output_path, 'w') as f:
    f.write(f"Resultados para el modelo MLP segun paper con capa oculta 30 :\n")
    f.write(f"Precisión general del modelo: {overall_accuracy:.2f}\n")
    f.write(f"Precision global (ponderada): {overall_precision:.2f}\n")
    f.write(f"Recall global (ponderado): {overall_recall:.2f}\n")
    f.write(f"F1-score global (ponderado): {overall_f1:.2f}\n")
    f.write(f"Reporte de clasificación general:\n{overall_report}\n")
    f.write(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}\n")
    f.write(f"Número total de iteraciones realizadas: {iteration_count}\n")






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

# Definir la ruta de salida para el archivo .txt
output_path_lstm = r'.\..\output\metricas_CCU\LSTM__resultados.txt'

# Crear los directorios si no existen

# Abrir el archivo y guardar los resultados
with open(output_path_lstm, 'w') as f:
    f.write(f"Resultados para el modelo LSTM:\n")
    f.write(f"Precisión del modelo: {accuracy:.2f}\n")
    f.write(f"Reporte de clasificación:\n{report}\n")
    f.write(f"Matriz de confusión:\n{conf_matrix}\n")

