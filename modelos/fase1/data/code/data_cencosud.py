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
ruta_archivo = r".\..\Input\CENCOSUD.SN.csv"
df = pd.read_csv(ruta_archivo)
data_CENCOSUD_SN = df.copy()
if 'Unnamed: 0' in df.columns:
    data_CENCOSUD_SN.drop(columns=['Unnamed: 0'], inplace=True)
print("columnas totales:", len(data_CENCOSUD_SN.columns))

#%%
# Selección de carácterísticas
static_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
ta_cols = [col for col in data_CENCOSUD_SN.columns if col not in static_cols + ['Date', 'Tendencia']]
features = static_cols + ta_cols

X = data_CENCOSUD_SN[features]
y = data_CENCOSUD_SN['Tendencia']

# Normalización de los datos
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_ta_scaled = X_scaled[:, [X.columns.get_loc(col) for col in ta_cols]]
X_static_scaled = X_scaled[:, [X.columns.get_loc(col) for col in static_cols]]
X_static_df = pd.DataFrame(X_static_scaled, columns=static_cols)

window_size = 3
dates = pd.to_datetime(data_CENCOSUD_SN['Date'])
train_size = int(len(dates) * 0.8)


#%%

# Regresión logística

# Aplicar RFE para seleccionar las mejores características
model_rfe = LogisticRegression(multi_class='ovr', max_iter=1000, penalty='l2')  # Se agrega penalty='l2'
rfe = RFE(model_rfe, n_features_to_select=7)

X_rfe = rfe.fit_transform(X_ta_scaled, y)
selected_technical_features = np.array(ta_cols)[rfe.support_]

print("Indicadores técnicos seleccionados por RFE:", selected_technical_features)

# Crear DataFrame con características seleccionadas y concatenar con columnas estáticas normalizadas
X_rfe_df = pd.DataFrame(X_rfe, columns=selected_technical_features)

X_final = pd.concat([X_rfe_df, X_static_df.reset_index(drop=True)], axis=1)

print("Columnas finales para el modelo:", X_final.columns)

results = []

# Acumular todas las predicciones y etiquetas verdaderas
all_y_true = []
all_y_pred = []

hora_de_inicio = datetime.now()

# Contador de iteraciones
iteration_count = 0

# Iterar sobre el 20% restante usando la ventana rodante
for start in range(train_size, len(dates) - window_size):
    test_indices = (dates >= dates.iloc[start]) & (dates < dates.iloc[start + window_size])
    train_indices = dates < dates.iloc[start]

    X_train = X_final[train_indices]
    y_train = y[train_indices]
    X_test = X_final[test_indices]
    y_test = y[test_indices]

    # Verificar clases en el conjunto de entrenamiento
    if len(np.unique(y_train)) < 2 or len(X_train) == 0 or len(X_test) == 0:
        continue

    model = LogisticRegression(multi_class='ovr', max_iter=1000, penalty='l2')  # Se agrega penalty='l2'
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

print(f"Accuracy general del modelo LR: {overall_accuracy:.2f}")
print(f"Precision global (ponderada): {overall_precision:.2f}")
print(f"Recall global (ponderado): {overall_recall:.2f}")
print(f"F1-score global (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación general:\n {overall_report}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Número total de iteraciones realizadas: {iteration_count}")


#%%

# Guardar los resultados en un archivo de texto
output_file_path = r'.\..\output\LR_CENCOSUD_Hiperparametro_paper.txt'
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

# Regresión logística con C=100, se probo con varios valores de C, siento este el mejor y solver=lbfgs

#Aqui igual pasa lo de la falta del RFE

results = []

# Acumular todas las predicciones y etiquetas verdaderas
all_y_true = []
all_y_pred = []

hora_de_inicio = datetime.now()

# Contador de iteraciones
iteration_count = 0

# Iterar sobre el 20% restante usando la ventana rodante
for start in range(train_size, len(dates) - window_size):
    test_indices = (dates >= dates.iloc[start]) & (dates < dates.iloc[start + window_size])
    train_indices = dates < dates.iloc[start]

    X_train = X_final[train_indices]
    y_train = y[train_indices]
    X_test = X_final[test_indices]
    y_test = y[test_indices]

    # Verificar clases en el conjunto de entrenamiento
    if len(np.unique(y_train)) < 2 or len(X_train) == 0 or len(X_test) == 0:
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

# Imprimir resultados en consola (opcional)
print(f"Accuracy general del modelo: {overall_accuracy:.2f}")
print(f"Precision global (ponderada): {overall_precision:.2f}")
print(f"Recall global (ponderado): {overall_recall:.2f}")
print(f"F1-score global (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación general:\n {overall_report}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Número total de iteraciones realizadas: {iteration_count}")

#%%
# Guardar los resultados en un archivo de texto
output_file_path = r'.\..\output\LR_CENCOSUD_Ajuste.txt'
with open(output_file_path, 'w') as f:
    f.write(f"Resultados para LR con C = 100 y ajuster de solver y multiclase:\n")
    f.write(f"Accuracy general del modelo: {overall_accuracy:.2f}\n")
    f.write(f"Precision global (ponderada): {overall_precision:.2f}\n")
    f.write(f"Recall global (ponderado): {overall_recall:.2f}\n")
    f.write(f"F1-score global (ponderado): {overall_f1:.2f}\n")
    f.write(f"Reporte de clasificación general:\n{overall_report}\n")
    f.write(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}\n")
    f.write(f"Número total de iteraciones realizadas: {iteration_count}\n")


#%%


# Árbol de decisión segun paper con algoritmo cart. cabe destacar que el algoritmo cart es bueno para predecir
# Problemas multiclase

# Aplicar RFE para seleccionar las mejores características utilizando un Árbol de Decisión
model_rfe = DecisionTreeClassifier(random_state=42)
rfe = RFE(model_rfe, n_features_to_select=7)  # Ajusta n_features_to_select según lo desees
X_rfe = rfe.fit_transform(X_ta_scaled, y)
selected_technical_features = np.array(ta_cols)[rfe.support_]

print("Indicadores técnicos seleccionados por RFE:", selected_technical_features)

# Crear DataFrame con características seleccionadas y concatenar con columnas estáticas normalizadas
X_rfe_df = pd.DataFrame(X_rfe, columns=selected_technical_features)

X_final = pd.concat([X_rfe_df, X_static_df.reset_index(drop=True)], axis=1)

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
    X_train, X_test = X_final[train_indices], X_final[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    # Verificar clases en el conjunto de entrenamiento
    if len(np.unique(y_train)) < 2 or len(X_train) == 0 or len(X_test) == 0:
        continue

    # Crear y ajustar el modelo con limitación de profundidad para evitar sobreajuste
    model = DecisionTreeClassifier(random_state=42, max_depth=10)
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

print(f"Accuracy general del modelo DT: {overall_accuracy:.2f}")
print(f"Precision global (ponderada): {overall_precision:.2f}")
print(f"Recall global (ponderado): {overall_recall:.2f}")
print(f"F1-score global (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación general:\n {overall_report}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Número total de iteraciones realizadas: {iteration_count}")

#%%

# Guardar resultados en archivo .text
output_path = r'.\..\output\DT_CENCOSUD_segun_paper.txt'
with open(output_path, 'w') as f:
    f.write(f"Resultados para el Árbol de Decisión segun el paper algoritmo Cart profundida 10, mejor que 5:\n")
    f.write(f"Accuracy general del modelo: {overall_accuracy:.2f}\n")
    f.write(f"Precision global (ponderada): {overall_precision:.2f}\n")
    f.write(f"Recall global (ponderado): {overall_recall:.2f}\n")
    f.write(f"F1-score global (ponderado): {overall_f1:.2f}\n")
    f.write(f"Reporte de clasificación general:\n{overall_report}\n")
    f.write(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}\n")
    f.write(f"Número total de iteraciones realizadas: {iteration_count}\n")



#%%


#Arbol, dejamos como conclusión que se han estando ocupando más hiperparametros, siendo este el que mejora un pcoo mejor
#logrando subajustar de menor manera la clase 0

# Aplicar RFE para seleccionar las mejores características utilizando un Árbol de Decisión
model_rfe = DecisionTreeClassifier(random_state=42)
rfe = RFE(model_rfe, n_features_to_select=7)  # Ajusta n_features_to_select según lo desees
X_rfe = rfe.fit_transform(X_ta_scaled, y)
selected_technical_features = np.array(ta_cols)[rfe.support_]

print("Indicadores técnicos seleccionados por RFE:", selected_technical_features)

# Crear DataFrame con características seleccionadas y concatenar con columnas estáticas normalizadas
X_rfe_df = pd.DataFrame(X_rfe, columns=selected_technical_features)

X_final = pd.concat([X_rfe_df, X_static_df.reset_index(drop=True)], axis=1)

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
    X_train, X_test = X_final[train_indices], X_final[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    # Verificar clases en el conjunto de entrenamiento
    if len(np.unique(y_train)) < 2 or len(X_train) == 0 or len(X_test) == 0:
        continue

    # Crear y ajustar el modelo de Árbol de Decisión con nuevos hiperparámetros
    model = DecisionTreeClassifier(
        random_state=42,
        max_depth=15,                  # Ajuste de profundidad
        min_samples_split=20,          # Mínimo de muestras para dividir un nodo
        min_samples_leaf=10,           # Mínimo de muestras por hoja
        criterion='entropy'           # Usar 'entropy' en lugar de 'gini'
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

# Imprimir resultados en consola
print(f"Accuracy general del modelo: {overall_accuracy:.2f}")
print(f"Precision global (ponderada): {overall_precision:.2f}")
print(f"Recall global (ponderado): {overall_recall:.2f}")
print(f"F1-score global (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación general:\n {overall_report}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Número total de iteraciones realizadas: {iteration_count}")

#%%
# Guardar resultados en archivo .text
output_path = r'.\..\output\DT_CENCOSUD_con_ajuste.txt'
with open(output_path, 'w') as f:
    f.write(f"Resultados para el Árbol de Decisión con cambios de hiperparametros:\n")
    f.write(f"Accuracy general del modelo: {overall_accuracy:.2f}\n")
    f.write(f"Precision global (ponderada): {overall_precision:.2f}\n")
    f.write(f"Recall global (ponderado): {overall_recall:.2f}\n")
    f.write(f"F1-score global (ponderado): {overall_f1:.2f}\n")
    f.write(f"Reporte de clasificación general:\n{overall_report}\n")
    f.write(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}\n")
    f.write(f"Número total de iteraciones realizadas: {iteration_count}\n")



#%%

# XGBoost

# Codificación de las etiquetas (transformación de [-1, 0, 1] a [0, 1, 2])
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Aplicar RFE para seleccionar las mejores características utilizando XGBoost
model_rfe = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
rfe = RFE(model_rfe, n_features_to_select=7)  # Ajusta n_features_to_select según lo desees
#X_rfe = rfe.fit_transform(X_ta_scaled, y)
X_rfe = rfe.fit_transform(X_ta_scaled, y_encoded)
selected_technical_features = np.array(ta_cols)[rfe.support_]

print("Indicadores técnicos seleccionados por RFE:", selected_technical_features)

# Crear DataFrame con características seleccionadas y concatenar con columnas estáticas normalizadas
X_rfe_df = pd.DataFrame(X_rfe, columns=selected_technical_features)

X_final = pd.concat([X_rfe_df, X_static_df.reset_index(drop=True)], axis=1)
selected_technical_features = np.array(ta_cols)[rfe.support_]

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

    X_train, X_test = X_final[train_indices], X_final[test_indices]
    y_train, y_test = y_encoded[train_indices], y_encoded[test_indices]

    # Verificar clases en el conjunto de entrenamiento
    if len(np.unique(y_train)) < 2 or len(X_train) == 0 or len(X_test) == 0:
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

print(f"Precisión general del modelo XgBoost: {overall_accuracy:.2f}")
print(f"Precision global (ponderada): {overall_precision:.2f}")
print(f"Recall global (ponderado): {overall_recall:.2f}")
print(f"F1-score global (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación general:\n {overall_report}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Número total de iteraciones realizadas: {iteration_count}")

#%%
# Guardar resultados en archivo .text
output_path = r'.\..\output\XGBoost_CENCOSUD_con_100_submodels.txt'
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

#Xgboost Segun paper

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

    X_train, X_test = X_final[train_indices], X_final[test_indices]
    y_train, y_test = y_encoded[train_indices], y_encoded[test_indices]

    # Verificar clases en el conjunto de entrenamiento
    if len(np.unique(y_train)) < 2 or len(X_train) == 0 or len(X_test) == 0:
        continue

    # Crear y ajustar el modelo XGBoost con 1000 submodelos
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

print(f"Precisión general del modelo XGBoost: {overall_accuracy:.2f}")
print(f"Precision global (ponderada): {overall_precision:.2f}")
print(f"Recall global (ponderado): {overall_recall:.2f}")
print(f"F1-score global (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación general:\n {overall_report}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Número total de iteraciones realizadas: {iteration_count}")

#%%
# Guardar resultados en archivo .text
output_path = r'.\..\output\XGBoost_CENCOSUD_Segun_paper.txt'
with open(output_path, 'w') as f:
    f.write(f"Resultados para el modelo XGBoost Segun el paper 1000 submodelos:\n")
    f.write(f"Precisión general del modelo: {overall_accuracy:.2f}\n")
    f.write(f"Precision global (ponderada): {overall_precision:.2f}\n")
    f.write(f"Recall global (ponderado): {overall_recall:.2f}\n")
    f.write(f"F1-score global (ponderado): {overall_f1:.2f}\n")
    f.write(f"Reporte de clasificación general:\n{overall_report}\n")
    f.write(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}\n")
    f.write(f"Número total de iteraciones realizadas: {iteration_count}\n")


#%%

# Random Forest segun el paper

# Codificación de las etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Aplicar RFE para seleccionar las mejores características utilizando Random Forest
model_rfe = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(model_rfe, n_features_to_select=7)
X_rfe = rfe.fit_transform(X_ta_scaled, y_encoded)
selected_technical_features = np.array(ta_cols)[rfe.support_]

print("Indicadores técnicos seleccionados por RFE:", selected_technical_features)

# Crear DataFrame con características seleccionadas y concatenar con columnas estáticas normalizadas
X_rfe_df = pd.DataFrame(X_rfe, columns=selected_technical_features)

X_final = pd.concat([X_rfe_df, X_static_df.reset_index(drop=True)], axis=1)

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

    X_train, X_test = X_final[train_indices], X_final[test_indices]
    y_train, y_test = y_encoded[train_indices], y_encoded[test_indices]

    # Verificar clases en el conjunto de entrenamiento
    if len(np.unique(y_train)) < 2 or len(X_train) == 0 or len(X_test) == 0:
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

print(f"Precisión general del modelo random forest: {overall_accuracy:.2f}")
print(f"Precision global (ponderada): {overall_precision:.2f}")
print(f"Recall global (ponderado): {overall_recall:.2f}")
print(f"F1-score global (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación general:\n {overall_report}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Número total de iteraciones realizadas: {iteration_count}")

#%%
# Guardar resultados en archivo .text
output_path = r'.\..\output\RF_CENCOSUD_Segun_paper.txt'
with open(output_path, 'w') as f:
    f.write(f"Resultados para el modelo de Random Forest Segun el paper con 100 submodelos:\n")
    f.write(f"Precisión general del modelo: {overall_accuracy:.2f}\n")
    f.write(f"Precision global (ponderada): {overall_precision:.2f}\n")
    f.write(f"Recall global (ponderado): {overall_recall:.2f}\n")
    f.write(f"F1-score global (ponderado): {overall_f1:.2f}\n")
    f.write(f"Reporte de clasificación general:\n{overall_report}\n")
    f.write(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}\n")
    f.write(f"Número total de iteraciones realizadas: {iteration_count}\n")

# %%

#Random forest con más hiperparametros para ver su rendimiento, se aumenta de 100 a 1000 submodelos y la profundidad
# de la hoja aumenta de 5 a 10, dado que anteriormente en arbol de decisión obtuvimos mejor rendimiento
# aumentando este parametro

#Se demora al rededor de 18 minutos


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

    X_train, X_test = X_final[train_indices], X_final[test_indices]
    y_train, y_test = y_encoded[train_indices], y_encoded[test_indices]

    # Verificar clases en el conjunto de entrenamiento
    if len(np.unique(y_train)) < 2 or len(X_train) == 0 or len(X_test) == 0:
        continue

    model = RandomForestClassifier(n_estimators=1000, max_depth=10, min_samples_split=2,
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
# Guardar resultados en archivo .text
output_path = r'.\..\output\RF_CENCOSUD_ajustado.txt'
with open(output_path, 'w') as f:
    f.write(f"Resultados para el modelo de Random Forest:\n")
    f.write(f"Precisión general del modelo: {overall_accuracy:.2f}\n")
    f.write(f"Precision global (ponderada): {overall_precision:.2f}\n")
    f.write(f"Recall global (ponderado): {overall_recall:.2f}\n")
    f.write(f"F1-score global (ponderado): {overall_f1:.2f}\n")
    f.write(f"Reporte de clasificación general:\n{overall_report}\n")
    f.write(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}\n")
    f.write(f"Número total de iteraciones realizadas: {iteration_count}\n")
    
    
#%%


# Naive Bayes

# Codificación de las etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

selector = SelectKBest(score_func=f_classif, k=7) #K = 7 igual que RFE = 7
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

    # Verificar clases en el conjunto de entrenamiento
    if len(np.unique(y_train)) < 2 or len(X_train) == 0 or len(X_test) == 0:
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

print(f"Precisión general del modelo naive bayes: {overall_accuracy:.2f}")
print(f"Precision global (ponderada): {overall_precision:.2f}")
print(f"Recall global (ponderado): {overall_recall:.2f}")
print(f"F1-score global (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación general:\n {overall_report}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Número total de iteraciones realizadas: {iteration_count}")

#%%
# Guardar resultados en archivo .text
output_path = r'.\..\output\NB_CENCOSUD_SelectKBest.txt'
with open(output_path, 'w') as f:
    f.write(f"Resultados para el Naive Bayes:\n")
    f.write(f"Precisión general del modelo: {overall_accuracy:.2f}\n")
    f.write(f"Precision global (ponderada): {overall_precision:.2f}\n")
    f.write(f"Recall global (ponderado): {overall_recall:.2f}\n")
    f.write(f"F1-score global (ponderado): {overall_f1:.2f}\n")
    f.write(f"Reporte de clasificación general:\n{overall_report}\n")
    f.write(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}\n")
    f.write(f"Número total de iteraciones realizadas: {iteration_count}\n")
    
    
#%%

# Naive Bayes con random forest, aqui tengamos ojo, dado que estamos entrenando el naive bayes también con hiperparametros
# del algoritmo random forest, o los hiperaprametros son para calcular el RFE?, duda aquí.....!!!!

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

    X_train, X_test = X_final[train_indices], X_final[test_indices]
    y_train, y_test = y_encoded[train_indices], y_encoded[test_indices]

    # Verificar clases en el conjunto de entrenamiento
    if len(np.unique(y_train)) < 2 or len(X_train) == 0 or len(X_test) == 0:
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
# Guardar resultados en archivo .text
output_path = r'.\..\output\NB_CENCOSUD_randomforest.txt'
with open(output_path, 'w') as f:
    f.write(f"Resultados para el Naive Bayes son 100submodelos para random forest:\n")
    f.write(f"Precisión general del modelo: {overall_accuracy:.2f}\n")
    f.write(f"Precision global (ponderada): {overall_precision:.2f}\n")
    f.write(f"Recall global (ponderado): {overall_recall:.2f}\n")
    f.write(f"F1-score global (ponderado): {overall_f1:.2f}\n")
    f.write(f"Reporte de clasificación general:\n{overall_report}\n")
    f.write(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}\n")
    f.write(f"Número total de iteraciones realizadas: {iteration_count}\n")




#%%


# MLP (Multi-Layer Perceptron) con random forest sin hiperparametros del paper

# Codificación de las etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Aplicar RFE para seleccionar las mejores características utilizando Random Forest
model_rfe = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(model_rfe, n_features_to_select=7)  # Ajusta n_features_to_select según lo desees
X_rfe = rfe.fit_transform(X_ta_scaled, y_encoded)
selected_technical_features = np.array(ta_cols)[rfe.support_]

print("Indicadores técnicos seleccionados por RFE:", selected_technical_features)

# Crear DataFrame con características seleccionadas y concatenar con columnas estáticas normalizadas
X_rfe_df = pd.DataFrame(X_rfe, columns=selected_technical_features)

X_final = pd.concat([X_rfe_df, X_static_df.reset_index(drop=True)], axis=1)

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

    X_train, X_test = X_final[train_indices], X_final[test_indices]
    y_train, y_test = y_encoded[train_indices], y_encoded[test_indices]

    # Verificar clases en el conjunto de entrenamiento
    if len(np.unique(y_train)) < 2 or len(X_train) == 0 or len(X_test) == 0:
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
output_path = r'.\..\output\MLP_CENCOSUD_basico_no_paper.text'
with open(output_path, 'w') as f:
    f.write(f"Resultados para el MLP:\n")
    f.write(f"Precisión general del modelo: {overall_accuracy:.2f}\n")
    f.write(f"Precision global (ponderada): {overall_precision:.2f}\n")
    f.write(f"Recall global (ponderado): {overall_recall:.2f}\n")
    f.write(f"F1-score global (ponderado): {overall_f1:.2f}\n")
    f.write(f"Reporte de clasificación general:\n{overall_report}\n")
    f.write(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}\n")
    f.write(f"Número total de iteraciones realizadas: {iteration_count}\n")


#%%

#MLP con random Forest e hiperaparametros del paper

# Codificación de las etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

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

    X_train, X_test = X_final[train_indices], X_final[test_indices]
    y_train, y_test = y_encoded[train_indices], y_encoded[test_indices]

    # Verificar clases en el conjunto de entrenamiento
    if len(np.unique(y_train)) < 2 or len(X_train) == 0 or len(X_test) == 0:
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

print(f"Resultados para el MLP:\n")
print(f"Precisión general del modelo: {overall_accuracy:.2f}")
print(f"Precision global (ponderada): {overall_precision:.2f}")
print(f"Recall global (ponderado): {overall_recall:.2f}")
print(f"F1-score global (ponderado): {overall_f1:.2f}")
print(f"Reporte de clasificación general:\n{overall_report}")
print(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}")
print(f"Número total de iteraciones realizadas: {iteration_count}")

#%%
# Guardar resultados en archivo .text
output_path = r'.\..\output\MLP_CENCOSUD_segun_paper.txt'
with open(output_path, 'w') as f:
    f.write(f"Resultados para el MLP con hiperparametros del paper 3 capas y 30 neuronas de capa oculta, y utilizamos 1000 submodelos:\n")
    f.write(f"Precisión general del modelo: {overall_accuracy:.2f}\n")
    f.write(f"Precision global (ponderada): {overall_precision:.2f}\n")
    f.write(f"Recall global (ponderado): {overall_recall:.2f}\n")
    f.write(f"F1-score global (ponderado): {overall_f1:.2f}\n")
    f.write(f"Reporte de clasificación general:\n{overall_report}\n")
    f.write(f"Tiempo de ejecución: {hora_de_fin - hora_de_inicio}\n")
    f.write(f"Número total de iteraciones realizadas: {iteration_count}\n")
    



#%%

#LSTM se acerca al paper, pero no en su totalidad

# Codificación de las etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Aplicar RFE para seleccionar las mejores características utilizando Random Forest
model_rfe = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(model_rfe, n_features_to_select=7)
X_rfe = rfe.fit_transform(X_ta_scaled, y_encoded)
selected_technical_features = np.array(ta_cols)[rfe.support_]

print("Indicadores técnicos seleccionados por RFE:", selected_technical_features)

# Crear DataFrame con características seleccionadas y concatenar con columnas estáticas normalizadas
X_rfe_df = pd.DataFrame(X_rfe, columns=selected_technical_features)

X_final = pd.concat([X_rfe_df, X_static_df.reset_index(drop=True)], axis=1)

# Acumular todas las predicciones y etiquetas verdaderas
all_y_true = []
all_y_pred = []

# Contador de iteraciones
iteration_count = 0

# Definir parámetros para la ventana deslizante
n_input = 20  # Pasos de tiempo
n_features = X_final.shape[1]  # Asegurarse de que n_features refleje el número de características en X_final

# Iterar sobre el 20% restante usando la ventana deslizante
for start in range(train_size, len(dates) - n_input):
    # Definir índices de entrenamiento y prueba, ajustando el rango de prueba para tener al menos n_input elementos
    test_indices = (dates >= dates.iloc[start]) & (dates < dates.iloc[start + n_input + 1])
    train_indices = dates < dates.iloc[start]

    X_train, X_test = X_final[train_indices], X_final[test_indices]
    y_train, y_test = y_encoded[train_indices], y_encoded[test_indices]

    # Verificar que haya suficientes clases en el conjunto de entrenamiento
    if len(np.unique(y_train)) < 2 or len(X_train) < n_input:
        continue  # Si solo hay una clase o no hay suficientes datos, pasa a la siguiente iteración

    # Crear el generador de secuencias para el entrenamiento
    train_generator = TimeseriesGenerator(X_train, y_train, length=n_input, batch_size=32)

    # Construcción del modelo LSTM
    model = Sequential()
    model.add(LSTM(150, activation='tanh', return_sequences=True, input_shape=(n_input, n_features)))
    model.add(Dropout(0.3))
    model.add(LSTM(100, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(len(label_encoder.classes_), activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Entrenamiento del modelo
    model.fit(train_generator, epochs=50, verbose=0)

    # Verificar que haya suficientes datos en X_test para el generador de prueba
    if len(X_test) <= n_input:
        continue  # Asegurarse de que hay suficientes datos para crear el generador de prueba

    # Crear el generador de secuencias para la prueba
    test_generator = TimeseriesGenerator(X_test, y_test, length=n_input, batch_size=1)

    # Predicción
    y_pred = model.predict(test_generator)

    # Convertir las predicciones a clases
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_aligned = y_test[n_input-1:]  # Ajustar y_test para alinear con y_pred

    # Acumular las predicciones y etiquetas verdaderas
    all_y_true.extend(y_test_aligned)
    all_y_pred.extend(y_pred_classes)

    # Incrementar el contador de iteraciones
    iteration_count += 1

# Convertir las listas acumuladas a arrays de NumPy
all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)

# Evaluación del modelo
accuracy = accuracy_score(all_y_true, all_y_pred)
report = classification_report(all_y_true, all_y_pred, target_names=[str(cls) for cls in label_encoder.classes_])
#%%
# Guardar resultados en un archivo de texto
with open('./output/LSTM_CENCOSUD.txt', 'w') as f:
    f.write(f"Precisión del modelo: {accuracy:.2f}\n")
    f.write("Reporte de clasificación para LSTM:\n" + report)

print(f"Precisión del modelo: {accuracy:.2f}")
print("Reporte de clasificación para LSTM:\n", report)
print(f"Número total de iteraciones realizadas: {iteration_count}")

# %%



