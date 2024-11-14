# Importar librerías
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

ruta_input = "../input/"
ruta_output = "../output/"

archivos = os.listdir(ruta_input)

window_size = 3
train_ratio = 0.8

modelos = {
    "LR": LogisticRegression(max_iter=1000, penalty='l2'),
    "DT": DecisionTreeClassifier(random_state=42, max_depth=10),
    "XGB": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "RF": RandomForestClassifier(n_estimators=100, random_state=42),
    "NB": GaussianNB(),
    "MLP": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)
}

empresas_ejecutadas = ['AGUAS-A']

for archivo in archivos:
    empresa = archivo.split(".")[0]
    if empresa in empresas_ejecutadas:
        continue
        
    ruta_archivo = os.path.join(ruta_input, archivo)
    
    df = pd.read_csv(ruta_archivo)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    
    y = df['Tendencia']
    X = df.drop(columns=['Date', 'Tendencia'])
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    dates = pd.to_datetime(df['Date'])
    train_size = int(len(dates) * train_ratio)
    
    for nombre_modelo, modelo in modelos.items():

        X_final = pd.DataFrame(X_scaled, columns=X.columns)
        
        y_true, y_pred = [], []
        for start in range(train_size, len(dates) - window_size):
            train_indices = dates < dates.iloc[start]
            test_indices = (dates >= dates.iloc[start]) & (dates < dates.iloc[start + window_size])
            
            X_train, X_test = X_final[train_indices], X_final[test_indices]
            y_train, y_test = y_encoded[train_indices], y_encoded[test_indices]
            
            if len(np.unique(y_train)) < 2:
                continue
            
            modelo.fit(X_train, y_train)
            y_pred.extend(modelo.predict(X_test))
            y_true.extend(y_test)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        report = classification_report(y_true, y_pred)
        
        output_file = f"{empresa}_{nombre_modelo}_metrics.txt"
        with open(os.path.join(ruta_output, output_file), 'w') as f:
            f.write(f"Resultados para {nombre_modelo} en {empresa}:\n")
            f.write(f"Accuracy: {accuracy:.2f}\n")
            f.write(f"Precision: {precision:.2f}\n")
            f.write(f"Recall: {recall:.2f}\n")
            f.write(f"F1-score: {f1:.2f}\n")
            f.write(f"Reporte de clasificación:\n{report}\n")
        
        print(f"Guardado de resultados completado para {nombre_modelo} en {empresa}")
    
    # Entrenamiento y evaluación del modelo LSTM
    print(f"Entrenando el modelo LSTM para {empresa}...")
    n_input = 20
    n_features = X_scaled.shape[1]
    
    y_encoded_lstm = np.eye(len(np.unique(y_encoded)))[y_encoded]
    
    train_generator = TimeseriesGenerator(X_scaled, y_encoded_lstm, length=n_input, batch_size=32)
    
    # Definir el modelo LSTM
    model_lstm = Sequential()
    model_lstm.add(LSTM(150, activation='tanh', return_sequences=True, input_shape=(n_input, n_features)))
    model_lstm.add(Dropout(0.3))
    model_lstm.add(LSTM(100, activation='tanh'))
    model_lstm.add(Dropout(0.3))
    model_lstm.add(Dense(100, activation='relu'))
    model_lstm.add(Dense(len(np.unique(y_encoded)), activation='softmax'))
    model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    model_lstm.fit(train_generator, epochs=50, verbose=1)
    
    test_generator = TimeseriesGenerator(X_scaled[-(len(X_scaled) + n_input):], y_encoded_lstm[-(len(X_scaled) + n_input):], length=n_input, batch_size=1)
    
    y_pred_lstm = model_lstm.predict(test_generator)
    
    y_pred_classes = np.argmax(y_pred_lstm, axis=1)
    y_test_aligned = np.argmax(y_encoded_lstm[-len(y_pred_classes):], axis=1)

    accuracy_lstm = accuracy_score(y_test_aligned, y_pred_classes)
    precision_lstm = precision_score(y_test_aligned, y_pred_classes, average='weighted')
    recall_lstm = recall_score(y_test_aligned, y_pred_classes, average='weighted')
    f1_lstm = f1_score(y_test_aligned, y_pred_classes, average='weighted')
    report_lstm = classification_report(y_test_aligned, y_pred_classes)
    
    output_file_lstm = f"{empresa}_LSTM_metrics.txt"
    with open(os.path.join(ruta_output, output_file_lstm), 'w') as f:
        f.write(f"Resultados para LSTM en {empresa}:\n")
        f.write(f"Accuracy: {accuracy_lstm:.2f}\n")
        f.write(f"Precision: {precision_lstm:.2f}\n")
        f.write(f"Recall: {recall_lstm:.2f}\n")
        f.write(f"F1-score: {f1_lstm:.2f}\n")
        f.write(f"Reporte de clasificación:\n{report_lstm}\n")
    
    print(f"Guardado de resultados completado para LSTM en {empresa}")
    empresas_ejecutadas.append(empresa)
