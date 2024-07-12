import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import zscore
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, auc
from xgboost import XGBClassifier
import joblib

# Filtrando o Dataset para anáilise
flights = pd.read_csv("df_selected_routes_grande.csv", encoding = "ISO-8859-1")

# Normalizar colunas específicas com z-score
cols_to_normalize = ['Month', 'DayofMonth', 'DayOfWeek']
scaler = StandardScaler()
df_selected_routes_grande[cols_to_normalize] = scaler.fit_transform(df_selected_routes_grande[cols_to_normalize])

# Codificar colunas categóricas
df_selected_routes_grande = pd.get_dummies(df_selected_routes_grande, columns=['Airline', 'Origin', 'Dest'], drop_first=True)

# Selecionar features e variável alvo
X = df_selected_routes_grande.drop(columns=['CategoriaAtraso', 'FlightDate', 'OriginCityName', 'DestCityName', 'DepDelayMinutes'])
y = df_selected_routes_grande['CategoriaAtraso']

# Codificar as classes de 'y' como valores numéricos
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_selected_routes_grande['CategoriaAtraso'])
# Salvar o label encoder
joblib.dump(label_encoder, 'label_encoder.pkl')

# Dividir os dados em conjuntos de treinamento, validação e teste com estratificação
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Treinar o modelo XGBoost
best_xgb_model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss')
best_xgb_model.fit(X_train, y_train)

joblib.dump(best_xgb_model, 'best_xgb_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Modelo treinado e salvo com sucesso!")
