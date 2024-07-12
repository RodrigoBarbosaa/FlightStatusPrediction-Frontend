from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

# Carregar o modelo treinado

origins = ['LAX', 'MCO', 'DEN', 'FLL', 'SEA', 'MIA', 'ORD', 'JFK', 'SLC', 'SFO', 'LAS', 'PHX', 'GEG']
destinations = ['SFO', 'PHX', 'ATL', 'PDX', 'DEN', 'LAS', 'SEA', 'LAX']
dias = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

dias_char = {"Sunday": 1, "Monday": 2, "Tuesday": 3, "Wednesday": 4, "Thursday": 5, "Friday": 6, "Saturday": 7}

airlines = [
    'American Airlines Inc.', 'Delta Air Lines Inc.', 'Frontier Airlines Inc.',
    'Horizon Air', 'JetBlue Airways', 'SkyWest Airlines Inc.',
    'Southwest Airlines Co.', 'Spirit Air Lines', 'United Air Lines Inc.'
    ]
routes = ["Los Angeles (LAX) --> San Francisco (SFO)",
            "Orlando (MCO) --> Atlanta (ATL)",
            "Denver (DEN) --> Phoenix (PHX)",
            "Atlanta (ATL) --> New York (JFK)",
            "Fort Lauderdale (FLL) --> Atlanta (ATL)",
            "Seattle (SEA) --> Phoenix (PHX)",
            "Seattle (SEA) --> Los Angeles (LAX)",
            "Chicago (ORD) --> Denver (DEN)",
            "Miami (MIA) --> Atlanta (ATL)",
            "New York (JFK) --> Denver (DEN)",
            "Salt Lake City (SLC) --> Los Angeles (LAX)",
            "Los Angeles (LAX) --> Las Vegas (LAS)",
            "Orlando (MCO) --> Seattle (SEA)",
            "Chicago (ORD) --> San Francisco (SFO)",
            "Seattle (SEA) --> San Francisco (SFO)",
            "Phoenix (PHX) --> Seattle (SEA)",
            "Spokane (GEG) --> Seattle (SEA)",
            "Las Vegas (LAS) --> Seattle (SEA)"]

@app.route('/')
def index():
    return render_template('index.html', days=dias, routes=routes, airlines=airlines)

@app.route('/predict', methods=['POST'])
def predict():
    route = request.form['route']
    month = int(request.form['month'])
    day_of_week_str = request.form['day_of_week']
    day_of_month = int(request.form['day_of_month'])
    airline = request.form['airline']
    
    origin = ""
    dest = ""
    
    if route == "Los Angeles (LAX) --> San Francisco (SFO)":
        origin = "LAX"
        dest = "SFO"
    elif route == "Orlando (MCO) --> Atlanta (ATL)":
        origin = "MCO"
        dest = "ATL"
    elif route == "Denver (DEN) --> Phoenix (PHX)":
        origin = "DEN"
        dest = "PHX"
    elif route == "Atlanta (ATL) --> New York (JFK)":
        origin = "ATL"
        dest = "JFK"
    elif route == "Fort Lauderdale (FLL) --> Atlanta (ATL)":
        origin = "FLL"
        dest = "ATL"
    elif route == "Seattle (SEA) --> Phoenix (PHX)":
        origin = "SEA"
        dest = "PHX"
    elif route == "Seattle (SEA) --> Los Angeles (LAX)":
        origin = "SEA"
        dest = "LAX"
    elif route == "Chicago (ORD) --> Denver (DEN)":
        origin = "ORD"
        dest = "DEN"
    elif route == "Miami (MIA) --> Atlanta (ATL)":
        origin = "MIA"
        dest = "ATL"
    elif route == "New York (JFK) --> Denver (DEN)":
        origin = "JFK"
        dest = "DEN"
    elif route == "Salt Lake City (SLC) --> Los Angeles (LAX)":
        origin = "SLC"
        dest = "LAX"
    elif route == "Los Angeles (LAX) --> Las Vegas (LAS)":
        origin = "LAX"
        dest = "LAS"
    elif route == "Orlando (MCO) --> Seattle (SEA)":
        origin = "MCO"
        dest = "SEA"
    elif route == "Chicago (ORD) --> San Francisco (SFO)":
        origin = "ORD"
        dest = "SFO"
    elif route == "Seattle (SEA) --> San Francisco (SFO)":
        origin = "SEA"
        dest = "SFO"
    elif route == "Phoenix (PHX) --> Seattle (SEA)":
        origin = "PHX"
        dest = "SEA"
    elif route == "Spokane (GEG) --> Seattle (SEA)":
        origin = "GEG"
        dest = "SEA"
    elif route == "Las Vegas (LAS) --> Seattle (SEA)":
        origin = "LAS"
        dest = "SEA"

    day_of_week = dias_char[day_of_week_str]

    # Lista de colunas no dataframe
    columns = ['Month', 'DayofMonth', 'DayOfWeek', 'Airline_American Airlines Inc.', 'Airline_Delta Air Lines Inc.', 'Airline_Envoy Air', 'Airline_Frontier Airlines Inc.', 'Airline_Horizon Air', 'Airline_JetBlue Airways', 'Airline_Republic Airlines', 'Airline_SkyWest Airlines Inc.', 'Airline_Southwest Airlines Co.', 'Airline_Spirit Air Lines', 'Airline_United Air Lines Inc.', 'Origin_DEN', 'Origin_FLL', 'Origin_GEG', 'Origin_JFK', 'Origin_LAS', 'Origin_LAX', 'Origin_MCO', 'Origin_MIA', 'Origin_ORD', 'Origin_PHX', 'Origin_SEA', 'Origin_SFO', 'Origin_SLC', 'Dest_DEN', 'Dest_FLL', 'Dest_LAS', 'Dest_LAX', 'Dest_PDX', 'Dest_PHX', 'Dest_SEA', 'Dest_SFO']

    # Codificação das variáveis categóricas
    airline_encoded = [1 if airline == col.split('_')[1] else 0 for col in columns if 'Airline_' in col]
    origin_encoded = [1 if origin == col.split('_')[1] else 0 for col in columns if 'Origin_' in col]
    dest_encoded = [1 if dest == col.split('_')[1] else 0 for col in columns if 'Dest_' in col]

    # Criação do dicionário de entrada
    input_dict = dict.fromkeys(columns, 0)
    input_dict['Month'] = month
    input_dict['DayofMonth'] = day_of_month
    input_dict['DayOfWeek'] = day_of_week

    # Atualização do dicionário com os valores codificados
    for idx, col in enumerate([col for col in columns if 'Airline_' in col]):
        input_dict[col] = airline_encoded[idx]
    for idx, col in enumerate([col for col in columns if 'Origin_' in col]):
        input_dict[col] = origin_encoded[idx]
    for idx, col in enumerate([col for col in columns if 'Dest_' in col]):
        input_dict[col] = dest_encoded[idx]

    # Conversão do dicionário para DataFrame
    input_df = pd.DataFrame([input_dict])
    # Carregar o modelo previamente treinado
    best_xgb_model = joblib.load('best_xgb_model.pkl')

    # Carregar o label encoder previamente salvo
    label_encoder = joblib.load('label_encoder.pkl')

    pred_proba = best_xgb_model.predict_proba(input_df)
    
    # Pegar a classe com a maior probabilidade
    predicted_class = np.argmax(pred_proba, axis=1)
    predicted_class_label = label_encoder.inverse_transform(predicted_class)

    if predicted_class_label[0] == "Atraso Leve":
        texto = "Small delay, up to 30 minutes"
    elif predicted_class_label[0] == "Atraso Moderado":
        texto = "Moderate delay, up to 60 minutes"
    else:
        texto = "No delay expected"
    # Fazer predição
    
    return render_template('index.html', prediction_text=texto, days=dias, routes=routes, airlines=airlines)

if __name__ == '__main__':
    app.run(debug=False)
