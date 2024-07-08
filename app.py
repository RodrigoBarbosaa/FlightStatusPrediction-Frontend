from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

# Carregar o modelo treinado

origins = ['MCO', 'LAX']  
destinations = ['SFO', 'ATL']

dias_char = {"Sunday": 1, "Monday": 2, "Tuesday": 3, "Wednesday": 4, "Thursday": 5, "Friday": 6, "Saturday": 7}

airlines = [
    'American Airlines Inc.', 'Delta Air Lines Inc.', 'Frontier Airlines Inc.',
    'Horizon Air', 'JetBlue Airways', 'SkyWest Airlines Inc.',
    'Southwest Airlines Co.', 'Spirit Air Lines', 'United Air Lines Inc.'
    ]

@app.route('/')
def index():
    routes = ["Los angeles(LAX) --> San francisco(SFO)", "Orlando(MCO) --> Atlanta(ATL)"]
    
    dias = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    
    return render_template('index.html', days=dias, routes=routes, airlines=airlines)

@app.route('/predict', methods=['POST'])
def predict():
    route = request.form['route']
    month = int(request.form['month'])
    day_of_week_str = request.form['day_of_week']
    airline = request.form['airline']
    
    day_of_week = dias_char[day_of_week_str]
    
    origin = int()
    dest = int()
    
    if route == "Los angeles(LAX) --> San francisco(SFO)":
        origin = "LAX"
        dest = "SFO"
    else:
        origin = "MCO"
        dest = "ATL"
    
    airline_encoded = [1 if airline == a else 0 for a in airlines]
    
    # Simplificar o encoding das rotas
    origin_encoded = 1 if origin == 'MCO' else 0
    dest_encoded = 1 if dest == 'SFO' else 0
    
    input_features = [month, 0, day_of_week] + airline_encoded + [origin_encoded, dest_encoded]
    print(input_features)
    input_features = np.array(input_features).reshape(1, -1)
    print(input_features)
    
    # Fazer predição
    
    return render_template('index.html', prediction_text=f"Predição de Atraso: 10 minutos")

if __name__ == '__main__':
    app.run(debug=False)
