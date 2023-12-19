# saved_model_inference_service.py
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from elasticsearch import Elasticsearch
from sklearn.preprocessing import StandardScaler, OneHotEncoder


app = Flask(__name__)

# Carica tutti i modelli specifici per il dispositivo
device_models = {}

# Carica i modelli da file o connessione a OpenSearch (a seconda delle tue esigenze)
for mac_address in lista_mac_address:
    device_models[mac_address] = tf.keras.models.load_model(f'path_to_models/model_{mac_address}')

@app.route('/predict', methods=['POST'])
def predict():
    # Connessione a OpenSearch (se necessario)
    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

    data = request.json['data']
    mac_address = data['mac_address']

    if mac_address in device_models:
        model = device_models[mac_address]

        # Esempio: esegui una query per ottenere dati specifici per il dispositivo da OpenSearch
        query = {
            "size": 1,
            "query": {
                "term": {"mac_address": mac_address}
            }
        }

        response = es.search(index="nome_tuo_indice", body=query)
        new_data = [hit["_source"] for hit in response["hits"]["hits"]]

        # Assicurati di applicare le stesse trasformazioni utilizzate durante l'addestramento

        # Selezione delle colonne di interesse
        selected_columns = ['cpu', 'cpu_max', 'ram', 'rx_bytes', 'tx_bytes', 'rx_packets', 'tx_packets', 'hour', 'day_of_week']

        # Normalizzazione dei dati
        scaler = StandardScaler()
        new_data_normalized = new_data[selected_columns].apply(lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten())


        # Esempio di aggiunta di colonne per ora del giorno e giorno della settimana
        new_data['hour'] = new_data['timestamp'].dt.hour
        new_data['day_of_week'] = new_data['timestamp'].dt.dayofweek

        # Codifica one-hot per ora del giorno e giorno della settimana
        encoder = OneHotEncoder(sparse=False)
        hour_onehot = encoder.fit_transform(new_data['hour'].values.reshape(-1, 1))
        day_of_week_onehot = encoder.fit_transform(new_data['day_of_week'].values.reshape(-1, 1))

        # Concatenazione delle colonne al formato richiesto dal modello
        new_data_for_inference = np.concatenate([new_data_normalized, hour_onehot, day_of_week_onehot], axis=1)

        # Esecuzione di inferenza sul modello specifico per il dispositivo
        predictions = model.predict(np.array([new_data_for_inference]))

        # Esecuzione di inferenza sul modello specifico per il dispositivo
        predictions = model.predict(np.array(new_data))

        return jsonify(predictions.tolist())
    else:
        return jsonify({"error": "Model not found for the specified mac_address"})

if __name__ == '__main__':
    app.run(port=5000)
