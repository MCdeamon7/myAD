# train_rnn_model.py
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from elasticsearch import Elasticsearch

# Connessione a OpenSearch
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

# Esempio: esegui una query per ottenere i dati dal tuo indice OpenSearch
query = {
    "size": 1000,  # Imposta la dimensione in base alle tue esigenze
    "query": {
        "match_all": {}
    }
}

response = es.search(index="nome_tuo_indice", body=query)
hits = response["hits"]["hits"]

# Estrai i dati dalla risposta di OpenSearch
data = [hit["_source"] for hit in hits]

# Trasforma i dati in input per i modelli
device_models = {}  # Dizionario per memorizzare i modelli per ogni "mac address"

for entry in data:
    mac_address = entry['mac_address']
    
    if mac_address not in device_models:
        device_models[mac_address] = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, input_shape=(sequence_length, len(selected_columns) + 12)),  # 12 per le colonne one-hot
            tf.keras.layers.Dense(len(selected_columns))
        ])

        device_models[mac_address].compile(optimizer='adam', loss='mse')

    # Trasforma i dati in input per il modello
    df = pd.DataFrame(data)  # Converte la lista di dizionari in un DataFrame pandas

    # Aggiunta di colonne per l'ora del giorno e il giorno della settimana
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    # Selezione delle colonne di interesse
    selected_columns = ['cpu', 'cpu_max', 'ram', 'rx_bytes', 'tx_bytes', 'rx_packets', 'tx_packets', 'hour', 'day_of_week']

    # Normalizzazione dei dati
    scaler = StandardScaler()
    df_normalized = df[selected_columns].apply(lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten())

    # Preparazione dei dati per la RNN
    sequence_length = 10
    X, y = [], []

    for i in range(len(df) - sequence_length):
        X.append(df_normalized.iloc[i:i+sequence_length].values)
        y.append(df_normalized.iloc[i+sequence_length].values)

    X = np.array(X)
    y = np.array(y)

    # Codifica one-hot per ora del giorno e giorno della settimana
    encoder = OneHotEncoder(sparse=False)
    hour_onehot = encoder.fit_transform(df['hour'].values.reshape(-1, 1))
    day_of_week_onehot = encoder.fit_transform(df['day_of_week'].values.reshape(-1, 1))

    # Aggiunta delle colonne codificate one-hot a X
    X = np.concatenate([X, hour_onehot[:-sequence_length], day_of_week_onehot[:-sequence_length]], axis=1)

    # Addestramento del modello specifico per il dispositivo
    device_models[mac_address].fit(X, y, epochs=50, verbose=0)

    # Salvataggio del modello specifico per il dispositivo (se necessario)
    device_models[mac_address].save(f'path_to_models/model_{mac_address}')

# Salvataggio di tutti i modelli (se necessario)
for mac_address, model in device_models.items():
    model.save(f'path_to_models/model_{mac_address}')