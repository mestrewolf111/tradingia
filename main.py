import pandas as pd
import time
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM
from sklearn.preprocessing import MinMaxScaler
from iqoptionapi.stable_api import IQ_Option
from keras.models import load_model
import os
import json
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
except Exception as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


def checktempo():
    while True:
        time.sleep(1)
        if datetime.datetime.now().second == 2:
            break


def checktempo2():
    while True:
        time.sleep(1)
        if datetime.datetime.now().second == 45:
            break


iq = IQ_Option("you email", "you pass")
iq.connect()
par = "EURUSD-OTC"
time_frame = 60
bet_money = 50

def rsi(data, window=16):
    data = pd.DataFrame(data)
    delta = data["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_scaled = (rsi - rsi.min()) / (rsi.max() - rsi.min())  # scale between 0 and 1
    RSI = pd.DataFrame({"RSI": rsi_scaled})
    RSI = RSI.fillna(2)
    return RSI


def stochastic(data, window=16, smooth=3):
    data = pd.DataFrame(data)
    lowest_low = data['min'].rolling(window=window, min_periods=0).min()
    highest_high = data['max'].rolling(window=window, min_periods=0).max()
    k_percent = 100 * (data['close'] - lowest_low) / (highest_high - lowest_low)
    d_percent = k_percent.rolling(window=smooth).mean()
    k_avg = k_percent.mean()
    d_avg = d_percent.mean()
    stochastic = (k_avg + d_avg) / 2
    stochastic = stochastic/100
    return stochastic

def mhi_strategy(data):
    data = pd.DataFrame(data)
    minority = np.where((data['close'].shift(-1) < data['open'].shift(-1)) & (data['close'] < data['open']), 1, 0)
    majority = np.where((data['close'].shift(-1) > data['open'].shift(-1)) & (data['close'] > data['open']), 0, 2)
    mhi = np.where(minority == 1, 1, np.where(majority == 0, 0, 2))
    return pd.Series(mhi)

def get_cores_cruzamento(data):
    data = data
    data['media9'] = data['close'].ewm(span=2, adjust=False).mean()
    data['media21'] = data['close'].ewm(span=9, adjust=False).mean()
    data['trend'] = np.where(data['media9'] > data['media21'], 0,
                                  np.where(data['media21'] > data['media9'], 1,
                                           2))
    data.fillna(2)
    return data['trend']

def get_price_signals(data):
    price_signals = np.zeros(len(data))  # Inicializa a lista de sinais de preço com zeros

    # Defina o número mínimo de velas para considerar um suporte ou resistência
    min_velas = 5

    for i in range(2, len(data)):
        if i >= min_velas:
            is_support = data['min'][i] == min(data['min'][i - min_velas:i + 1])
            is_resistance = data['max'][i] == max(data['max'][i - min_velas:i + 1])
        else:
            is_support = False
            is_resistance = False

        price_signals[i] = np.where(is_support, 2, np.where(is_resistance, -2, 0))
    data['pricesignals'] = pd.DataFrame(price_signals)
    return data['pricesignals']

def getcoresLstm(data):
    data['forca'] = np.where(data['open'] < data['close'], 1,
                             np.where(data['open'] > data['close'], -1,
                                      0))
    forca_total = 0
    # Inicialize uma lista para armazenar as porcentagens de força total
    porcentagens_forca_total = []
    # Defina um limiar para considerar como "muita força"
    limiar = 60
    # Percorra o DataFrame e calcule a porcentagem de força total
    for forca_atual in data['forca']:
        if forca_atual == -1:
            forca_total += -1
        elif forca_atual == 1:
            forca_total += 1
        else:
            forca_total = 0
        # Calcule a porcentagem de força total
        porcentagens_forca_total.append(forca_total)
    # Crie uma coluna no DataFrame para a porcentagem de força total
    data['forca_pct'] = porcentagens_forca_total
    # Ajuste a porcentagem de força para que vá até 0.9 no máximo
    data['forca_pct'] = np.where(data['forca_pct'] > limiar, limiar, data['forca_pct'])
    data['forca_pct'] = data['forca_pct'] / limiar * 0.9

    return data[['forca_pct']]

def get_data_iq(par, iq, time_frame):
    velas = iq.get_candles(par, time_frame, 1000, time.time())
    data = pd.DataFrame(velas)
    data['pips'] = data['close'] - data['open']
    data['verifypips'] = np.where(data['pips'] >= 0.00001, 0,
                                  np.where(data['pips'] <= -0.00001, 1,
                                           2))
    data['cores'] = np.where(data['open'] < data['close'], 1,
                                 np.where(data['open'] > data['close'], 0,
                                          2))
    data['rsi'] = rsi(data, window=16)
    data['stocastico'] = stochastic(data, window=16, smooth=3)
    data['maxcalc'] = data['max'] - data['min']
    data['closexcalc'] = data['open'] - data['close']
    data['SMA1'] = data['close'].rolling(window=3).mean()
    data['SMA21'] = data['close'].rolling(window=50).mean()
    data['SMA9'] = data['close'].rolling(window=12).mean()
    data['SMA4'] = data['close'].rolling(window=9).mean()
    data['medium'] = data['SMA1'] - data['SMA4']
    data['maxcal'] = data['max'] - data['min']
    data['mhi'] = get_cores_cruzamento(data)
    data['pricesignals'] = get_price_signals(data)
    data['tendencia'] = getcoresLstm(data)
    X = data[["tendencia","open","max","min","close","volume","maxcalc","SMA1","SMA4","SMA9","SMA21","closexcalc","rsi","stocastico","verifypips","cores","pricesignals"]]
    X.isna().sum()
    X = X.fillna(2)
    X = X.loc[~X.index.duplicated(keep='first')]
    scaler = MinMaxScaler()
    indexes = X.index
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, index=indexes)
    return X
path = "modelo.h5"
class CustomAgentCheckpoint(keras.Model):
    def __init__(self, num_actions):
        super().__init__()
        self.dense1 = keras.layers.Dense(32, activation="relu")
        self.dropout1 = keras.layers.Dropout(0.2)
        self.batchNorm1 = keras.layers.BatchNormalization()
        self.lstm = LSTM(32)  # Alterei para usar uma camada LSTM
        self.dense2 = keras.layers.Dense(32, activation="relu")
        self.dropout2 = keras.layers.Dropout(0.2)
        self.batchNorm2 = keras.layers.BatchNormalization()
        self.dense3 = keras.layers.Dense(num_actions, activation="softmax")

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dropout1(x)
        x = self.batchNorm1(x)
        x = self.lstm(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        x = self.batchNorm2(x)
        x = self.dense3(x)
        return x
num_actions = 3  # Comprar , Vender, esperar
model = CustomAgentCheckpoint(num_actions)
# Otimizador e função de perda
# Otimizador e função de perda
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.MeanSquaredError()

replay_buffer = []
max_replay_buffer_size = 10000
min_replay_buffer_size = 8
batch_size = 4  # Tamanho do lote para treinamento
gamma = 0.99
epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 0.999
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

# Restaurar o checkpoint se ele existir
if os.path.exists(checkpoint_dir):
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    print('Checkpoint carregado!')

total_reward = 0

# Função para executar uma ação no ambiente
def execute_action(action):
    global total_reward
    trade = False
    print("Action ==>", action)
    print("Rewards ==>", total_reward)

    if action == 0:  # Comprar
        if datetime.datetime.now().second > 25:
            while datetime.datetime.now().second < 58:
                pass
        check, id = iq.buy(bet_money, par, "call", 1)
        print('BUY IN ACTION')
        trade = True
        checar = iq.check_win_v3(id)
    elif action == 1:  # Vender
        if datetime.datetime.now().second > 25:
            while datetime.datetime.now().second < 58:
                pass
        check, id = iq.buy(bet_money, par, "put", 1)
        print('SELL IN ACTION')
        trade = True
        checar = iq.check_win_v3(id)
    else:  # Esperar
        trade = False
        checar = False
        time.sleep(5)

    if checar >= 1 and trade:
        print("WIN")
        print(checar, "PRACAR")
        total_reward += checar
    elif checar <= 0 and trade:
        print(checar, "PRACAR")
        total_reward += checar
        print("LOSS")
    else:
        print('semoperacao')

    return trade, checar

for episode in range(100):
    state = get_data_iq(par, iq, time_frame)
    state = tf.convert_to_tensor(get_data_iq(par, iq, time_frame), dtype=tf.float32)
    done = False
    checar = False

    while not done:
        # Seleciona ação com base na política epsilon-greedy
        if np.random.rand() < epsilon:
            action = np.random.randint(num_actions)
        else:
            q_values = model.predict(state,training=True)
            action = np.argmax(q_values[0])

        # Executa ação no ambiente e observa novo estado e recompensa
        trade, checar = execute_action(action)
        new_state = get_data_iq(par, iq, time_frame)
        done = False
        new_state = tf.convert_to_tensor(get_data_iq(par, iq, time_frame), dtype=tf.float32)
        replay_buffer.append((state, action, total_reward, new_state, done))
        if len(replay_buffer) >= batch_size:
            sample_indices = np.random.choice(len(replay_buffer), batch_size, replace=False)
            minibatch = [replay_buffer[i] for i in sample_indices]

            states = tf.convert_to_tensor([transition[0] for transition in minibatch], dtype=tf.float32)
            actions = np.array([transition[1] for transition in minibatch])
            rewards = np.array([transition[2] for transition in minibatch])
            next_states = tf.convert_to_tensor([transition[3] for transition in minibatch], dtype=tf.float32)

            with tf.GradientTape() as tape:
                target_Qs = model(next_states)
                target_Q = rewards + gamma * tf.reduce_max(target_Qs, axis=1)
                target_Q = tf.stop_gradient(target_Q)
                Q_values = model(states)
                Q_action = tf.reduce_sum(Q_values * tf.one_hot(actions, num_actions), axis=1)
                loss = loss_fn(target_Q, Q_action)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epsilon = max(min_epsilon, epsilon * epsilon_decay)

        checkpoint.save(file_prefix=checkpoint_prefix)

       # checkpoint.save(file_prefix=checkpoint_prefix)

