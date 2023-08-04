import pandas as pd
import time
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from iqoptionapi.stable_api import IQ_Option
from keras.models import load_model
import os
import json


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


iq = IQ_Option("seu email", "sua senha")
iq.connect()
par = "EURUSD"
time_frame = 60
bet_money = 2

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



def get_data_iq(par, iq, time_frame):
    velas = iq.get_candles(par, time_frame, 350, time.time())
    data = pd.DataFrame(velas)
    data['pips'] = data['close'] - data['open']
    data['verifypips'] = np.where(data['pips'] >= 0.00003, 0,
                                  np.where(data['pips'] <= -0.00003, 1,
                                           2))
    data['cores'] = np.where(data['open'] < data['close'], 1,
                                 np.where(data['open'] > data['close'], 0,
                                          2))
    data['rsi'] = rsi(data, window=16)
    data['stocastico'] = stochastic(data, window=16, smooth=3)
    data['maxcalc'] = data['max'] - data['min']
    data['closexcalc'] = data['open'] - data['close']
    data['mhi'] = get_cores_cruzamento(data)
    print(data['mhi'])
    X = data[["mhi","maxcalc","closexcalc","rsi","stocastico","verifypips","cores"]]
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
        self.dense1 = keras.layers.Dense(32, activation="tanh")
        self.dropout1 = keras.layers.Dropout(0.2)
        self.batchNorm1 = keras.layers.BatchNormalization()
        self.lstm = keras.layers.LSTM(32)
        self.dense2 = keras.layers.Dense(32, activation="tanh")
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
optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()
replay_buffer = []
max_replay_buffer_size = 10000
min_replay_buffer_size = 1000
batch_size = 64
gamma = 0.99
epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 0.999
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
total_reward = 0
for episode in range(100):
    state = get_data_iq(par, iq, time_frame)
    done = False
    while not done:
        # Seleciona ação com base na política epsilon-greedy
        if np.random.rand() < epsilon:
            action = np.random.randint(num_actions)
        else:
            q_values = model.predict(state)
            action = np.argmax(q_values[0])
        # Executa ação no ambiente e observa novo estado e recompensa
        trade = False
        blablabla = iq.get_balance()
        print("Action ==>",action)
        print("Rewards ==>", total_reward)
        if action == 0:  # Comprar
            if datetime.datetime.now().second > 25:
                while datetime.datetime.now().second < 58:
                    pass
            check, id = iq.buy(bet_money, par, "call", 1)
            print('BUY IN ACTION')
            trade = True
            time.sleep(38)
            checktempo()
        elif action == 1:  # Vender
            if datetime.datetime.now().second > 25:
                while datetime.datetime.now().second < 58:
                    pass
            check, id = iq.buy(bet_money, par, "put", 1)
            print('SELL IN ACTION')
            trade = True
            time.sleep(38)
            checktempo()
        else:  # Esperar
            trade = False
            time.sleep(60)
        betsies = iq.get_balance()
        vaisefude = betsies - blablabla
        print(vaisefude, "PRACAR")
        if vaisefude >= 1 and trade:
            print("WIN")
            total_reward += 1
            blablabla = betsies  # atualizar
        elif vaisefude <= 0 and trade:
            total_reward += -1
            print("LOSS")
        else:
            print('semoperacao')
        new_state = get_data_iq(par, iq, time_frame)
        done = False
        replay_buffer.append((state, action, total_reward, new_state, done))
        state = new_state
        if len(replay_buffer) > max_replay_buffer_size:
            replay_buffer.pop(0)
        if len(replay_buffer) > min_replay_buffer_size:
            minibatch = np.random.choice(replay_buffer, batch_size, replace=False)
            states = np.array([transition[0] for transition in minibatch])
            actions = np.array([transition[1] for transition in minibatch])
            rewards = np.array([transition[2] for transition in minibatch])
            next_states = np.array([transition[3] for transition in minibatch])
            with tf.GradientTape() as tape:
                target_Qs = model(next_states)
                target_Q = rewards + gamma * np.max(target_Qs, axis=1)
                target_Q = tf.stop_gradient(target_Q)
                Q_values = model(states)
                Q_action = np.sum(Q_values * tf.one_hot(actions, num_actions), axis=1)
                loss = huber_loss(target_Q, Q_action)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epsilon = max(min_epsilon, epsilon * epsilon_decay)
        checkpoint.save(file_prefix = checkpoint_prefix)

# Carregando checkpoint
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
