__author__ = 'paul'

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

in_out_neurons = 2
hidden_neurons = 300

model = Sequential()
model.add(LSTM(in_out_neurons, hidden_neurons, return_sequences=False))
model.add(Dense(hidden_neurons, in_out_neurons))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="rmsprop")

import pandas as pd
from random import random

flow = (list(range(1,10,1)) + list(range(10,1,-1)))*1000
pdata = pd.DataFrame({"a":flow, "b":flow})
pdata.b = pdata.b.shift(9)
data = pdata.iloc[10:] * random()  # some noise

import numpy as np

def _load_data(data, n_prev = 100):
    """
    data should be pd.DataFrame()
    """

    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(data.iloc[i+n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY

def train_test_split(df, test_size=0.1):
    """
    This just splits data to training and testing parts
    """
    ntrn = round(len(df) * (1 - test_size))

    X_train, y_train = _load_data(df.iloc[0:ntrn])
    X_test, y_test = _load_data(df.iloc[ntrn:])

    return (X_train, y_train), (X_test, y_test)


(X_train, y_train), (X_test, y_test) = train_test_split(data)  # retrieve data

# and now train the model
# batch_size should be appropriate to your memory size
# number of epochs should be higher for real world problems
model.fit(X_train, y_train, batch_size=450, nb_epoch=10, validation_split=0.05)

predicted = model.predict(X_test)
rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))

# and maybe plot it
pd.DataFrame(predicted[:100]).plot()
pd.DataFrame(y_test[:100]).plot()


from keras.models import Sequential
from keras.layers.core import Dense, Activation, Droupout
from keras.layers.recurrent import LSTM


model = Sequential()
model.add(LSTM(5, 300, return_sequences=True))
model.add(LSTM(300, 500, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(500, 200, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(200, 3))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="rmsprop")

