'''blah
'''
__author__ = 'paul'

from __future__ import print_function
import pandas as pd
import numpy as np
from random import random
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import matplotlib.pyplot as plt

print(keras.__version__) # 0.3.1

# Simulate data
flow = (list(range(1,10,1)) + list(range(10,1,-1)))*1000
pdata = pd.DataFrame({"a":flow, "b":flow, "c":flow})
pdata.b = pdata.b.shift(5)
pdata.c = pdata.c.shift(9)
data = pdata.iloc[10:] * random()  # some noise

dates = pd.date_range(start='2009-01-01', end='2015-12-31', freq='D')
n = len(dates)
a = np.sin(np.arange(n) * 2 * np.pi / 7) * 100 + np.random.normal(1000, 10, n)
# b = np.sin(np.arange(n) * 2 * np.pi / 7) * 500 + np.random.normal(1000, 100, n)
# c = np.sin(np.arange(n) * 2 * np.pi / 7) * 1000 + np.random.normal(2000, 100, n)
# pdata = pd.DataFrame({"a":a, "b":b, "c":c})
pdata = pd.DataFrame({"a":a}, index=dates)
data = pdata
data[data < 0] = 0
data.iloc[:100, ].reset_index('date').plot()


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
    ntrn = int(round(len(df) * (1 - test_size)))

    X_train, y_train = _load_data(df.iloc[0:ntrn])
    X_test, y_test = _load_data(df.iloc[ntrn:])

    return (X_train, y_train), (X_test, y_test)

# in_neurons = 1
# hidden_neurons = 10
# out_neurons = 1
#
# model = Sequential()
# model.add(LSTM(input_dim=in_neurons, output_dim=hidden_neurons, return_sequences=False))
# model.add(Dense(output_dim=out_neurons))
# model.add(Activation("linear"))
# model.compile(loss="mean_squared_error", optimizer="rmsprop")
#
# model = Sequential()
# model.add(LSTM(input_dim=in_neurons, output_dim=300, return_sequences=True))
# model.add(LSTM(input_dim=300, output_dim=500, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(input_dim=500, output_dim=200, return_sequences=False))
# model.add(Dropout(0.2))
# model.add(Dense(input_dim=200, output_dim=out_neurons))
# model.add(Activation("linear"))
# model.compile(loss="mean_squared_error", optimizer="rmsprop")

# define model structure
in_out_neurons = 1
hidden_neurons = 300
model = Sequential()
model.add(LSTM(input_dim=in_out_neurons, output_dim=hidden_neurons, return_sequences=False))
model.add(Dense(output_dim=in_out_neurons))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="rmsprop")

# retrieve data
(X_train, y_train), (X_test, y_test) = train_test_split(data)

# fit model
model.fit(X_train, y_train, batch_size=450, nb_epoch=10, validation_split=0.05)

predicted = model.predict(X_test)
rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))
print(rmse)

# and maybe plot it
n_plot = 100

plt.figure()
plt.plot(predicted[:n_plot])
plt.plot(y_test[:n_plot])
plt.show()
