'''blah
'''
__author__ = 'paul'

from __future__ import print_function
import pandas as pd
import numpy as np
from random import seed, random
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import matplotlib.pyplot as plt

seed(1)
print(keras.__version__) # 0.3.1

# Simulate data
dates = pd.date_range(start='2009-01-01', end='2015-12-31', freq='D')
n = len(dates)
a = np.sin(np.arange(n) * 2 * np.pi / 7)
# b = np.sin(np.arange(n) * 2 * np.pi / 7)
# c = np.sin(np.arange(n) * 2 * np.pi / 7)
# pdata = pd.DataFrame({"a":a, "b":b, "c":c})
pdata = pd.DataFrame({"a":a}, index=dates)
data = pdata
n_plot = 100
data.iloc[:n_plot, ].plot()

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
model.fit(X_train, y_train, batch_size=100, nb_epoch=10, validation_split=0.05)

train_prediction = model.predict(X_train)
test_prediction = model.predict(X_test)
train_rmse = np.sqrt(((train_prediction - y_train) ** 2).mean(axis=0))
test_rmse = np.sqrt(((test_prediction - y_test) ** 2).mean(axis=0))
print(train_rmse)
print(test_rmse)

# and maybe plot it
_, axarr = plt.subplots(5, sharex=True, sharey=True)
axarr[0].plot(a[(len(a) - n_plot):len(a)])
axarr[0].set_title('Test Population')
axarr[1].plot(y_test[:n_plot])
axarr[1].set_title('Test Observation')
axarr[2].plot(test_prediction[:n_plot])
axarr[2].set_title('Prediction')
axarr[3].plot(y_test[:n_plot])
axarr[3].plot(test_prediction[:n_plot])
axarr[3].set_title('Test Observation and Prediction')
axarr[4].plot(a[(len(a) - n_plot):len(a)])
axarr[4].plot(test_prediction[:n_plot])
axarr[4].set_title('Test Population and Prediction')
