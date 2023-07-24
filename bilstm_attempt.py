

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential

from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from keras.layers import Dropout

#Custom Activation Function -- swish for Bidirectional Layer
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
from keras.layers import Activation

def custom_activation(x, beta = 1):
        return (K.sigmoid(beta * x) * x)

get_custom_objects().update({'custom_activation': Activation(custom_activation)})

"""Dataset used: [https://www.kaggle.com/stanley11291985/hk-macroeconomics-data](https://www.kaggle.com/stanley11291985/hk-macroeconomics-data)"""

from google.colab import files
uploaded = files.upload()

df=pd.read_excel('Housing market data.xlsx', parse_dates=['Date'])
df

df=df[['Date', 'Private Domestic (Price Index)']]
df.index=df['Date']

"""As the time series is irregularly sampled, we resample (to get daily frequency) and interpolate the missing Private Domestic values"""

#Moving date to index for ease of interpolation
del df['Date']

df=df.resample('D').mean()

df['Private Domestic (Price Index)']=df['Private Domestic (Price Index)'].interpolate().values.ravel().tolist()

df

#Restoring Date as a column
df['Date']=df.index

#Plotting graph to view how Private Domestic varies with Date
plt.figure(figsize = (10, 6))

plt.plot(df['Date'],df['Private Domestic (Price Index)'], color='tab:orange')

"""Splitting into test and train data sets + Scaling


"""

train_size = int(len(df)*0.8)
date=df.pop('Date')
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]

scaler = MinMaxScaler().fit(train_data)
train_scaled = scaler.transform(train_data)
test_scaled = scaler.transform(test_data)

def create_dataset (X, past):
    Xs, ys = [], []

    for i in range(len(X)-past):
        v = X[i:i+past]
        Xs.append(v)
        ys.append(X[i+past])

    return np.array(Xs), np.array(ys)
past = 31
X_train, y_train = create_dataset(train_scaled,past)
X_test, y_test = create_dataset(test_scaled,past)

"""Model creation"""

def create_bilstm(units):
    model = Sequential()
    model.add(Activation(custom_activation,name = "Swish"))
    model.add(Bidirectional(
              LSTM(units = units, return_sequences=True),
              input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(units = units)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer="adam",loss="mse",metrics=['accuracy'])
    return model

model_bilstm = create_bilstm(64)

def fit_model(model):
    path_checkpoint = "model_checkpoint.h5"
    early_stop = keras.callbacks.EarlyStopping(monitor = "val_loss",
                                               patience = 10)

    modelckpt_callback = keras.callbacks.ModelCheckpoint(
    monitor="val_loss",
    filepath=path_checkpoint,
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
    )
    history = model.fit(X_train, y_train, epochs = 30,
                        validation_split = 0.40,
                        batch_size = 32, shuffle = False,
                        callbacks = [modelckpt_callback])
    return history

history_bilstm = fit_model(model_bilstm)

y_test = scaler.inverse_transform(y_test)
y_train = scaler.inverse_transform(y_train)

plt.figure(figsize = (12, 8), dpi=100)
plt.plot(history_bilstm.history["loss"])
plt.plot(history_bilstm.history["val_loss"])
plt.title("Model Train vs Validation Loss for BiLSTM")
plt.ylabel("Loss")
plt.xlabel("epoch")
plt.legend(["Train loss", "Validation loss"], loc="upper right")

def prediction(model):
    prediction = model.predict(X_test)
    prediction = scaler.inverse_transform(prediction)
    return prediction


prediction_bilstm = prediction(model_bilstm)
# Plot test data vs prediction
def plot_future(prediction, model_name, y_test):
    plt.figure(figsize=(10,6))
    range_future = len(prediction)
    plt.plot(np.arange(range_future), np.array(y_test),
             label="Test data")
    plt.plot(np.arange(range_future),
             np.array(prediction),label="Prediction")
    plt.title("Test data vs prediction for " + model_name)
    plt.legend(loc="upper left")
    plt.xlabel("Time (day)")
    plt.ylabel("Private Domestic (Price Index)")

plot_future(prediction_bilstm, "Bidirectional LSTM", y_test)

#Caluclating Mean Absolute Percentage Error
mape=(sum(abs((y_test-prediction_bilstm)/y_test)))/len(y_test)*100
mape
