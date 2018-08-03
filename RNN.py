import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset = pd.read_csv('Hindalco_stock_price.csv')
training_set = dataset.iloc[:240, 1:2].values
real_stock_price = dataset.iloc[240:, 1:2].values
dataset_total=dataset.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 240):
    X_train.append(training_set_scaled[i-60:i,])
    y_train.append(training_set_scaled[i,])
X_train, y_train = np.array(X_train), np.array(y_train)

#Building the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

# Adding the four LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'RMSprop', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

#Making the predictions and visualising the results
# Getting the predicted stock price
inputs = dataset_total[len(dataset_total) - len(real_stock_price) - 60:]
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 70):
    X_test.append(inputs[i-60:i,])
X_test = np.array(X_test)
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Hindalco Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Hindalco Stock Price')
plt.title('Hindalco Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Hindalco Stock Price')
plt.legend()
plt.show()
