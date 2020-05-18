# RNN
# Twitter + Mobility vs Covid New case prediction

import numpy as np              
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler


#-----------------------------------------------------------------------------#

# Part 1 - Data Preprocessing

# Importing the training set
dataset_train = pd.read_csv('E:\DataScience\DSTI\Artificial Neural Networks\Project\Final\Feature_vs_Covid_ANN_forTrain_RNN.csv')
X = dataset_train.iloc[:, 5:].values  
y = dataset_train.iloc[:, [2]].values  
# training_set = covid_newcase, mob_transitstation, mob_resident, twitter_others, twitter_covid
# .values after choosing the column will change this column into numpy array



# Feature Scaling
sc = MinMaxScaler(feature_range = (0, 1))    
# Feature Scaling by Normalization
# feature_range = (0,1) because normalization formula = [x-min(x)]/[max(x)-min(x)]
# which will between 0 and 1
X_train_scaled = sc.fit_transform(X)
y_train_scaled = sc.fit_transform(y)


# Creating a data structure with 3 timesteps and 1 output
# 3 timesteps = At each time T, the RNN wil look at the features between 3 days before time T
# Based on the trends there, it will try to predict the next output
# The optimal timesteps is something we have to try 


X_train = []
y_train = []
for i in range(3, 35):  # X_scaled, y_scaled has 34 rows >> But range will exclude the last no. so we set upper bound 31
    X_train.append(X_train_scaled[i-3:i, :])
    y_train.append(y_train_scaled[i, 0])  


# X_train, y_train are list
# Input for NN must be numpy >> change them to numpy
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping for future RNN from 2D to 3D
# New dimension that we gonna add = Indicator

# More detail go to Keras documentation/ recurrent layer/ Input shapes
# Input shapes = 3D tensor (array) with shape (batch_size, timesteps, input_dim)
# 1st argument = batch_size = Total no. of observation
# = Total row we have in X_train >> Indicate total row as X_train.shape[0]
# 2nd argument = Timesteps = Total column = .shape[1]


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 4))


#-----------------------------------------------------------------------------#

# Part 2 - Building the RNN

# Initialising the RNN 
regressor = Sequential()   # Sequence of layer


# Adding the first LSTM layer and some Dropout regularisation
# Even our input array is 3d but we just have to indicate the last 2 dimensions
# Since the 1st dimension (Total observation) will be automatically taken into account
# return_seq = True because we will build stacked LSTM = LSTM with several layers
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 4)))


# Add Dropout to avoid overfitting
#regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
# Since this is not the 1st layer, so we don't have to mention the input layer anymore
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
# Mean squared error because we do the regression

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
# Note that too little loss function value = Overfit

#-----------------------------------------------------------------------------#

# Part 3 - Making the predictions and visualising the results

# Getting the real new case covid
dataset_test = pd.read_csv('E:\DataScience\DSTI\Artificial Neural Networks\Project\Final\Feature_vs_Covid_ANN_forTest_RNN.csv')
real_features = dataset_test.iloc[:,5:].values
real_covid_new_case = dataset_test.iloc[:,2].values

# Getting the predicted stock price of 2017
features_total = pd.concat((dataset_train.iloc[:,5:], dataset_test.iloc[:,5:]), axis = 0)
inputs = features_total[len(features_total) - len(dataset_test) - 3:].values

# To predict the 1st day of test set (26th April), input = 26th April - 3 days as lower bounds
# Upper bounds = We want to predict the last day of test set (4th May)
# So, the input = All info just before the last day that we predict 

# Index of 26th April after concatenate = [len(dataset_total)-len(dataset_test)]
# Lower bound = 26th April - 3 days = [index of 26th April - 3]
# Upper bound just before the last day (last row) = [Lower bound:]

# Reshape to create 3D structure of input as before (Total observation, timesteps, indicator)
inputs = np.reshape(inputs,(inputs.shape[0], inputs.shape[1]))
# sc object was already fitted to the training set, so we used only transform method here
inputs = sc.transform(inputs)
X_test = []

for i in range(3, 12):   #3 = days overlap, 12 = rows of input
    X_test.append(inputs[i-3:i, :])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 4))
predicted_covid_new_case = regressor.predict(X_test)
predicted_covid_new_case = sc.inverse_transform(predicted_covid_new_case)

# Visualising the results
plt.plot(real_covid_new_case, color = 'red', label = 'Covid19 New Case')
plt.plot(predicted_covid_new_case, color = 'blue', label = 'Prediction',  linestyle = "--")
plt.xlabel('Day',  frontsize = 20)
plt.ylabel('Cases', fronsize = 20)
plt.legend()
plt.show()














