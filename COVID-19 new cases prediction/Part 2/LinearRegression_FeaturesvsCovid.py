# Linear Regression: Feature vs Covid

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression




# Importing the data set
dataset_raw = pd.read_csv('E:\DataScience\DSTI\Artificial Neural Networks\Project\Final\Feature_vs_Covid_ANN.csv')
dataset = dataset_raw.iloc[:, [2,5,6,7,8]] 
X = dataset.iloc[:,1:].values
y = dataset.iloc[:,0].values
# .values after choosing the column will change this column into numpy array

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10, shuffle = True)

# Linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred = linear_model.predict(X_test)
date_forTest = dataset_raw.iloc[35:,0]

# Plot
plt.plot(date_forTest, y_pred,linestyle = "--", label = "Prediction")
plt.plot(y_test, label = "Covid19 New Cases")
plt.title("Covid19 New Cases", weight='heavy')
plt.ylabel("Cases")
plt.xlabel("Date")
plt.xticks(rotation = 25)
plt.legend()
plt.show()

accuracy = linear_model.score(X_test,y_test)
print(accuracy*100,'%')

