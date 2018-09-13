#Simple Linear Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataSet = pd.read_csv('Salary_Data.csv')
#The colon means we take all the lines
X = dataSet.iloc[:,:-1].values
y = dataSet.iloc[:, 1].values

#Splitting to test and training set
from sklearn.cross_validation import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size = 1/3, random_state=0)

#fitting SLR to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xTrain, yTrain)

#Prediciton
yPred = regressor.predict(xTest)

#Graphing the data results
plt.scatter(xTrain, yTrain, color = 'red')
plt.plot(xTrain, yTrain, regressor.predict(xTrain), color='blue')
plt.title('Salary vs Experience(Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Switching the results
plt.scatter(xTest, yTest, color = 'red')
plt.plot(xTrain, yTrain, regressor.predict(xTrain), color='blue')
plt.title('Salary vs Experience(Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()