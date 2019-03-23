# Simple_Linear_Regression

#Importing the Libararies.....
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



#Importing Data Sets pp
dataset=pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 1].values

z=dataset.iloc[:, 0].values

# Splitting the dataset into Training set and Test Set
from  sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=1/3, random_state=0) 

'''
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)
'''

# Fitting Simple Linear Regression to the training Set.....
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train, y_train)

#Predicting the Test Set Resuts
y_pred=regressor.predict(x_test)
#---> y_pred do predicts the salary from simple linear regression model.

# Visualising training set results
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising training set results
plt.scatter(x_test, y_test, color='red')
#plt.scatter(x_test, y_pred, color='green')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()













