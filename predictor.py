# importing all dependencies

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# creating a pandas dataframe and loading the dataset into it
diabetes_dataset = pd.read_csv('diabetes predictor\diabetes.csv')

# splitting the dataset with all labels except outcome into x and outcome into y
x = diabetes_dataset.drop(columns = 'Outcome', axis = 1)
y = diabetes_dataset['Outcome']

# standardising the data 
scalar = StandardScaler()
standardised_data = scalar.fit_transform(x)
x = standardised_data

# divide the data into 80-20 ratio of train-test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify = y, random_state = 2)

# upload the data into support vector machine 
classifier = svm.SVC(kernel = 'linear')
classifier.fit(x_train, y_train)

# check train accuracy
x_train_prediction = classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)

# check test accuracy
x_test_prediction = classifier.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)

#input data to be checked
input_data = (4,110,92,0,0,37.6,0.191,30)

# change input data into numpy array and reshape it for one instance
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardise numpy array
standardised_input_data = scalar.transform(input_data_reshaped)

# predict the result ( 0 -> healthy ; 1-> diabetic )
prediction = classifier.predict(standardised_input_data)
print(prediction) 