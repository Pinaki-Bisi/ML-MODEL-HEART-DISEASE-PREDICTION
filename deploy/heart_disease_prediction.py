import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

"""DATA COLLECTION"""

#CONVERTING INTO COMMA SEPARATED VALUE FILE
heart= pd.read_csv("heart.csv")

#DISPLAY THE NO. OF ROWS AND COLUMNS
heart.shape

#DISPLAY THE FIRST SPECIFIED ELEMENTS
heart.head(7)

#CHECKS THE COLUMN WHICH AR3E NULL
heart.isnull().sum()

"""DATA ANALYSIS AND DATA VISUALIZATION"""

heart.describe()

heart['target'].value_counts()

x=heart.drop(columns='target',axis=1)
y=heart['target']

print(x)

print(y)

"""DATA PROCESSING"""

x=heart.drop('target',axis=1)

print(x)

"""LABEL BINARIZATION"""

Y = heart['target'].apply(lambda y_value: 1 if y_value>=1 else 0 )

print (Y)

"""SPLITTING TRAIN AND TEST DATA"""

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)

print(x.shape,x_train.shape,x_test.shape)

"""TRAINING THE DATA"""

model= LogisticRegression()

model.fit(x_train,y_train)

"""ACCURACY SCORE

LOGISTIC REGRESSION
"""

#accuracy on train data
x_train_prediciton= model.predict(x_train)
training_data_accuracy= accuracy_score(x_train_prediciton,y_train)
print('accuracy on training data is:',training_data_accuracy)

#accuracy on test data
x_test_prediciton= model.predict(x_test)
test_data_accuracy= accuracy_score(x_test_prediciton,y_test)
print('accuracy on testing data is:',test_data_accuracy)
pickle.dump(model,open("model.pkl","wb"))

