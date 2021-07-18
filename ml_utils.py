from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from datetime import datetime, date, time, timezone
import gdown
from pandas import read_csv
from credit_data_actual_values import substitute

import copy
import matplotlib.pyplot as plt
from logging import Logger

# define a Gaussain NB classifier
clf = GaussianNB()
clflr = LogisticRegression(penalty='l2',C=1.0, max_iter=10000)

# define the class encodings and reverse encodings
classes = {0: "Good Risk", 1: "Bad Risk"}
r_classes = {y: x for x, y in classes.items()}
y_test ={}

# function to train and load the model during startup
def load_model():
    # load the dataset from the official sklearn datasets
     #url = 'https://drive.google.com/uc?id=' + '1WC-3iPQJrud1WTer883rYdHMHGqgIVJ9' #(URI ID)
     url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'
     output = 'C:\Datascience\data\german.data' # Destination directory
     gdown.download(url, output, quiet=False) 

     df=read_csv(output, sep=" ", header=None)

     #df_vis = copy.deepcopy(df)
     # Call the method substitute from credit_data_actual_values.py to display the real world values
     #df_vis = substitute(df_vis)

     # split the data frame into inputs and outputs
     last_ix = len(df.columns) - 1
     X, y = df.drop(last_ix, axis=1), df[last_ix]

     # Categorical features has to be converted into integer values for the model to process. 
     #This is done through one hot encoding.
     # select categorical features
     cat_ix = X.select_dtypes(include=['object', 'bool']).columns
     # one hot encode categorical features only
     ct = ColumnTransformer([('o',OneHotEncoder(),cat_ix)], remainder='passthrough')
     X = ct.fit_transform(X)
     # label encode the target variable to have the classes 0 and 1
     y = LabelEncoder().fit_transform(y)

     # do the test-train split and train the model
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
     clf.fit(X_train, y_train)
     clflr.fit(X_train, y_train)

     # calculate the print the accuracy score
     acc = accuracy_score(y_test, clf.predict(X_test))
     acclr = accuracy_score(y_test, clflr.predict(X_test))
     print(f"{datetime.now()} Model trained with accuracy: {round(acc, 3)}")
     print(f"{datetime.now()} Logistic Regression Model trained with accuracy: {round(acclr,3)}")
   

# function to predict the flower using the model
def predict(query_data):
    x = list(query_data.dict().values())
    prediction = clf.predict([x])
    acc = accuracy_score(y_test, prediction)
    print(f"Model trained with accuracy: {round(acc, 3)}")
    print(f"Model prediction: {classes[prediction]}")
    return classes[prediction]

# function to predict the flower using the model
def predictlr(query_data):
    x = list(query_data.dict().values())
    prediction_lr = clflr.predict([x])
    print(f"Model prediction: {classes[prediction_lr]}")
    return classes[prediction_lr]



# function to retrain the model as part of the feedback loop
def retrain(data):
    # pull out the relevant X and y from the FeedbackIn object
    X = [list(d.dict().values())[:-1] for d in data]
    y = [r_classes[d.flower_class] for d in data]

    # fit the classifier again based on the new data obtained
    clf.fit(X, y)
