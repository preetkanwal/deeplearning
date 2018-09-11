
# Import dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn import datasets



# Define input data
#Loading the iris dataset
iris = datasets.load_iris()
data, labels = iris['data'], iris['target']
feature_names = iris['feature_names']
dataset = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= feature_names + ['target'])



X_train, X_test, y_train, y_test = train_test_split(dataset[feature_names], dataset[['target']], test_size=0.25, random_state=5)



y_train_one_hot = pd.get_dummies(y_train, columns=['target'])
y_test_one_hot = pd.get_dummies(y_test, columns=['target'])


