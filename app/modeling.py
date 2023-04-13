"""
This Python file is for build, train and save
a Logistic Regression and a PCA
"""
import os
import numpy as np
import tensorflow as tf
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from lib.utils import PCA
import warnings
warnings.filterwarnings('ignore')


# create train and validation sets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# standardize between 0 and 1
x_train = (x_train - x_train.min()) / (x_train.max() - x_train.min())
x_test = (x_test - x_test.min()) / (x_test.max() - x_test.min())

# reshape datasets to can be passed to a Logistic Regression
X_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
X_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

# implement PCA

pca = PCA(n_components=250)
X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)

# train model
lr = LogisticRegression()
lr.fit(X_train2, y_train)

# save model and PCA to use it in production
path = 'app/models'
joblib.dump(lr, os.path.join(path, 'model.joblib'))
joblib.dump(pca, os.path.join(path, 'pca.joblib'))
print(f"The model has been trained with a validation accuracy of {accuracy_score(y_test, lr.predict(X_test2)):.1%} and stored with PCA at {path} folder")