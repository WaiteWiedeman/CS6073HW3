# imports
import tensorflow as tf
import numpy as np


# defining function to test a model
# function takes keras model object and test data as input and returns the prediction
def test_model(model,X_test,y_test):
    model.evaluate(X_test, y_test)  # evaluate model performance with test data
    # loop adds dimension to each image
    for i in range(len(X_test)):
        X_test[i] = X_test[i][np.newaxis, ...]
    y_pred = model.predict(X_test)  # model makes prediction

    return y_pred
