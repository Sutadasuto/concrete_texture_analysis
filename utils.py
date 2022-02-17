import matplotlib.pyplot as plt
import os
import re

from shutil import copy
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score as metric

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow_addons.metrics import F1Score

from see_classification import save_keras_classification_comparison

# Used to write csv files with proper numerical sorting of image names
def key(value):
    """Extract numbers from string and return a tuple of the numeric values"""
    return tuple(map(int, re.findall('\d+', value)))


# Define the neural network
def create_model(input_size, output_size):
    model = Sequential()
    model.add(Dense(16, input_dim=input_size, activation="relu"))
    model.add(Dense(4, activation="relu"))
    model.add(Dense(output_size, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=[F1Score(num_classes=output_size, average="macro")])
    return model
