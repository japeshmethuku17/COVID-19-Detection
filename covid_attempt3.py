# -*- coding: utf-8 -*-
"""
Created on Sat May 30 23:27:23 2020
COVID-19 Detection Attempt 3 with a CNN model
@author: japes
"""
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import utils
from tensorflow.keras import datasets, layers, models, callbacks
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras import optimizers
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Add, AveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import Adam
import pydot
from matplotlib.pyplot import imshow
import scipy.misc
import os 
import glob
import gc
import numpy as np
import pandas as pd
from imutils import paths
import argparse
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import os
import seaborn as sns
import cv2


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot_3.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str, default="covid19_3.model",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())
# initialize the initial learning rate, number of epochs to train for,
# and batch size
learning_rate = 1e-3
EPOCHS = 25
batch_size = 8

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []
# loop over the image paths
for imagePath in imagePaths:
	# extract the class label from the filename
    label = imagePath.split(os.path.sep)[-2]
    # load the image, swap color channels, and resize it to be a fixed
    # 224x224 pixels while ignoring aspect ratio
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    # update the data and labels lists, respectively
    data.append(image)
    labels.append(label)
# convert the data and labels to NumPy arrays while scaling the pixel
# intensities to the range [0, 1]
data = np.array(data) / 255.0
labels = np.array(labels)

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)
# initialize the training data augmentation object
trainAug = ImageDataGenerator(
	rotation_range=15,
	fill_mode="nearest")

#Let's try CNN Model

model =models.Sequential()
#Block 1
model.add(Conv2D(32, kernel_size=(3, 3), dilation_rate=(1,1), padding = 'same', strides=(1,1), kernel_initializer='glorot_uniform', activation='relu', input_shape=(224, 224, 3)))
model.add(Conv2D(32, kernel_size=(3, 3), dilation_rate=(1,1), padding = 'same', strides=(1,1), kernel_initializer='glorot_uniform', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#Block 2
model.add(Conv2D(64, kernel_size=(3, 3), dilation_rate=(1,1), padding = 'same', strides=(1,1), kernel_initializer='glorot_uniform', activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), dilation_rate=(1,1), padding = 'same', strides=(1,1), kernel_initializer='glorot_uniform', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#Classification block
model.add(Flatten())
model.add(Dense(64, kernel_initializer='glorot_uniform',activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))

    
#Compile the model
print("[INFO] compiling the model...")
opt = Adam(lr=learning_rate, decay = learning_rate/EPOCHS)
model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = ["accuracy"])

#Training the model
history = model.fit_generator(
	trainAug.flow(trainX, trainY, batch_size=batch_size),
	steps_per_epoch=len(trainX) // batch_size,
	validation_data=(testX, testY),
	validation_steps=len(testX) // batch_size,
	epochs=EPOCHS)

#make predictions on test set

print("[INFO] evaluating the network...")
preds = model.predict(testX, batch_size = batch_size)
preds = np.argmax(preds, axis = 1)
print(classification_report(testY.argmax(axis=1),preds,target_names = lb.classes_))

# compute the confusion matrix and and use it to derive the raw
# accuracy, sensitivity, and specificity
cm = confusion_matrix(testY.argmax(axis=1), preds)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
# show the confusion matrix, accuracy, sensitivity, and specificity
print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on COVID-19 Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

#serialize the model to disk
print("[INFO] saving COVID-19 detection model to disk")
model.save(args["model"], save_format = "h5")