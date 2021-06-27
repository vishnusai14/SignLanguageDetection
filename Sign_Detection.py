# All The Imports



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
import numpy as np
import cv2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image



# Important Steps
# Creating The Trained Model
#Here we are Creating A Sequential Model
#Don't change The input_shape()
#Try Changing The kernel size to check if accuracy is being changed

model = Sequential()

#Adding A List Of CNN Model To Make The Model To Learn From The Data Set
#filter = set the number of training process
#kernel_size = determines the kernel dimensions
#activation = 'relu' is the name of activation model which increases the accuracy 
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64,64,3)))
#maxpool2d pool_size(2,2) - increases the speed by adding the step of 2 
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Flatten())





#Don't Change Any Value here
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(33, activation = 'softmax'))


#Compiling The CNN
model.compile(optimizer = SGD(learning_rate = 0.01),loss = 'categorical_crossentropy',metrics = ['accuracy'])


#Creating The Train Process
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


#Creating The Train Set
#Try Changing The Batch value only
#Change The Path To Training Set
training_set = train_datagen.flow_from_directory(
        '/content/drive/MyDrive/final data/train image',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')


#Same For Test data
#Change The Path To Testing Set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        '/content/drive/MyDrive/final data/test image',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

#Try Increase The epochs 
model_fit = model.fit(training_set,epochs=25,validation_data = test_set)

# For getting next batch of testing imgs...
imgs, labels = next(test_set) 
scores = model.evaluate(imgs, labels, verbose=0)
print(f'{model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')


#Once the model is fitted we save the model using model.save()  function.


model.save('modelfit.h5')



