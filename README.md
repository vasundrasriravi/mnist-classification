# EX 3 Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

## Neural Network Model
![MODEL](https://github.com/user-attachments/assets/736569e2-739b-4ce2-8b34-b25cb97038b9)

## DESIGN STEPS
### STEP 1:
Import tensorflow and preprocessing libraries.
### STEP 2:
load the dataset
### STEP 3:
Scale the dataset between it's min and max values
### STEP 4:
Using one hot encode, encode the categorical values
### STEP 5:
Split the data into train and test
### STEP 6:
Build the convolutional neural network model
### STEP 7:
Train the model with the training data
### STEP 8:
Plot the performance plot
### STEP 9:
Evaluate the model with the testing data
### STEP 10:
Fit the model and predict the single input

## PROGRAM

### Name: VASUNDRA SRI R
### Register Number: 212222230168
```py
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from tensorflow.keras.metrics import CategoricalCrossentropy
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train.shape

X_test.shape

single_image= X_train[0]
single_image.shape

plt.imshow(single_image,cmap='gray')

y_train.shape

X_train.min()

X_train.max()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

X_train_scaled.min()

X_train_scaled.max()

y_train[0]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

type(y_train_onehot)
y_train_onehot.shape

print("VASUNDRA SRI R")
single_image = X_train[400]
plt.imshow(single_image,cmap='gray')

y_train_onehot[500]
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

Name: VASUNDRA SRI R
Register Number: 212222230168

model = keras.Sequential()
model.add(Conv2D(32,(3,3),activation="relu",input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(16,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(32,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(16,activation='relu'))
model.add(Dense(10,activation="softmax"))

model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64,
          validation_data=(X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)

metrics.head()

print('VASUNDRA SRI R')
metrics[['accuracy','val_accuracy']].plot()

print('VASUNDRA SRI R')
metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print('VASUNDRA SRI R')
print(confusion_matrix(y_test,x_test_predictions))

print('VASUNDRA SRI R')
print(classification_report(y_test,x_test_predictions))

img = image.load_img('image 8.jpg')

type(img)

img = image.load_img('image 8.jpg')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)

print('VASUNDRA SRI R')
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0


x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)

print("VASUNDRA SRI R")
print(x_single_prediction)
```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
![dimage1](https://github.com/user-attachments/assets/9882bc77-f770-44bb-89e1-ee7c2fa1f1ea)
![dimage2](https://github.com/user-attachments/assets/9583b198-da2c-49c2-8c06-887453cae1ca)

### Classification Report
![dimage3](https://github.com/user-attachments/assets/c410cce3-daa7-4efe-b616-27fddcb9bfd7)

### Confusion Matrix
![dimage4](https://github.com/user-attachments/assets/5924000c-20b0-4c9d-9a8e-e5c746ae4742)

### New Sample Data Prediction
## Input
![dimage5](https://github.com/user-attachments/assets/8564854d-06d2-4d18-a4a1-76756c02509d)

## Output

![dimage6](https://github.com/user-attachments/assets/7dc7f5a2-71a8-41e6-98ea-96a308236cbe)

## RESULT
Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully.
