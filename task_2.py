# Import necessary libraries¶

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization
from keras.optimizers import Adam

# Load our data
# Here, we are using MNIST dataset

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# to get the shape of the data
print("x_train shape:", x_train.shape)
print("x_test shape", x_test.shape)

# Understanding and visualizing our data
# understanding and visualizing our data
# The MNIST database (Modified National Institute of Standards and Technology database) is a large database
# of handwritten digits that is commonly used for training various image processing systems. The database is
# also widely used for training and testing in the field of machine learning. The MNIST database contains
# 60,000 training images and 10,000 testing images, with their correspoinding labels. These images are
# grayscale image of 28 by 28 pixels.

plt.figure(figsize=(8, 8))
for i in range(25):
  plt.subplot(5, 5, i+1)
  plt.title(str(y_train[i]), fontsize=16, color='black', pad=2)
  plt.imshow(x_train[i], cmap=plt.cm.binary)
  plt.xticks([])
  plt.yticks([])

plt.show()

# Spliting test data for validation and testing
# Here, we will divide test data into validation data and test data so that we can use validation data to
# avoid overfitting and can use testing data to test the performance of our CNN model. Here we have used
# 90% of our test data for validation and remaining 10% for testing.

val_images = x_test[:9000]
test_images = x_test[9000:]

# Normalizing and reshaping
# Here, pixel value for our training, validating and testing images are in range between 0 to 255. In order
# to reduce data inconsistency we have to normalize the data And, we are reshaping our images so that all
# images are of same shape and all can be feed into the network
# Here, train images have maximum value of 255. and minimum value of 0. So, for normalizing we simply can
# divide data by 255.

val_images = val_images.astype('float32') / 255.0
val_images = np.reshape(val_images, (val_images.shape[0], 28, 28, 1))

test_images = test_images.astype('float32') / 255.0
test_images = np.reshape(test_images, (test_images.shape[0], 28, 28, 1))

train_images = x_train.astype("float32") / 255.0
train_images = np.reshape(train_images, (train_images.shape[0], 28, 28, 1))

# Adding noise
# Here we are adding random numbers the our images so that our image look noisy, and we can feed them as
# input to our network along with non noisy image as target so that our network learns to denoise image.

factor = 0.39
train_noisy_images = train_images + factor * np.random.normal(loc=0.0, scale=1.0, size=train_images.shape)
val_noisy_images = val_images + factor * np.random.normal(loc=0.0, scale=1.0, size=val_images.shape)
test_noisy_images = test_images + factor * np.random.normal(loc=0.0, scale=1.0, size=test_images.shape)

# here maximum pixel value for our images may exceed 1 so we have to clip the images
train_noisy_images = np.clip(train_noisy_images, 0., 1.)
val_noisy_images = np.clip(val_noisy_images, 0., 1.)
test_noisy_images = np.clip(test_noisy_images, 0., 1.)

# Visualizing images after adding noise

plt.figure(figsize=(8, 8))

for i in range(25):
      plt.subplot(5, 5, i+1)
      plt.title(str(y_train[i]), fontsize=16, color='black', pad=2)
      plt.imshow(train_noisy_images[i].reshape(1, 28, 28)[0], cmap=plt.cm.binary)
      plt.xticks([])
      plt.yticks([])

plt.show()

# Defining our autoencoder model

# model = Sequential()
#
# # encoder network
# model.add(Conv2D(filters=128, kernel_size=(2, 2), activation='relu', padding='same', input_shape=(28, 28, 1)))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(Conv2D(filters=128, kernel_size=(2, 2), activation='relu', padding='same'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(Conv2D(filters=256, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(Conv2D(filters=256, kernel_size=(2, 2), activation='relu', padding='same'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(Conv2D(filters=512, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same'))
#
# # decoder network
# model.add(Conv2D(filters=512, kernel_size=(2, 2), activation='relu', padding='same'))
# model.add(tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=(2, 2), strides=(2, 2),
#                                           activation='relu', padding='same'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(Conv2D(filters=256, kernel_size=(2, 2), activation='relu', padding='same'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(Conv2D(filters=256, kernel_size=(2, 2), activation='relu', padding='same'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(Conv2D(filters=128, kernel_size=(2, 2), activation='relu', padding='same'))
#
#
# model.add(tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=(2, 2),
#                                           activation='relu', padding='same'))
# model.add(Conv2D(filters=64, kernel_size=(2, 2), activation='relu', padding='same'))
# model.add(tf.keras.layers.BatchNormalization())
#
# model.add(Conv2D(filters=1, kernel_size=(2, 2), activation='relu', padding='same'))
#
# # to get the summary of the model
# model.summary()

model = Sequential()

# encoder network
model.add(Conv2D(filters=16, kernel_size=(2, 2), activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(filters=16, kernel_size=(2, 2), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=(2, 2), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same'))

# decoder network
model.add(Conv2D(filters=64, kernel_size=(2, 2), activation='relu', padding='same'))
model.add(Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2),
                          activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(2, 2), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=(2, 2), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=16, kernel_size=(2, 2), activation='relu', padding='same'))


model.add(Conv2DTranspose(filters=16, kernel_size=(2, 2), strides=(2, 2),
                          activation='relu', padding='same'))
model.add(Conv2D(filters=16, kernel_size=(2, 2), activation='relu', padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(filters=1, kernel_size=(2, 2), activation='relu', padding='same'))

# to get the summary of the model
model.summary()


# Compile model

optimizer = Adam(learning_rate=0.001)
loss = 'mean_squared_error'
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Fitting the model

epochs = 5
batch_size = 256
validation = (val_noisy_images, val_images)
history = model.fit(train_noisy_images, train_images, batch_size=batch_size, epochs=epochs,
                    validation_data=validation)

# Model evaluation
# loss and accuracy curve

plt.subplot(2, 1, 1)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(loc='best')
plt.subplot(2, 1, 2)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend(loc='best')
plt.show()

# Save model
model.save('my_model.h5')
loaded_model = load_model('my_model.h5')
# Visualizing our predicted images along with real and noised images

plt.figure(figsize=(18, 18))
for i in range(10, 19):
    plt.subplot(9, 9, i)
    if i == 14:
        plt.title('Clean Images', fontsize=25, color='Green')
    plt.imshow(test_images[i].reshape(1, 28, 28)[0], cmap=plt.cm.binary)
plt.gcf()
plt.savefig('clean_images.png')
plt.show()

plt.figure(figsize=(18, 18))
for i in range(10, 19):
    if i == 15:
        plt.title('Noisy Images', fontsize=25, color='red')
    plt.subplot(9, 9, i)
    plt.imshow(test_noisy_images[i].reshape(1, 28, 28)[0], cmap=plt.cm.binary)
plt.gcf()
plt.savefig('noisy_images.png')
plt.show()

plt.figure(figsize=(18, 18))
for i in range(10, 19):
    if i == 15:
        plt.title('Denoised Images', fontsize=25, color='Blue')

    plt.subplot(9, 9, i)
    plt.imshow(loaded_model.predict(test_noisy_images[i].reshape(1, 28, 28, 1)).reshape(1, 28, 28)[0],
               cmap=plt.cm.binary)
plt.gcf()
plt.savefig('denoised_images.png')
plt.show()
