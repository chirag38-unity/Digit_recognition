import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import cv2
from keras.models import load_model


def ready_train():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    num_of_train_imgs = x_train.shape[0]  # 60000 here
    num_of_test_imgs = x_test.shape[0]  # 10000 here
    img_width = 28
    img_height = 28

    x_train = x_train.reshape(x_train.shape[0], img_height, img_width, 1)
    x_test = x_test.reshape(x_test.shape[0], img_height, img_width, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test


def ready_model():
    model = Sequential()
    input_shape = (28, 28, 1)
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))  # first convolution layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))  # second convolution layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))  # third convolution layer
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))  # first dense layer
    model.add(Dense(32, activation='relu'))  # second dense layer
    model.add(Dense(10, activation='softmax'))  # last dense layer

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy'])

    model.save('trained_model.h5')

    return model


def model_train(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train,
              batch_size=128,
              epochs=50,
              verbose=1,
              validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save('trained_model.h5')


def ready_cam():
    width = 640
    height = 480
    camera_no = 0
    cap = cv2.VideoCapture(camera_no)
    cap.set(3, width)
    cap.set(4, height)


def img_process(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    norm_img = tf.keras.utils.normalize(resized, axis=1)
    norm_img = np.array(norm_img).reshape(-1, 28, 28, 1)

    return norm_img


def predict(img):
    model = load_model('trained_model.h5')
    predictions = model.predict(img)
    predictions = np.argmax(predictions)

    return predictions
