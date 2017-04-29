import numpy as np
np.set_printoptions(precision=3, suppress=True)
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale, StandardScaler

from keras.models import Sequential
from keras.layers import Dense, Activation
import tensorflow as tf
import numpy as np

import scipy.io as sio

train = sio.loadmat("/rigel/edu/coms4995/datasets/train_32x32.mat")
test = sio.loadmat("/rigel/edu/coms4995/datasets/test_32x32.mat")

x_train = train['X']
y_train = train['y']
x_test = test['X']
y_test = test['y']


batch_size = 128
num_classes = 11
epochs = 12

# input image dimensions
img_rows, img_cols = 32, 32

# the data, shuffled and split between train and test sets
x_train=np.rollaxis(x_train, 3, 0)
x_test=np.rollaxis(x_test,3,0)
input_shape = (img_rows, img_cols, 3)
from keras.utils.np_utils import to_categorical
y_train_binary = to_categorical(y_train)
y_test_xbinary=to_categorical(y_test)


from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout
from keras.layers import BatchNormalization

num_classes = 11
cnn = Sequential()
cnn.add(Conv2D(16, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Conv2D(32, (5, 5),activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Conv2D(64, (5, 5),activation='relu'))
#cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Flatten())
cnn.add(Dropout(0.5))
cnn.add(Dense(16, activation='relu'))
cnn.add(Dense(num_classes, activation='softmax'))

cnn.compile("adam", "categorical_crossentropy", metrics=['accuracy'])
history_cnn = cnn.fit(x_train, y_train_binary,
                      batch_size = 200, epochs=50, verbose=1, validation_split=.2)


print(cnn.evaluate(x_test, y_test_xbinary))


num_classes = 11
cnn_small_bn = Sequential()
cnn_small_bn.add(Conv2D(8, kernel_size=(3, 3),
                 input_shape=input_shape))
cnn_small_bn.add(Activation("relu"))
cnn_small_bn.add(BatchNormalization())
cnn_small_bn.add(MaxPooling2D(pool_size=(2, 2)))
cnn_small_bn.add(Conv2D(8, (3, 3)))
cnn_small_bn.add(Activation("relu"))
cnn_small_bn.add(BatchNormalization())
cnn_small_bn.add(MaxPooling2D(pool_size=(2, 2)))
cnn_small_bn.add(Flatten())
cnn_small_bn.add(Dense(64, activation='relu'))
cnn_small_bn.add(Dense(num_classes, activation='softmax'))
cnn_small_bn.compile("adam", "categorical_crossentropy", metrics=['accuracy'])
history_cnn_small_bn = cnn_small_bn.fit(x_train, y_train_binary,
                                        batch_size=200, epochs=50, verbose=1, validation_split=.2)

print(cnn_small_bn.evaluate(x_test,y_test_xbinary))
