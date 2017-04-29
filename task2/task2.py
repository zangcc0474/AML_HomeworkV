from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
import keras
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.grid_search import GridSearchCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
model.compile("adam", "categorical_crossentropy", metrics=['accuracy'])


from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()


X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

num_classes = 10
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


vanila = model.fit(X_train, y_train, batch_size=128, epochs=20, verbose=1, validation_split=.1)


def make_model_vanila(optimizer="adam", hidden_size=32):
    model = Sequential([
        Dense(32, input_shape=(784,)),
        Activation('relu'),
        Dense(10),
        Activation('softmax'),
    ])
    model.compile("adam", "categorical_crossentropy", metrics=['accuracy'])
    return model


clf_vanila = KerasClassifier(make_model_vanila)
param_grid = {'epochs': [1, 5, 10],  # epochs is fit parameter, not in make_model!
              'hidden_size': [32, 64, 256]}
grid_vanila = GridSearchCV(clf_vanila, param_grid=param_grid, cv=5)
grid_vanila.fit(X_train, y_train)


res_vanila = pd.DataFrame(grid_vanila.cv_results_)
res_vanila.pivot_table(index=["param_epochs", "param_hidden_size"],
                values=['mean_train_score', "mean_test_score"])


from keras.layers import Dropout
model_dropout = Sequential([
    Dense(1024, input_shape=(784,), activation='relu'),
    Dropout(.5),
    Dense(1024, activation='relu'),
    Dropout(.5),
    Dense(10, activation='softmax'),
])
model_dropout.compile("adam", "categorical_crossentropy", metrics=['accuracy'])
history_dropout = model_dropout.fit(X_train, y_train, batch_size=128,
                            epochs=20, verbose=1, validation_split=.1)


def make_model_batch(optimizer="adam", hidden_size=32):
    model = Sequential([
        Dense(32, input_shape=(784,)),
        Activation('relu'),
        Dense(10),
        Activation('softmax'),
    ])
    model.compile("adam", "categorical_crossentropy", metrics=['accuracy'])
    return model


clf_batch = KerasClassifier(make_model_vanila)
param_grid = {'epochs': [1, 5, 10],  # epochs is fit parameter, not in make_model!
              'hidden_size': [32, 64, 256]}
grid_batch = GridSearchCV(clf_batch, param_grid=param_grid, cv=5)
grid_batch.fit(X_train, y_train)


res_batch = pd.DataFrame(grid_batch.cv_results_)
res_batch.pivot_table(index=["param_epochs", "param_hidden_size"],
                values=['mean_train_score', "mean_test_score"])


df_vanila = pd.DataFrame(vanila.history)
df_vanila[['acc', 'val_acc']].plot()
plt.ylabel("accuracy")
df_vanila[['loss', 'val_loss']].plot(linestyle='--', ax=plt.twinx())
plt.ylabel("loss")


df_dropout = pd.DataFrame(history_dropout.history)
df_dropout[['acc', 'val_acc']].plot()
plt.ylabel("accuracy")
df_dropout[['loss', 'val_loss']].plot(linestyle='--', ax=plt.twinx())
plt.ylabel("loss")

