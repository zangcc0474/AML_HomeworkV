
# coding: utf-8

# In[11]:

from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import keras
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.rcParams["figure.dpi"] = 200


# In[2]:

model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
model.compile("adam", "categorical_crossentropy", metrics=['accuracy'])


# In[3]:

from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[4]:

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

num_classes = 10
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# In[5]:

vanila = model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1)


# In[ ]:




# In[ ]:




# In[6]:

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


# In[12]:

df = pd.DataFrame(vanila.history)
df[['acc']].plot()
df_dropout = pd.DataFrame(history_dropout.history)
df_dropout[['acc']].plot()
plt.ylabel("accuracy")


# In[17]:

df = pd.DataFrame(history_dropout.history)
df[['acc', 'val_acc']].plot()
plt.ylabel("accuracy")
plt.ylim(.9, 1)


# In[ ]:



