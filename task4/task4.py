import os
from keras.preprocessing import image
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras import applications
from sklearn.linear_model import LogisticRegressionCV
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC


image_name = [i for i in os.listdir('/rigel/edu/coms4995/datasets/pets/') if 'jpg' in i]
image_name_loc = []
for j in image_name:
    tmp = '/rigel/edu/coms4995/datasets/pets/'+j
    image_name_loc.append(tmp)

im = [image.load_img(i, target_size=(224, 224)) for i in image_name_loc]

X = np.array([image.img_to_array(j) for j in im])

# build the VGG16 network
model = applications.VGG16(include_top=False,
                           weights='imagenet')

X_pre = preprocess_input(X)
features = model.predict(X_pre)


features_ = features.reshape(int(features.shape[0]), -1)

y = np.zeros(int(features.shape[0]), dtype='int')
y[int(features.shape[0])/2:] = 1
X_train, X_test, y_train, y_test = train_test_split(features_, y, stratify=y)


oneall = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X_train, y_train)
oneall_lr = OneVsOneClassifier(LogisticRegressionCV(random_state=0)).fit(X_train, y_train)

print("LinearSVC score: {:.3f}".format(oneall.score(X_test, y_test)))
print("LogisticRegressionCV score: {:.3f}".format(oneall_lr.score(X_test, y_test)))

model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(25088,)))
model.add(Dense(1024, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=128, epochs=20, verbose=1, validation_split=.1)
score = model.evaluate(X_test, y_test, verbose=0)
print(score)
