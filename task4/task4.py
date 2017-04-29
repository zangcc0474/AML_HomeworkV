import os
from keras.preprocessing import image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras import applications
from sklearn.linear_model import LogisticRegressionCV

image_list = pd.read_csv('/rigel/edu/coms4995/users/cz2431/AML_HomeworkV/task4/list.txt',sep=' ',skiprows = 6,header= None)
image_names = image_list[:][0]
image_names = np.asarray(image_names)

imagesNames = [image.load_img(os.path.join("/rigel/edu/coms4995/datasets/pets", name + '.jpg'), target_size=(224, 224)) for name in image_names]
X = np.array([image.img_to_array(img) for img in imagesNames])

# build the VGG16 network
model = applications.VGG16(include_top=False,
                           weights='imagenet')

X_pre = preprocess_input(X)
features = model.predict(X_pre)


features_ = features.reshape(int(features.shape[0]), -1)

y = np.zeros(int(features.shape[0]), dtype='int')
y[int(features.shape[0])/2:] = 1
X_train, X_test, y_train, y_test = train_test_split(features_, y, stratify=y)


lr = LogisticRegressionCV().fit(X_train, y_train)

print("Train score: " + str(lr.score(X_train, y_train)))
print("Test score: "  + str(lr.score(X_test, y_test)))