import os
from keras.preprocessing import image
import numpy as np
import pandas as pd


image_list = pd.read_csv('/rigel/edu/coms4995/users/cz2431/list.txt',sep=' ',skiprows = 6,header= None)
image_names = image_list[:][0]
image_names = np.asarray(image_names)

from keras.preprocessing import image

imagesNames = [image.load_img(os.path.join("/rigel/edu/coms4995/datasets/pets", name + '.jpg'), target_size=(224, 224)) for name in image_names]

X = np.array([image.img_to_array(img) for img in imagesNames])


from keras import applications

# build the VGG16 network
model = applications.VGG16(include_top=False,
                           weights='imagenet')


from keras.applications.vgg16 import preprocess_input
X_pre = preprocess_input(X)
features = model.predict(X_pre)


print(features.shape)

