
# coding: utf-8

# In[43]:

import os
from keras.preprocessing import image
import numpy as np
import pandas as pd


# In[51]:

image_list = pd.read_csv('/Users/chenchaozang/Downloads/annotations/list.txt',sep=' ',skiprows = 6,header= None)
image_names = image_list[:][0]
image_names = np.asarray(image_names)

from keras.preprocessing import image

imagesNames = [image.load_img(os.path.join("/Users/chenchaozang/Downloads/images", name + '.jpg'), target_size=(224, 224)) for name in image_names]

X = np.array([image.img_to_array(img) for img in imagesNames])


# In[52]:

from keras import applications

# build the VGG16 network
model = applications.VGG16(include_top=False,
                           weights='imagenet')


# In[54]:

from keras.applications.vgg16 import preprocess_input
X_pre = preprocess_input(X)
features = model.predict(X_pre)


# In[ ]:

print(features.shape)

