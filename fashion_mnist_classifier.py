# It should be a .ipynb file, but something went wrong and i convert it to .py

#!/usr/bin/env python
# coding: utf-8

# In[36]:


"""In this notebook, I will train a CNN to recognize images of clothing that belong to 10 different classes. I will apply some data augmentation techniques to reduce the risk of overfitting and will be using the Fashion MNIST dataset provided by TensorFlow."""


# In[2]:


#import dataset
from tensorflow.keras.datasets import fashion_mnist


# In[3]:


# reshape training and test sets
(features_train, label_train), (features_test, label_test) = fashion_mnist.load_data()


# In[23]:


# need to reshape from (60000, 28, 28) to (60000, 28, 28, 1) for training set and so on for test set
features_train = features_train.reshape((60000, 28, 28, 1))
features_test = features_test.reshape((10000, 28, 28, 1))


# In[25]:


img_height = 28
img_width = 28


# In[26]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[27]:


# Creating a data generator with data augmentation
train_img_gen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
val_img_gen = ImageDataGenerator(rescale=1./255)


# In[28]:


train_data_gen = train_img_gen.flow(features_train, label_train, batch_size=14)
val_data_gen = train_img_gen.flow(features_test, label_test, batch_size=14)


# In[29]:


# now i'll create NN
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


# In[30]:


model = tf.keras.Sequential()


# In[31]:


# adding layers
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Flatten())
#fully connected layers
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


# In[32]:


optimizer = tf.keras.optimizers.Adam(0.001)


# In[33]:


model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


# In[35]:


# training the model
model.fit(train_data_gen, steps_per_epoch=len(features_train) // 14, epochs=5, validation_data=val_data_gen, validation_steps=len(features_test) // 14)
