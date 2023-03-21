#!/usr/bin/env python
# coding: utf-8

# In[23]:


import tensorflow_datasets as tfds
[test_set_raw, valid_set_raw, train_set_raw], info = tfds.load("tf_flowers",
split=["train[:10%]", "train[10%:25%]", "train[25%:]"], as_supervised=True,
with_info=True)


# In[24]:


info


# In[25]:


class_names = info.features["label"].names
n_classes = info.features["label"].num_classes
dataset_size = info.splits["train"].num_examples


# In[26]:


import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
index = 0
sample_images = train_set_raw.take(9) 
for image, label in sample_images:
    index += 1
    plt.subplot(3, 3, index)
    plt.imshow(image)
    plt.title("Class: {}".format(class_names[label]))
    plt.axis("off")
    
plt.show(block=False)


# In[27]:


def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    return resized_image, label


# In[28]:


import tensorflow as tf
batch_size = 32
train_set = train_set_raw.map(preprocess).shuffle(dataset_size).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map(preprocess).batch(batch_size).prefetch(1) 
test_set = test_set_raw.map(preprocess).batch(batch_size).prefetch(1)


# In[29]:


train_set


# In[30]:


plt.figure(figsize=(8, 8))
sample_batch = train_set.take(1)
for X_batch, y_batch in sample_batch:
    for index in range(12):
        plt.subplot(3, 4, index + 1)
        plt.imshow(X_batch[index]/255.0)
        plt.title("Class: {}".format(class_names[y_batch[index]]))
        plt.axis("off")
plt.show()


# In[31]:


from functools import partial
from tensorflow import keras
DefaultConv2D = partial(keras.layers.Conv2D, 
                        kernel_size=3,
                        activation='relu', 
                        padding="SAME")


# In[32]:


model = keras.models.Sequential([
    keras.layers.Rescaling(1./127.5, -1),
    DefaultConv2D(filters=64, kernel_size=3, input_shape=[224, 224, 1]),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=128),
    #DefaultConv2D(filters=128),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=256),
    DefaultConv2D(filters=256),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Flatten(),
    #keras.layers.Dense(units=128, activation='relu'),
    #keras.layers.Dropout(0.5),
    keras.layers.Dense(units=64, activation='relu'),
    #keras.layers.Dropout(0.5),
    keras.layers.Dense(units= 5, activation='softmax')])


# In[33]:


optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, decay = 0.01)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
history = model.fit(train_set, validation_data = valid_set, epochs=10)


# In[34]:


from sklearn.metrics import accuracy_score
acc_train = model.evaluate(train_set)
acc_valid = model.evaluate(valid_set)
acc_test = model.evaluate(test_set)
print(acc_train, acc_valid, acc_test)


# In[36]:


import pickle
file = open("simple_cnn_acc.pkl", 'wb')
pickle.dump((acc_train, acc_valid, acc_test), file)

file = open("simple_cnn_acc.pkl",'rb')
a = pickle.load(file)
file.close()
print(a)


# In[37]:


def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    final_image = tf.keras.applications.xception.preprocess_input(resized_image)
    return final_image, label


# In[38]:


batch_size = 32
train_set = train_set_raw.map(preprocess).shuffle(dataset_size).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map(preprocess).batch(batch_size).prefetch(1) 
test_set = test_set_raw.map(preprocess).batch(batch_size).prefetch(1)


# In[39]:


plt.figure(figsize=(8, 8))
sample_batch = train_set.take(1)
for X_batch, y_batch in sample_batch:
    for index in range(12):
        plt.subplot(3, 4, index + 1)
        plt.imshow(X_batch[index] / 2 + 0.5)
        plt.title("Class: {}".format(class_names[y_batch[index]]))
        plt.axis("off")
plt.show()


# In[40]:


base_model = tf.keras.applications.xception.Xception(weights="imagenet", include_top=False)


# In[41]:


avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation="softmax")(avg)
model = keras.models.Model(inputs=base_model.input, outputs=output)


# In[42]:


# for index, layer in enumerate(base_model.layers):
#     print(index, layer.name)


# In[43]:


for layer in base_model.layers:
    layer.trainable = False


# In[45]:


optimizer = keras.optimizers.SGD(learning_rate=0.2, momentum=0.9, decay=0.01)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
history = model.fit(train_set, validation_data=valid_set, epochs=5)


# In[46]:


for layer in base_model.layers:
    layer.trainable = True


# In[47]:


optimizer = keras.optimizers.SGD(learning_rate=0.2, momentum=0.9,decay=0.01)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
history = model.fit(train_set, validation_data=valid_set, epochs=10)


# In[ ]:


acc_train = model.evaluate(train_set)
acc_valid = model.evaluate(valid_set)
acc_test = model.evaluate(test_set)
print(acc_train, acc_valid, acc_test)


# In[ ]:


file = open("xception_acc.pkl", 'wb')
pickle.dump((acc_train, acc_valid, acc_test), file)

file = open("xception_acc.pkl",'rb')
a = pickle.load(file)
file.close()
print(a)


# In[ ]:





# In[ ]:




