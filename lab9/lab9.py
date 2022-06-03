#!/usr/bin/env python
# coding: utf-8

# In[96]:


import tensorflow as tf
from tensorflow import keras


# In[97]:


from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)


# In[98]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( iris.data[['petal length (cm)', 'petal width (cm)']], iris.target, test_size=0.2)


# In[99]:


pd.concat([iris.data, iris.target], axis=1).plot.scatter(
x = 'petal length (cm)',
y = 'petal width (cm)',
c = 'target',
colormap='viridis' )


# In[100]:


iris.data[['petal length (cm)', 'petal width (cm)']]


# In[101]:


from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
per_clf = Perceptron()
lista = []
lista2 = []
y_train_0 = (y_train == 0).astype(int)
per_clf.fit(X_train, y_train_0)
y_pred = per_clf.predict(X_train)
ac_1 = accuracy_score(y_train_0, y_pred)
w_0 = per_clf.intercept_[0]
w_1 = per_clf.coef_[0][0]

y_test_0 = (y_test == 0).astype(int)
y_pred = per_clf.predict(X_test)
ac_2 = accuracy_score(y_test_0, y_pred)
w_2= per_clf.coef_[0][0]
lista2.append((w_0,w_1, w_2))
lista.append((ac_1,ac_2))


# In[102]:


per_clf = Perceptron()
y_train_1 = (y_train == 1).astype(int)
per_clf.fit(X_train, y_train_1)
y_pred = per_clf.predict(X_train)
ac_1 = accuracy_score(y_train_1, y_pred)
w_0 = per_clf.intercept_[0]
w_1 = per_clf.coef_[0][0]

y_test_1 = (y_test == 1).astype(int)
per_clf.fit(X_test, y_test_1)
y_pred = per_clf.predict(X_test)
ac_2 = accuracy_score(y_test_1, y_pred)
w_2= per_clf.coef_[0][0]
lista2.append((w_0, w_1, w_2))
lista.append((ac_1, ac_2))


# In[103]:


per_clf = Perceptron()
y_train_2 = (y_train == 2).astype(int)
per_clf.fit(X_train, y_train_2)
y_pred = per_clf.predict(X_train)
ac_1 = accuracy_score(y_train_2, y_pred)
w_0 = per_clf.intercept_[0]
w_1 = per_clf.coef_[0][0]

y_test_2 = (y_test == 2).astype(int)
per_clf.fit(X_test, y_test_2)
y_pred = per_clf.predict(X_test)
ac_2 = accuracy_score(y_test_2, y_pred)
w_2= per_clf.coef_[0][0]
lista2.append((w_0,w_1, w_2))
lista.append((ac_1, ac_2))


# In[104]:


import pickle


# In[105]:


file = open("per_acc.pkl", 'wb')
pickle.dump(lista, file)

file = open("per_acc.pkl",'rb')
a = pickle.load(file)
file.close()
print(a)


# In[106]:


file = open("per_wght.pkl", 'wb')
pickle.dump(lista2, file)

file = open("per_wght.pkl",'rb')
a = pickle.load(file)
file.close()
print(a)


# In[107]:


#2


# In[108]:


X = np.array([[0, 0],
[0, 1],
[1, 0],
[1, 1]])
y = np.array([0,1,1,0])

xor_per_clf = Perceptron()
xor_per_clf.fit(X, y)


# In[109]:


while True:
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[2, 1]))
    model.add(keras.layers.Dense(2, activation="tanh"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    opt = tf.keras.optimizers.Adam(learning_rate=0.15)
    model.compile(loss="binary_crossentropy", optimizer=opt)
    history = model.fit(X, y, epochs=100, verbose=False)
    #print(history.history["loss"])
    y_pred = model.predict(X)
    if y_pred[0] < 0.1 and y_pred[1] > 0.9 and y_pred[2] > 0.9 and y_pred[3] < 0.1:
        print(y_pred)
        break


# In[110]:


file = open("mlp_xor_weights.pkl", 'wb')
pickle.dump(model.get_weights(), file)

file = open("mlp_xor_weights.pkl",'rb')
a = pickle.load(file)
file.close()
print(a)


# In[ ]:





# In[ ]:




