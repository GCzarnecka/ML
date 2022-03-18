#!/usr/bin/env python
# coding: utf-8

# In[116]:


from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)


# In[117]:


import numpy as np
np.version.version
print((np.array(mnist.data.loc[42]).reshape(28, 28) > 0).astype(int))


# In[118]:


X = mnist.data
y = mnist.target
y = y.sort_values()
X = X.reindex(y.index)


# In[119]:


X_train, X_test = X[:56000], X[56000:]
y_train, y_test = y[:56000], y[56000:]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[120]:


print(y_train)
print(y_test)


# In[121]:


#4


# In[122]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[123]:


from sklearn.linear_model import SGDClassifier


# In[124]:


y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8) 
y_train_0 = (y_train == 0)
y_test_0 = (y_test == 0)


# In[125]:


sgd_clf = SGDClassifier(random_state = 42)


# In[126]:


sgd_clf.fit(X_train, y_train_0)


# In[127]:


from sklearn.metrics import accuracy_score
accuracy = [accuracy_score(y_train_0, sgd_clf.predict(X_train)),
          accuracy_score(y_test_0, sgd_clf.predict(X_test))]
print(accuracy)


# In[128]:


import pickle
pickle.dump(accuracy, open( "sgd_acc.pkl", "wb" ))


# In[129]:


from sklearn.model_selection import cross_val_score


# In[130]:


score = cross_val_score(sgd_clf, X_train, y_train_0, cv=3, scoring="accuracy", n_jobs = -1)
pickle.dump(score, open( "sgd_cva.pkl", "wb" ) )


# In[132]:


#5


# In[133]:


from sklearn.metrics import confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

sgd_clf = SGDClassifier(random_state = 42, n_jobs = -1)
sgd_clf.fit(X_train, y_train)

c_mx = confusion_matrix(mnist.target, sgd_clf.predict(mnist.data))

pickle.dump(c_mx, open("sgd_cmx.pkl", "wb"))

