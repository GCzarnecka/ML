#!/usr/bin/env python
# coding: utf-8

# In[59]:


from sklearn import datasets


# In[60]:


import numpy as np
from sklearn.model_selection import train_test_split


# In[61]:


data_breast_cancer = datasets.load_breast_cancer(as_frame=True)


# In[62]:


data_iris = datasets.load_iris(as_frame=True)


# In[63]:


X = data_breast_cancer.data[["mean area","mean smoothness"]]
y = data_breast_cancer.target


# In[64]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[65]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# In[66]:


svm_clf_1 = Pipeline([ ("linear_svc", LinearSVC(C=1, loss="hinge", random_state=42)) ])
svm_clf_1.fit(X_train, y_train)


# In[67]:


svm_clf_2 = Pipeline([ ("scaler", StandardScaler()), ("linear_svc", LinearSVC(C=1, loss="hinge", random_state=42))])
svm_clf_2.fit(X_train, y_train)


# In[68]:


from sklearn.metrics import accuracy_score
accuracy_score(y_train, svm_clf_1.predict(X_train))
accuracy_score(y_test, svm_clf_1.predict(X_test))


# In[69]:


lista = [accuracy_score(y_train, svm_clf_1.predict(X_train)),
         accuracy_score(y_test, svm_clf_1.predict(X_test)),
         accuracy_score(y_train, svm_clf_2.predict(X_train)),
         accuracy_score(y_test, svm_clf_2.predict(X_test))]


# In[70]:


import pickle
file = open('bc_acc.pkl', 'wb')
pickle.dump(lista, file)


# In[71]:


X = data_iris.data[["petal length (cm)","petal width (cm)"]]
y = data_iris.target


# In[72]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[73]:


svm_clf_1 = Pipeline([("linear_svc", LinearSVC(C=1, loss="hinge", random_state=42))])
svm_clf_1.fit(X_train, y_train)


# In[74]:


svm_clf_2 = Pipeline([("scaler", StandardScaler()), ("linear_svc", LinearSVC(C=1, loss="hinge", random_state=42)) ])
svm_clf_2.fit(X_train, y_train)


# In[75]:


lista2 = [accuracy_score(y_train, svm_clf_1.predict(X_train)),
         accuracy_score(y_test, svm_clf_1.predict(X_test)),
         accuracy_score(y_train, svm_clf_2.predict(X_train)),
         accuracy_score(y_test, svm_clf_2.predict(X_test))]


# In[76]:


file = open('iris_acc.pkl', 'wb')
pickle.dump(lista2, file)


# In[ ]:





# In[ ]:




