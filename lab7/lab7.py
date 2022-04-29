#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.datasets import fetch_openml
import numpy as np
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.target = mnist.target.astype(np.uint8)
X = mnist["data"]
y = mnist["target"]


# In[3]:


from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score


# In[4]:


lista =[]
for k in range(8,13):
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_pred = kmeans.fit_predict(X)
    lista.append(silhouette_score(X, kmeans.labels_))
    print(silhouette_score(X, kmeans.labels_))


# In[5]:


import pickle
file = open("kmeans_sil.pkl","wb")
pickle.dump(lista, file)


# In[6]:


file = open("kmeans_sil.pkl","rb")
a = pickle.load(file)
file.close()
print(a)


# In[7]:


kmeans = KMeans(n_clusters=10, random_state=42)
y_pred = kmeans.fit_predict(X)


# In[8]:


kmeans.labels_


# In[9]:


from sklearn.metrics import confusion_matrix
cf = confusion_matrix(y, y_pred)
print(cf)


# In[10]:


print(y_pred)
print(y)


# In[11]:


lista2 = []
for i in cf:
    lista2.append(i.argmax())
lista2 = list(set(lista2))
lista2.sort()


# In[12]:


print(lista2)


# In[13]:


file = open("kmeans_argmax.pkl","wb")
pickle.dump(lista2, file)


# In[14]:


file = open("kmeans_argmax.pkl","rb")
a = pickle.load(file)
file.close()
print(a)


# In[15]:


from sklearn.cluster import DBSCAN


# In[46]:


lista3 = []
lista4 = []
for i in range(0, 300):
    for j in range(i+1, 300):
        lista3.append(np.linalg.norm(X[i]-X[j]))
lista3.sort()
lista4 = lista3[:10]
lista4.sort()


# In[37]:


file = open("dist.pkl","wb")
pickle.dump(lista4, file)


# In[38]:


file = open("dist.pkl","rb")
a = pickle.load(file)
file.close()
print(a)


# In[40]:


s = (lista4[0] + lista4[1] + lista4[2]) / 3.0
print(s)


# In[41]:


lista5 = []
i = s
while i <=  s + 0.1 * s:
    dbscan = DBSCAN(eps = i)
    dbscan.fit(X)
    lista5.append(len(set(dbscan.labels_)))
    i += i * 0.04


# In[42]:


print(lista5)


# In[43]:


file = open("dbscan_len.pkl","wb")
pickle.dump(lista5, file)


# In[44]:


file = open("dbscan_len.pkl","rb")
a = pickle.load(file)
file.close()
print(a)


# In[ ]:





# In[ ]:





# In[ ]:




