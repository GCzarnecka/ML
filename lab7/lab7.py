#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml
import numpy as np
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.target = mnist.target.astype(np.uint8)
X = mnist["data"]
y = mnist["target"]


# In[2]:


from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score


# In[3]:


lista =[]
for k in range(8,13):
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_pred = kmeans.fit_predict(X)
    lista.append(silhouette_score(X, kmeans.labels_))
    print(silhouette_score(X, kmeans.labels_))


# In[4]:


import pickle
file = open("kmeans_sil.pkl","wb")
pickle.dump(lista, file)


# In[5]:


file = open("kmeans_sil.pkl","rb")
a = pickle.load(file)
file.close()
print(a)


# In[6]:


kmeans = KMeans(n_clusters=10, random_state=42)
y_pred = kmeans.fit_predict(X)


# In[7]:


kmeans.labels_


# In[8]:


from sklearn.metrics import confusion_matrix
cf = confusion_matrix(y, y_pred)
print(cf)


# In[9]:


print(y_pred)
print(y)


# In[10]:


lista2 = []
for i in cf:
    lista2.append(i[i.argmax()])
lista2 = list(set(lista2))
lista2.sort(reverse=True)


# In[11]:


print(lista2)


# In[12]:


file = open("kmeans_argmax.pkl","wb")
pickle.dump(lista2, file)


# In[13]:


file = open("kmeans_argmax.pkl","rb")
a = pickle.load(file)
file.close()
print(a)


# In[14]:


from sklearn.cluster import DBSCAN


# In[15]:


lista3 = []
lista4 =[]
for i in range(0, 300):
    for j in range(0, 300):
        if i != j:
            lista3.append(np.linalg.norm(X[i]-X[j]))
lista3.sort()
lista4 = lista3[:10]
lista4.sort(reverse = True)


# In[16]:


file = open("dist.pkl","wb")
pickle.dump(lista4, file)


# In[17]:


file = open("dist.pkl","rb")
a = pickle.load(file)
file.close()
print(a)


# In[18]:


lista4.sort()
s = (lista4[0] + lista4[1] + lista4[2]) / 3.0
print(s)


# In[19]:


lista5 = []
i = s
while i <=  s + 0.1 * s:
    dbscan = DBSCAN(eps = i)
    dbscan.fit(X)
    lista5.append(len(set(dbscan.labels_)))
    i += i * 0.04


# In[20]:


print(lista5)


# In[21]:


file = open("dbscan_len.pkl","wb")
pickle.dump(lista5, file)


# In[22]:


file = open("dbscan_len.pkl","rb")
a = pickle.load(file)
file.close()
print(a)


# In[ ]:





# In[ ]:





# In[ ]:




