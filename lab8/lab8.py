#!/usr/bin/env python
# coding: utf-8

# In[411]:


from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer()


# In[412]:


from sklearn.datasets import load_iris
data_iris = load_iris()


# In[413]:


import numpy as np
from sklearn.decomposition import PCA
import pickle


# In[414]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
bc_scaled = scaler.fit_transform(data_breast_cancer.data)


# In[415]:


pca_bc = PCA(n_components = 0.9)
bc = pca_bc.fit_transform(bc_scaled)


# In[416]:


file = open("pca_bc.pkl", 'wb')
pickle.dump(pca_bc.explained_variance_ratio_, file)
    
file = open("pca_bc.pkl",'rb')
a = pickle.load(file)
file.close()
print(a)


# In[417]:


idx_bc = [np.argmax(row) for row in pca_bc.components_]


# In[418]:


file = open("idx_bc.pkl", 'wb')
pickle.dump(idx_bc, file)
    
file = open("idx_bc.pkl",'rb')
a = pickle.load(file)
file.close()
print(a)


# In[419]:


ir_scaled = scaler.fit_transform(data_iris.data)


# In[420]:


pca_ir = PCA(n_components = 0.9)
ir = pca_ir.fit_transform(ir_scaled)


# In[421]:


file = open("pca_ir.pkl", 'wb')
pickle.dump(pca_ir.explained_variance_ratio_, file)
    
file = open("pca_ir.pkl",'rb')
a = pickle.load(file)
file.close()
print(a)


# In[422]:


idx_ir = [np.argmax(row) for row in pca_ir.components_]


# In[423]:


file = open("idx_ir.pkl", 'wb')
pickle.dump(idx_ir, file)
    
file = open("idx_ir.pkl",'rb')
a = pickle.load(file)
file.close()
print(a)


# In[ ]:





# In[ ]:





# In[ ]:




