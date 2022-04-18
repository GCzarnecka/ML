#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
data_breast_cancer = datasets.load_breast_cancer(as_frame=True)


# In[2]:


X_train, X_test, y_train, y_test = train_test_split(data_breast_cancer['data'][['mean texture','mean symmetry']], data_breast_cancer['target'], test_size=0.2)


# In[3]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier


# In[4]:


tree_clf = DecisionTreeClassifier()


# In[5]:


log_clf = LogisticRegression()


# In[6]:


knn = KNeighborsClassifier()


# In[7]:


voting_clf_hard = VotingClassifier(
    estimators=[('tree', tree_clf), ('log', log_clf), ('knn', knn)], voting='hard')


# In[8]:


voting_clf_soft = VotingClassifier(
    estimators=[('tree', tree_clf), ('log', log_clf), ('knn', knn)], voting='soft')


# In[9]:


lista = []
classifiers = []
for clf in [tree_clf, log_clf, knn, voting_clf_hard, voting_clf_soft]:
    clf.fit(X_train, y_train)
    lista.append((
        accuracy_score(y_train, clf.predict(X_train)), accuracy_score(y_test, clf.predict(X_test))
    ))
    classifiers.append(clf)


# In[11]:


import pickle


# In[12]:


file = open("acc_vote.pkl", 'wb')
pickle.dump(lista, file)

file = open("acc_vote.pkl",'rb')
a = pickle.load(file)
file.close()
print(a)


# In[13]:


file = open("vote.pkl", 'wb')
pickle.dump(classifiers, file)

file = open("vote.pkl",'rb')
a = pickle.load(file)
file.close()
print(a)


# In[14]:


from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[15]:


bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, max_samples=1.0, bootstrap=True, random_state=42)

bag_clf_half = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, max_samples=0.5, bootstrap=True, random_state=42)

pasting = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, max_samples=1.0, bootstrap=False, random_state=42)

pasting_half = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, max_samples=0.5, bootstrap=False, random_state=42)

random_forest = RandomForestClassifier(n_estimators=30)

ada_boost = AdaBoostClassifier(n_estimators=30)

gradient_boosting = GradientBoostingClassifier(n_estimators=30).fit(X_train, y_train)


# In[16]:


lista = []
classifiers = []
for bag in [bag_clf, bag_clf_half, pasting, pasting_half, random_forest, ada_boost, gradient_boosting]:
    bag.fit(X_train, y_train)
    lista.append((
        accuracy_score(y_train, bag.predict(X_train)), accuracy_score(y_test, bag.predict(X_test))
    ))
    classifiers.append(bag)


# In[17]:


file = open("acc_bag.pkl", 'wb')
pickle.dump(lista, file)

file = open("acc_bag.pkl",'rb')
a = pickle.load(file)
file.close()
print(a)


# In[18]:


file = open("bag.pkl", 'wb')
pickle.dump(classifiers, file)

file = open("bag.pkl",'rb')
a = pickle.load(file)
file.close()
print(a)


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(
    data_breast_cancer['data'], data_breast_cancer['target'], test_size=0.2)


# In[20]:


bag = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=30, max_samples=0.5,
    bootstrap_features=True, max_features=2)
bag.fit(X_train, y_train)


# In[21]:


file = open("acc_fea.pkl", 'wb')
pickle.dump([accuracy_score(y_train, bag.predict(X_train)), accuracy_score(y_test, bag.predict(X_test))], file)

file = open("acc_fea.pkl",'rb')
a = pickle.load(file)
file.close()
print(a)


# In[22]:


file = open("fea.pkl", 'wb')
pickle.dump([bag], file)

file = open("fea.pkl",'rb')
a = pickle.load(file)
file.close()
print(a)


# In[23]:


df = pd.DataFrame({
    "acc train":[],
    "acc test":[],
    "features":[]})


# In[24]:


for i, e in enumerate(bag.estimators_):
    features = data_breast_cancer.feature_names[np.array(bag.estimators_features_[i])]
    df_row = pd.DataFrame({
        'acc train': [accuracy_score(y_train, e.predict(X_train[features]))],
        'acc test': [accuracy_score(y_test, e.predict(X_test[features]))],
        'features': [features]
    })
    df = pd.concat([df, df_row])


# In[25]:


df.sort_values(by='acc train', ascending=False, inplace=True)
df.sort_values(by='acc test', ascending=False, inplace=True)
df


# In[26]:


file = open("acc_fea_rank.pkl", 'wb')
pickle.dump(df, file)
    
file = open("acc_fea_rank.pkl",'rb')
a = pickle.load(file)
file.close()
print(a)


# In[ ]:





# In[ ]:




