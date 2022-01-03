#!/usr/bin/env python
# coding: utf-8

# In[109]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# In[110]:


df_raw = pd.read_csv('data.csv', encoding= 'unicode_escape')
df_raw.head()


# In[111]:


df = df_raw[['job_title', 'age', 'salary', 'exp. period']]
df.head()


# In[112]:


from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values= np.nan, strategy = 'mean')
imp = imp.fit(df['salary'].values.reshape(-1, 1))
df['salary'] = imp.transform(df['salary'].values.reshape(-1, 1))
df['salary']


# In[113]:


from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values= np.nan, strategy = 'mean')
imp = imp.fit(df['age'].values.reshape(-1, 1))
df['age'] = imp.transform(df['age'].values.reshape(-1, 1))
df['age']


# In[114]:


X = df[['age', 'salary', 'exp. period']]
X.head()


# In[115]:


y = df['job_title']
y.head()


# In[116]:


Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size = 0.8, stratify=y)


# In[117]:


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)


# In[118]:


from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_model)


# In[119]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=10)
scores.mean()


# In[120]:


from sklearn.neighbors import KNeighborsClassifier


# In[121]:


model = KNeighborsClassifier(n_neighbors = 3)


# In[122]:


model.fit(Xtrain, ytrain)


# In[123]:


y_model = model.predict(Xtest)


# In[124]:


accuracy_score(ytest, y_model)


# In[125]:


scores = cross_val_score(model, X, y, cv=10)
scores.mean()


# In[ ]:




