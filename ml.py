#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


df=pd.read_csv("glass.csv")


# In[4]:


df


# In[5]:


X=df.iloc[:,:-1]


# In[6]:


y=df.iloc[:,-1]


# In[7]:


print(X)


# In[8]:


print(y)


# In[11]:


#aaplying the linear regression
# df.plot[x='Na',y='Type',style='o'?,
# color='red']
from sklearn.model_selection import train_test_split


# In[12]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=100)


# In[13]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()


# In[14]:


regressor.fit(X_train,y_train)


# In[16]:


y_pred=regressor.predict(X_test)


# In[17]:


print(regressor.coef_)


# In[18]:


print(regressor.intercept_)


# In[21]:


#aaplying Descion tree algorithm
X=df.iloc[:,:-1]
y=df.iloc[:,-1]


# In[22]:


from sklearn.model_selection import train_test_split


# In[24]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=100)


# In[26]:


from sklearn.tree import DecisionTreeClassifier
regressor=DecisionTreeClassifier()


# In[27]:


regressor.fit(X_train,y_train)


# In[28]:


y_pred=regressor.predict(X_test)


# In[29]:


y_pred


# In[30]:


from sklearn.metrics import accuracy_score
print('traing accuracy',accuracy_score(y_train,y_pred=regressor.predict(X_train)))


# In[31]:


print('testing accuracy',accuracy_score(y_test,y_pred=y_pred))


# In[33]:


#applying random forest algorithm
from sklearn.ensemble import RandomForestClassifier 


# In[34]:


clfr=RandomForestClassifier(random_state=100)


# In[37]:


clfr.fit(X_train,y_train)


# In[38]:


clfr.predict(X_test)


# In[39]:


print('testing accuracy',accuracy_score(y_test,y_pred=y_pred))


# In[40]:


#aaplying svm 
from sklearn.svm import SVC


# In[42]:


sc=SVC(kernel='linear')


# In[43]:


sc.fit(X_train,y_train)


# In[44]:


y_pred=sc.predict(X_test)


# In[45]:


y_pred


# In[49]:


from sklearn.metrics import classification_report,confusion_matrix


# In[51]:


print(confusion_matrix(y_test,y_pred))


# In[52]:


print(classification_report(y_test,y_pred))


# In[ ]:




