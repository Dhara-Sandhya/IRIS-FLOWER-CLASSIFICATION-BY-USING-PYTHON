#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[19]:


# Data Reading 
data=pd.read_csv(r"C:\Users\Darasandhya\OneDrive\Desktop\onlineinternship2\Iris.csv")
data


# In[20]:


# Top head records
data.head()


# In[21]:


# Last 5 records 
data.tail()


# In[22]:


# Describing that data 
data.describe()


# In[23]:


# Visualazing that data 
sns.pairplot(data,hue="Species") # Cheking the which flower having high and low length & width 


# In[24]:


# Now lets sepparate the data
df=data.values # we are importing the flower data values on one variable df
x=df[:,0:5]
y=df[:,5]

df #so the actual values of the data this we having 6 columns and 150 rows


# In[25]:


x # x value here i am taking only 5 columns and 150


# In[26]:


y # y value here i am taking it only prints the last varibles in the coumn 


# In[27]:


# preparing and spliting the data into testing and traing 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2) # here we splitting data into 20% of test and 80% train


# ## Support Vector Machine

# In[28]:


from sklearn.svm import SVC # importing the support vector machines as svc

# Fitting the training data into the model
model_svc = SVC()
model_svc.fit(x_train,y_train)


# In[30]:


prediction1= model_svc.predict(x_test)
# Now we are going to caluclating the accurace

from sklearn.metrics import accuracy_score
 # from here we are calculating the accurancy score
print(accuracy_score(y_test,prediction1)*100)
    


# In[31]:


# if you want to check manually you can use this form prediction

for i in range(len(prediction1)):
    print(y_test[i],prediction1[i]) # here we can check manually

