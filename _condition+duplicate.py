#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
data=pd.read_csv("germany.csv")
data


# In[2]:


data.sample


# In[3]:


data.sample(frac=.3)


# In[5]:


data.shape


# In[6]:


data.info()


# In[7]:


data.describe()


# In[8]:


data.values


# In[9]:


data["Insulin"].unique()


# In[10]:


data["Insulin"].nunique()


# In[11]:


data.head()


# In[12]:


data.groupby(by="Pregnancies").count()


# In[13]:


data.filter(like='100',axis=0)


# In[14]:


data.where(data.Glucose>130)


# In[6]:


data.where(data.Glucose>150)


# In[19]:


data.set_index("SkinThickness")


# In[20]:


data.duplicated()


# In[21]:


data.duplicated().sum()


# In[43]:


data.drop_duplicates(subset="Pregnancies",inplace=True)
data


# In[18]:


data.index.name='diabates'
data


# In[20]:


data[data.Insulin>130][['Glucose','Insulin']]


# In[21]:


data.query('BloodPressure>80')[['BMI','BloodPressure']]


# In[25]:


data.loc[50:70,['BMI','BloodPressure','Glucose']]


# In[27]:


data.iloc[20:30,2:-3]


# In[28]:


data.index


# In[29]:


data["BloodPressure"].value_counts()


# In[30]:


data.head(10)


# In[31]:


data.tail(7)


# In[32]:


data["total"]=data.Pregnancies*3
data


# In[37]:


data.groupby("Pregnancies").get_group(2)


# In[8]:


data.isnull().sum()


# In[ ]:





# In[ ]:




