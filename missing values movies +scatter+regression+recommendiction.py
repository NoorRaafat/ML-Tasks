#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
data=pd.read_csv("movies1.csv")
data


# In[2]:


data.isnull()


# In[3]:


data.isnull().sum()


# In[4]:


data.interpolate(method="linear" ,inplace= True) # linear (2/(الاول +التاني))


# In[7]:


data.isnull().sum()


# In[5]:


data.notnull().sum()#عشان اشوف مجموع القيم اللي مش فاضيه ف كل كولوم


# In[6]:


data.columns[data.isna().any()]#عشان أرجع الكولوم اللي فيها قيم فاضيه)


# In[7]:


data.drop(['Revenue (Millions)', 'Metascore'],axis=1)#( وده غلط)أمسح العمود اللي فيها missing value وده اول حل


# In[8]:


data


# In[9]:


data.dropna()#عشان اشيل الرو الفاضيه


# In[10]:


data.fillna(0)#عشان احط قيمه مكان اي قيمه فاضيه ف الداتا سيت


# In[11]:


data['Revenue (Millions)'].fillna("noor")


# In[12]:


data.fillna(method="ffill")#fill forward(pad) the nan value according to the value in the forward


# In[13]:


data.fillna(method="bfill")#back forward (backfill)the nan value according to the value in the back


# In[14]:


data.fillna(data.mean()) # nan =mean here


# In[15]:


data.interpolate(method="linear") # linear (2/(الاول +التاني))


# In[16]:


from sklearn.impute import SimpleImputer


# In[17]:


cols=data.loc[:,['Revenue (Millions)', 'Metascore']]#عشان اجيب الكولوم اللي فيها قيم فاضيه واشتغل عليهم لوحدهم
cols


# In[18]:


my_imputer=SimpleImputer()#  (strategy: mean, most_frequent )عشان اخد اوبجكت من سمبل امبيوتد


# In[19]:


imputed_values=my_imputer.fit_transform(cols)#by deafult mean value instead of nan value
imputed_values


# In[20]:


new_values=pd.DataFrame(imputed_values)# to transform arraryvalues to datafram values
new_values


# In[21]:


new_values.columns=cols.columns# to take the same name columns upon
new_values


# In[22]:


data["Revenue (Millions)"]=new_values["Revenue (Millions)"]
data


# In[23]:


data.info()


# In[24]:


x=data.loc[:600,["Votes"]]
y=data.loc[:600,["Rating"]]
x_test=data.loc[:6001,["Votes"]]
y_desired=data.loc[:6001,["Rating"]]


# In[25]:


plt.scatter(x,y)
plt.show()


# In[26]:


from sklearn.linear_model import LinearRegression


# In[27]:


model=LinearRegression()


# In[28]:


model.fit(x,y)


# In[29]:


y_predicted=model.predict(x_test)
y_predicted


# In[30]:


data


# In[31]:


data.isnull().sum()


# In[32]:


x
y


# In[33]:


data


# In[34]:


data.drop(["Genre","Year","Rank","Runtime (Minutes)","Director","Description"],axis=1,inplace=True)


# In[35]:


data


# In[ ]:




