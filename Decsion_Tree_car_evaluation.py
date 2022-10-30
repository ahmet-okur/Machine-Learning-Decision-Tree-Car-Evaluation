#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import seaborn as sns


# In[37]:


car_quailty_df = pd.read_csv("/Users/ahmetokur/Desktop/Datasets/car_evaluation.csv")


# In[38]:


car_quailty_df


# In[39]:


car_quailty_df.info()


# In[40]:


car_quailty_df.columns


# In[41]:


headers = ['Buying_price', 'Maint_cost', 'Doors', 'Person_Capasity', 'lug_boot', 'Safety', 'Class']


# In[42]:


car_quailty_df.shape


# In[43]:


car_quailty_df.columns = headers


# In[44]:


features = ['Buying_price', 'Maint_cost', 'Doors', 'Person_Capasity', 'lug_boot', 'Safety']


# In[45]:


car_quailty_df.info()


# In[46]:


Buying_price = {'vhigh':1, 'high':2, 'med':3, 'low':4}
car_quailty_df.Buying_price = [Buying_price[i] for i in car_quailty_df.Buying_price]


# In[47]:


Maint_cost = {'vhigh':1, 'high':2, 'med':3, 'low':4}
car_quailty_df.Maint_cost = [Maint_cost[i] for i in car_quailty_df.Maint_cost]


# In[48]:


Doors = {'2':2, '3':3, '4':4, '5more':5}
car_quailty_df.Doors = [Doors[i] for i in car_quailty_df.Doors]


# In[49]:


Person_Capasity = {'2':2, '4':4, 'more':1}
car_quailty_df.Person_Capasity = [Person_Capasity[i] for i in car_quailty_df.Person_Capasity]


# In[50]:


lug_boot = {'big':1, 'med':2, 'small':3}
car_quailty_df.lug_boot = [lug_boot[i] for i in car_quailty_df.lug_boot]


# In[52]:


Safety = {'high':1, 'med':2, 'low':3}
car_quailty_df.Safety = [Safety[i] for i in car_quailty_df.Safety]


# In[53]:


Class = {'vgood':1, 'good':2, 'acc':3, 'unacc':4}
car_quailty_df.Class = [Class[i] for i in car_quailty_df.Class]


# In[55]:


car_quailty_df.head(10)


# In[56]:


x = car_quailty_df[features]
y = car_quailty_df.Class


# In[57]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[58]:


from sklearn.tree import DecisionTreeClassifier


# In[59]:


dt = DecisionTreeClassifier()


# In[60]:


dt.fit(x_train, y_train)


# In[66]:


y_pred = dt.predict(x_test)
y_pred


# In[65]:


dt.score(x_test, y_test)


# In[ ]:




