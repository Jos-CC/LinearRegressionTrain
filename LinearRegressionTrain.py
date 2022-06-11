#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd

class LinearRegression:
    def __init__(self, le=0.001, iters=1000): #define class members 
        self.le=le
        self.iters=iters
        self.weights=None
        self.bias=None
        
    def fitting(self,X,y): #gradient descent
        n_samples, n_features=X.shape
        self.weights=np.zeros(n_features)
        self.bias=0
        
        for _ in range(self.iters):
            y_predicted=np.dot(X,self.weights)+self.bias
            
            dw=(1/n_samples)*np.dot(X.T,(y_predicted-y))
            dinter=(1/n_samples)*(np.sum(y_predicted-y))
            
            self.weights-=self.le*dw
            self.bias=self.le*dinter
            
    def predict(self,X): #get our Y_predicted based on LMS algorithm 
        y_predicted=np.dot(X,self.weights)+self.bias
        return y_predicted


# In[2]:


train=pd.read_csv(r'C:\Users\hp\Desktop\Datasets\train.csv')


# In[3]:


train


# In[4]:


train.info()


# In[6]:


train.describe


# In[12]:


#Figure out categorical variables
train.select_dtypes(exclude=['number'])


# In[13]:


train.Street.unique()


# In[14]:


train.MSZoning.unique()


# In[ ]:




