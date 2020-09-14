#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scratch import *


# In[2]:


""" Running Linear Regression on Alabone dataset """


# In[3]:


dataset = pd.read_csv("LR_dataset/abalone/Dataset.data",sep="\s+", 
                 skiprows=1,  usecols=[0,1,2,3,4,5,6,7,8], 
                 names=['sex','length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight', 'rings' ])


# In[4]:


dataset.head()


# In[5]:


dataset = dataset.sample(frac = 1)
dataset.head()


# In[6]:


train_columns = ['sex','length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight']
test_columns = ['rings']
X = np.array(dataset[train_columns])
Y = np.array(dataset[test_columns])

print(X)
print(Y)
print(X.shape)
print(Y.shape)


# In[7]:


def preprocess_dataset(X):
    """
    Preprocess the dataset, converts the 'sex' column numerical and normalise the dataset

    Parameters
    ----------
    X : 2-dimensional numpy array of shape (n_samples, n_features) which is data.
        
    Returns
    -------
    X : Preprocessed X
    """
    m = X.shape[0]
    for i in range(m):
        if(X[i,0] == 'M'):
            X[i,0] = 1  #if M then 1
        elif(X[i,0] == 'F'):
            X[i,0] = 2  #if F then 2
        elif(X[i,0] == 'I'):
            X[i,0] = 3
    X = X.T
#     print(X.shape)
    for i in range(X.shape[0]):
        mean = X[i, :].mean()
        std =  X[i, :].std()
        X[i,:] = (X[i,:] - mean)/std
    
    return X.T
    


# In[8]:


X = preprocess_dataset(X)
print(X.shape)
print(X)


# In[9]:


def k_fold_cross_validation(X, y, k=5):
    """ Performs K fold cross validation
    Parameters
    ----------
    model : instance of the model to be used
    X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as data.
    y : 1-dimensional numpy array of shape (n_samples,) which acts as labels.
    k : number of folds, default = 5
    
    Returns
    -------
    model : instance of model
    """
    m = X.shape[0]  #number of examples
    fold_size = int(m/k)
    start = 0
    end = fold_size
    rmse_train_history = []
    mae_train_history = []
    rmse_val_history = []
    mae_val_history = []
    for i in range(k):
        Xtrain_i = np.concatenate((X[0:start], X[end+1:]))
        ytrain_i = np.concatenate((y[0:start],y[end+1:]))
        X_test =  X[start:end]
        y_test = y[start:end]
        model = MyLinearRegression()
        model.fit(Xtrain_i,ytrain_i,X_test,y_test)
#         y_pred = model.predict(X_test)
#         rmse_val = np.sum(((y_test - y_pred)**2)**0.5)/len(y_pred)
#         mae_val = np.sum(np.abs(y_test - y_pred)/len(y_pred))
#         rmse_train_history.append(model.get_rmse_loss())
#         mae_train_history.append(model.get_mae_loss())
#         rmse_val_history.append(rmse_val)
#         mae_val_history.append(mae_val)
#         print(Xtrain_i.shape,ytrain_i.shape,X_test.shape,y_test.shape)
    
    return model,rmse_train_history,mae_train_history,rmse_val_history,mae_val_history
    


# In[10]:


a,b,c,d,e = k_fold_cross_validation(X,Y)


# In[ ]:



                          


# In[ ]:




