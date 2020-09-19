#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scratch import *
import random 


# In[2]:


preprocess = MyPreProcessor()
X,Y=preprocess.pre_process(2)
print(X)
print(Y)


# In[3]:


ratio = X.shape[0]//10
X_train = X[0:7*ratio, :]
y_train = Y[0:7*ratio, :]
X_val = X[7*ratio : 8*ratio, :]
y_val = Y[7*ratio : 8*ratio, :]
X_test = X[8*ratio:, :]
y_test = Y[8*ratio:,:]
print(X_train.shape,y_train.shape )
print(X_test.shape,y_test.shape )
print(X_val.shape, y_val.shape)


# In[4]:


pp = MyPreProcessor()
W = pp.get_analytical_sol()
print(W)
X,y = pp.pre_process(0)
bias = np.zeros((X.shape[0],1))
bias.fill(1)
X = np.append(X,bias, axis = 1)
y_pred = np.dot(X,W)
error = (np.sum((y-y_pred)**2)/y_pred.shape[0])**0.5
mae = np.sum(abs(y-y_pred))/y_pred.shape[0]
print(error)
print(mae)


# In[5]:


"""Running SGD on Banknote dataset and with LR = 0.0001, 0.01, 10"""
L = [0.0001, 0.0001,0.01,10]
Epochs = [10000,400000, 10000, 10000]


# In[6]:


for i in range(4):
    print("\n For learning rate ", L[i], "------------------------------------------------------------ ")
    log_regressor_sgd = MyLogisticRegression()
    log_regressor_sgd.fit(X_train, y_train, X_val, y_val, epochs = Epochs[i] ,learning_rate = L[i], grad_type = "sgd")
    print(log_regressor_sgd.W)
    print(log_regressor_sgd.b)
    loss_history_sgd = log_regressor_sgd.loss_history
    val_loss_history_sgd = log_regressor_sgd.val_loss_history
    train_acc_history_sgd = log_regressor_sgd.train_acc_history
    val_acc_history_sgd = log_regressor_sgd.val_acc_history
    plt.plot([x for x in range(len(loss_history_sgd))], loss_history_sgd, label = "Training  loss " )
    plt.plot([x for x in range(len(val_loss_history_sgd))], val_loss_history_sgd, label = "Validation  loss")
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    plt.plot([x for x in range(len(train_acc_history_sgd))], train_acc_history_sgd, label = "Training  accuracy " )
    plt.plot([x for x in range(len(val_acc_history_sgd))], val_acc_history_sgd, label = "Validation  accuracy")
    plt.xlabel('iterations')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
    
    print("\n -------------------------------------------------------------------------------------------")


# In[7]:


"""Running BGD on Banknote dataset and with LR = 0.0001, 0.01, 10"""
L = [0.0001, 0.0001,0.01,10]
Epochs = [10000,400000, 10000, 10000]


# In[8]:


for i in range(4):
    print("\n For learning rate ", L[i], "------------------------------------------------------------ ")
    log_regressor_bgd = MyLogisticRegression()
    log_regressor_bgd.fit(X_train, y_train, X_val, y_val, epochs = Epochs[i] ,learning_rate = L[i], grad_type = "bgd")
    print(log_regressor_bgd.W)
    print(log_regressor_bgd.b)
    loss_history_bgd = log_regressor_bgd.loss_history
    val_loss_history_bgd = log_regressor_bgd.val_loss_history
    train_acc_history_bgd = log_regressor_bgd.train_acc_history
    val_acc_history_bgd = log_regressor_bgd.val_acc_history
    plt.plot([x for x in range(len(loss_history_bgd))], loss_history_bgd, label = "Training  Loss " )
    plt.plot([x for x in range(len(val_loss_history_bgd))], val_loss_history_bgd, label = "Validation  Loss")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    plt.plot([x for x in range(len(train_acc_history_bgd))], train_acc_history_bgd, label = "Training  accuracy " )
    plt.plot([x for x in range(len(val_acc_history_bgd))], val_acc_history_bgd, label = "Validation  accuracy")
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    print("\n -------------------------------------------------------------------------------------------")


# In[9]:


"Custom learning rate chosen by hit and trial"


# In[10]:


""" SGD on 0.005 learning rate"""
log_regressor_sgd = MyLogisticRegression()
log_regressor_sgd.fit(X_train, y_train, X_val, y_val, epochs = 10000 ,learning_rate = 0.05, grad_type = "sgd")
print(log_regressor_sgd.W)
print(log_regressor_sgd.b)
loss_history_sgd = log_regressor_sgd.loss_history
val_loss_history_sgd = log_regressor_sgd.val_loss_history
train_acc_history_sgd = log_regressor_sgd.train_acc_history
val_acc_history_sgd = log_regressor_sgd.val_acc_history
plt.plot([x for x in range(len(loss_history_sgd))], loss_history_sgd, label = "Training  loss " )
plt.plot([x for x in range(len(val_loss_history_sgd))], val_loss_history_sgd, label = "Validation  loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
plt.plot([x for x in range(len(train_acc_history_sgd))], train_acc_history_sgd, label = "Training  accuracy " )
plt.plot([x for x in range(len(val_acc_history_sgd))], val_acc_history_sgd, label = "Validation  accuracy")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()


# In[11]:


log_regressor_bgd = MyLogisticRegression()
log_regressor_bgd.fit(X_train, y_train, X_val, y_val, epochs = 10000 ,learning_rate = 0.05, grad_type = "bgd")
print(log_regressor_bgd.W)
print(log_regressor_bgd.b)
loss_history_bgd = log_regressor_bgd.loss_history
val_loss_history_bgd = log_regressor_bgd.val_loss_history
train_acc_history_bgd = log_regressor_bgd.train_acc_history
val_acc_history_bgd = log_regressor_bgd.val_acc_history
plt.plot([x for x in range(10000)], loss_history_bgd, label = "Training  Loss " )
plt.plot([x for x in range(10000)], val_loss_history_bgd, label = "Validation  Loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
plt.plot([x for x in range(10000)], train_acc_history_bgd, label = "Training  accuracy " )
plt.plot([x for x in range(10000)], val_acc_history_bgd, label = "Validation  accuracy")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()


# In[12]:


"""Exploratory Data Analysis for the BankNote dataset """


# In[13]:


df = pd.read_csv("LoR_dataset/data_banknote_authentication.txt",sep="," , names = ["col1", "col2", "col3", "col4" , "val"])


# In[14]:


df.head()


# In[15]:


import seaborn as sns
print(df.corr())
sns.heatmap(df.corr(), annot = True)


# In[16]:


df.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8); # ; avoid having the matplotlib verbose informations


# In[30]:


sns.pairplot(df,hue='val',diag_kind="hist")


# In[17]:


print(df)


# In[28]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[29]:


logistic_regression = LogisticRegression(max_iter=10000)
logistic_regression.fit(X_train,y_train)
y_pred = logistic_regression.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy on the test set :",accuracy*100, " %")
y_pred_train = logistic_regression.predict(X_train)
accuracy = metrics.accuracy_score(y_pred_train, y_train)
print("Accuracy Train:",accuracy*100, " %")


# In[ ]:




