#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scratch import *


# In[2]:


""" Running Linear Regression on Alabone dataset """


# In[3]:


pp = MyPreProcessor()
X,Y = pp.pre_process(0)
print(X)
print(Y)


# In[4]:


def k_fold_cross_validation(X, y, k=5, loss = "rmse", epochs = 8000, learning_rate = 0.01):
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
    models = {}
    for i in range(k):
        Xtrain_i = np.concatenate((X[0:start], X[end+1:]))
        ytrain_i = np.concatenate((y[0:start],y[end+1:]))
        X_test =  X[start:end]
        y_test = y[start:end]
        model = MyLinearRegression()
        print("For fold : ", i , "/", k)
        model.fit(Xtrain_i,ytrain_i,X_test,y_test,epochs,learning_rate, loss)
        if(loss == "rmse"):
            models[i] = (model.rmse_train_history[-1], model.rmse_val_history[-1], np.array(model.rmse_train_history), np.array(model.rmse_val_history))
        if(loss == "mae"):
            models[i] = (model.mae_train_history[-1], model.mae_val_history[-1], np.array(model.mae_train_history), np.array(model.mae_val_history))
        print(model.W)
        print(model.b)
        start+=fold_size
        end+=fold_size
        
    avg_train = 0
    avg_val  = 0
    for i in range(len(models)):
        avg_train+=models[i][2]
        avg_val+=models[i][3]
            
    
    avg_train = avg_train/k
    avg_val = avg_val/k
    return models, avg_train, avg_val
    


# In[5]:


k_models = {}
for i in range(2,11):
    models_rmse,avg_train_loss, avg_val_loss = k_fold_cross_validation(X,Y, k = i, epochs = 5000, learning_rate = 0.01, loss = "rmse")
#     print("The average loss with k = ", i, " is ", " train loss = ", avg_train_loss[-1], "")
    k_models[i] = (avg_train_loss[-1], avg_val_loss[-1])
print(k_models)
    


# In[6]:


for i in range(2,11):
    print("The average loss with k = ", i, " is ", " train loss = ", k_models[i][0], " validation loss = ", k_models[i][1])


# In[7]:


models_mae,avg_train_loss, avg_val_loss = k_fold_cross_validation(X,Y, k = 10, epochs = 5000, learning_rate = 0.01, loss = "mae")


# In[8]:


import matplotlib.pyplot as plt


# In[9]:


plt.plot([x for x in range(len(avg_val_loss))],avg_val_loss, label = "Average Validation Loss")
plt.plot([x for x in range(len(avg_train_loss))], avg_train_loss, label = "Average Training Loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.plot([x for x in range(len(models_mae[1][2]))],models_mae[1][2], label = "Best Training Loss")
plt.plot([x for x in range(len(models_mae[1][3]))], models_mae[1][3], label = "Best Validation Loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


# In[10]:


models,avg_train_loss, avg_val_loss = k_fold_cross_validation(X,Y, k = 10, epochs = 5000, learning_rate = 0.01)


# In[11]:


plt.plot([x for x in range(len(avg_val_loss))],avg_val_loss, label = "Average Validation Loss")
plt.plot([x for x in range(len(avg_train_loss))], avg_train_loss, label = "Average Training Loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.plot([x for x in range(len(models[1][2]))],models[1][2], label = "Best Training Loss")
plt.plot([x for x in range(len(models[1][3]))], models[1][3], label = "Best Validation Loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


# In[12]:


print("Stats for LR MAE Loss")
for i in range(len(models_mae)):
    print("For CV number ",i, "the train loss = ", models_mae[i][0], " and the val loss = ", models_mae[i][1] )
print("Stats for LR RMSE Loss")
for i in range(len(models)):
    print("For CV number ",i, "the train loss = ", models[i][0], " and the val loss = ", models[i][1] )


# In[13]:


"""LR on Video Game dataset """


# In[14]:


pp = MyPreProcessor()
X,Y = pp.pre_process(1)
print(X)
print(Y)


# In[15]:


k_models_video_game = {}
for i in range(2,11):
    models_rmse,avg_train_loss, avg_val_loss = k_fold_cross_validation(X,Y, k = i, epochs = 5000, learning_rate = 0.01, loss = "rmse")
#     print("The average loss with k = ", i, " is ", " train loss = ", avg_train_loss[-1], "")
    k_models_video_game[i] = (avg_train_loss[-1], avg_val_loss[-1])
print(k_models_video_game)
    


# In[16]:


for i in range(2,11):
    print("The average loss with k = ", i, " is ", " train loss = ", k_models_video_game[i][0], " validation loss = ", k_models_video_game[i][1])


# In[17]:


video_game_model_rmse,avg_train_loss, avg_val_loss = k_fold_cross_validation(X,Y,k=10, epochs = 3500, learning_rate = 0.005)


# In[18]:


plt.plot([x for x in range(len(avg_val_loss))],avg_val_loss, label = "Average Validation Loss")
plt.plot([x for x in range(len(avg_train_loss))], avg_train_loss, label = "Average Training Loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


plt.plot([x for x in range(len(video_game_model_rmse[6][2]))],video_game_model_rmse[6][2], label = "Best Training Loss")
plt.plot([x for x in range(len(video_game_model_rmse[6][3]))], video_game_model_rmse[6][3], label = "Best Validation Loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


# In[19]:


video_game_model_mae,avg_train_loss, avg_val_loss = k_fold_cross_validation(X,Y,k=10, epochs = 3500,learning_rate = 0.005,loss = "mae")


# In[20]:


plt.plot([x for x in range(len(avg_val_loss))],avg_val_loss, label = "Average Validation Loss")
plt.plot([x for x in range(len(avg_train_loss))], avg_train_loss, label = "Average Training Loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


plt.plot([x for x in range(len(video_game_model_mae[6][2]))],video_game_model_mae[6][2], label = "Best Training Loss")
plt.plot([x for x in range(len(video_game_model_mae[6][3]))], video_game_model_mae[6][3], label = "Best Validation Loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


# In[21]:


print("Stats for LR MAE Loss")
for i in range(len(video_game_model_mae)):
    print("For CV number ",i, "the train loss = ", video_game_model_mae[i][0], " and the val loss = ", video_game_model_mae[i][1] )
print("Stats for LR RMSE Loss")
for i in range(len(video_game_model_rmse)):
    print("For CV number ",i, "the train loss = ", video_game_model_rmse[i][0], " and the val loss = ", video_game_model_rmse[i][1] )


# In[ ]:





# In[23]:


df = pd.read_csv("LR_dataset/VideoGameDataset - Video_Games_Sales_as_at_22_Dec_2016.csv",  usecols=['Critic_Score','User_Score','Global_Sales'])
df.head()


# In[24]:


df.isna().sum()


# In[27]:


df["Critic_Score"].value_counts()


# In[28]:


df["User_Score"].value_counts()


# In[74]:


def get_analytical_sol(X,y):
#     bias = np.zeros((X.shape[0],1))
#     bias.fill(1)
#     X = np.append(X,bias, axis = 1)
#     print(X)
    W = np.dot(np.linalg.inv(np.dot(X.T,X)), np.dot(X.T,y))
    return W


# In[75]:


preproc = MyPreProcessor()
X,y = preproc.pre_process(0)


# In[76]:


m = X.shape[0]  #number of examples
fold_size = int(m/10)
start = fold_size
end = 2*fold_size


# In[77]:


Xtrain_i = np.concatenate((X[0:start], X[end+1:]))
ytrain_i = np.concatenate((y[0:start],y[end+1:]))
X_test =  X[start:end]
y_test = y[start:end]

bias = np.zeros((Xtrain_i.shape[0],1))
bias.fill(1)
Xtrain_i = np.append(Xtrain_i,bias, axis = 1)

bias = np.zeros((X_test.shape[0],1))
bias.fill(1)
X_test = np.append(X_test,bias, axis = 1)


# In[78]:


print(X_test)
print(Xtrain_i)
print(Xtrain_i.shape)
print(X_test.shape)
print(ytrain_i.shape)
print(y_test.shape)
print(ytrain_i)
print(y_test)


# In[79]:


W = get_analytical_sol(Xtrain_i,ytrain_i)
print(W)


# In[80]:


y_pred_train = np.dot(Xtrain_i,W)
error = (np.sum(abs(ytrain_i-y_pred_train))/y_pred_train.shape[0])
print(error)


# In[81]:


y_pred_test = np.dot(X_test,W)
error = (np.sum(abs(y_test-y_pred_test))/y_pred_test.shape[0])
print(error)


# In[45]:


Xtrain_i = Xtrain_i[:,:-1]
X_test = X_test[:,:-1]


# In[ ]:




