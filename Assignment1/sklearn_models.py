import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

dataset = pd.read_csv("LR_dataset/abalone/Dataset.data",sep="\s+", 
                 skiprows=1,  usecols=[0,1,2,3,4,5,6,7,8], 
                 names=['sex','length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight', 'rings' ])
dataset.head()
dataset = dataset.sample(frac = 1)
dataset.head()
train_columns = ['sex','length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight']
test_columns = ['rings']
X = np.array(dataset[train_columns])
Y = np.array(dataset[test_columns])

print(X)
print(Y)
print(X.shape)
print(Y.shape)

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

X = preprocess_dataset(X)
# print(X_train.shape)
print(X)
X_train = X[0:3000]
Y_train = Y[0:3000]
X_test = X[3000:]
Y_test = Y[3000:]
# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, Y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)
error = (np.sum((Y_test-y_pred)**2)/y_pred.shape[0])**0.5
print(error)


from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logistic_regression = LogisticRegression()
logistic_regression.fit(X_train,y_train)
y_pred = logistic_regression.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy on the test set :",accuracy*100, " %")
y_pred_train = logistic_regression.predict(X_train)
accuracy = metrics.accuracy_score(y_pred_train, y_train)
print("Accuracy Train:",accuracy*100, " %")