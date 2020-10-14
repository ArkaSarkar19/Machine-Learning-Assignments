import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
df = pd.read_csv("Dataset/weight-height.csv")


X = np.array(df["Weight"])  #height
y = np.array(df["Height"])  #weight
np.random.seed(123)
index = np.random.permutation(X.shape[0])
np.take(X, index, axis = 0, out = X)
np.take(y, index, axis = 0, out = y)
X = X.reshape(-1,1)


from sklearn.preprocessing import StandardScaler
standardScalar = StandardScaler()
X = standardScalar.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)



n_samples = 100
y_pred_b = []
for i in range(n_samples):
    X_train_b = []
    y_train_b = []
    for _ in range(X_train.shape[0]):
        idx = np.random.randint(0,X_train.shape[0])
        X_train_b.append(X_train[idx,:])
        y_train_b.append(y_train[idx])
    X_train_b = np.array(X_train_b)
    y_train_b = np.array(y_train_b)
    
    X_train_b = np.array(X_train_b)
    y_train_b = np.array(y_train_b)
    
    model = LinearRegression()
    model.fit(X_train_b,y_train_b)
    y_pred = model.predict(X_test)
    y_pred_b.append(y_pred)
   
y_pred_b = np.array(y_pred_b)    


avg = np.sum(y_pred_b,axis = 0)/y_pred_b.shape[0]


bias = abs(y_test - avg).mean()
print("Bias : ", bias)

var = (np.sum((y_pred_b - avg)**2,axis = 0)/99).mean()
print("Variance :", var)

MSE = (np.sum((y_pred_b - y_test)**2,axis = 0)/99).mean()
print("MSE",MSE)

noise = MSE - (bias)**2 - var.mean()

print("MSE - bias**2 - Variance : ", noise)