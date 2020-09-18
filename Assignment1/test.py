# from scratch import MyLinearRegression, MyLogisticRegression, MyPreProcessor
# import numpy as np

# preprocessor = MyPreProcessor()

# print('Linear Regression')

# X, y = preprocessor.pre_process(0)

# # Create your k-fold splits or train-val-test splits as required

# Xtrain = np.empty((0,0))
# ytrain = np.empty((0))
# Xtest = np.empty((0,0))
# ytest = np.empty((0))

# linear = MyLinearRegression()
# linear.fit(Xtrain, ytrain)

# ypred = linear.predict(Xtest)

# print('Predicted Values:', ypred)
# print('True Values:', ytest)

# print('Logistic Regression')

# X, y = preprocessor.pre_process(2)

# # Create your k-fold splits or train-val-test splits as required

# Xtrain = np.empty((0,0))
# ytrain = np.empty((0))
# Xtest = np.empty((0,0))
# ytest = np.empty((0))

# logistic = MyLogisticRegression()
# logistic.fit(Xtrain, ytrain)

# ypred = logistic.predict(Xtest)

# print('Predicted Values:', ypred)


# print('True Values:', ytest)
import numpy as np
import pandas as pd
from scratch import *

preprocess = MyPreProcessor()
X,Y=preprocess.pre_process(2)
print(X)
print(Y)

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

L = [0.0001,0.01,10]
for i in range(3):
    print("\n For learning rate ", L[i], "------------------------------------------------------------ ")
    log_regressor_sgd = MyLogisticRegression()
    log_regressor_sgd.fit(X_train, y_train, X_test, y_test, epochs = 2000 ,learning_rate = L[i], grad_type = "sgd")
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