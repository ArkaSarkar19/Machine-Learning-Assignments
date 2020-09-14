import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MyPreProcessor():
    """
    My steps for pre-processing for the three datasets.
    """

    def __init__(self):
        pass

    def pre_process_x(self, X, dataset):
        """
        Preprocessing the input array.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features)

        dataset : integer with acceptable values 0, 1, or 2
        0 -> Abalone Dataset
        1 -> VideoGame Dataset
        2 -> BankNote Authentication Dataset
        
        Returns
        -------
        X_new : 2-dimensional numpy array of shape (n_samples, n_features)
        """

        X_new = np.zeros(X.shape)        

        if dataset == 0:
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
            pass
        elif dataset == 1:
            # Implement for the video game dataset
            pass
        elif dataset == 2:
            # Implement for the banknote authentication dataset
            pass

        return X_new

    def pre_process_y(self, y, dataset):
        """
        Preprocessing the output array.

        Parameters
        ----------
        y : 1-dimensional numpy array of shape (n_samples,)

        dataset : integer with acceptable values 0, 1, or 2
        0 -> Abalone Dataset
        1 -> VideoGame Dataset
        2 -> BankNote Authentication Dataset
        
        Returns
        -------
        y_new : 1-dimensional numpy array of shape (n_samples,)
        """

        y_new = np.zeros(y.shape)

        if dataset == 0:
            # Implement for the abalone dataset
            pass
        elif dataset == 1:
            # Implement for the video game dataset
            pass
        elif dataset == 2:
            # Implement for the banknote authentication dataset
            pass

        return y_new


class MyLinearRegression():
    """
    My implementation of Linear Regression.
    """

    def __init__(self):
        pass

    def fit(self, X, y, X_test = None, y_test = None, learning_rate = 0.005, loss = "rmse"):
        """
        Fitting (training) the linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.
        
        Returns
        -------
        self : an instance of self
        """
        m,n = X.shape           #number of training examples, and features
        W = np.zeros((n,1))     #weights
        b = 0                   #bias

        y = y.reshape((m,1))
        epochs = 10000

        rmse_train_history = []
        mae_train_history = []
        rmse_val_history = []
        mae_val_history = []
        for _ in range(epochs):

            #forward propagation
            y_hat = np.dot(X,W)  + b
            rmse, mae = self.loss(y_hat,y)

            rmse_train_history.append(rmse)
            mae_train_history.append(mae)

            #backward propagation and calculation of gradients
            error = y-y_hat

            if(loss == "rmse"):
                #gradient for rmse loss
                dW = (-1/m)*(np.sum(np.dot(X.T,error),axis = 1)/(np.sum(((1/m)*(error)**2)))**0.5)
                db = (-1/m)*(np.sum(error)/(np.sum(((1/m)*(error)**2))**0.5))
            else:
                epsilon = 10**-7
                dW = (-1/m)*(np.sum(abs(error))/np.sum(error))*np.sum(X.T, axis = 1)
                db = (-1/m)*(np.sum(abs(error))/np.sum(error))

            dW = dW.reshape((n,1))

            # print(dW.shape)
            # print(db.shape)
            W = W - learning_rate*dW
            b = b - learning_rate*db

            self.W = W
            self.b = b

            if(X_test is not None):
                y_pred = self.predict(X_test)
                # y_pred = y_pred.reshape()
                error = y_test - y_pred
                test_n = y_pred.shape[0]
                rmse_val = (np.sum(((error)**2))/test_n)**0.5
                mae_val = np.sum(np.abs(error))/test_n
                rmse_val_history.append(rmse_val)
                mae_val_history.append(mae_val)


                if(_%500 ==0):
                    # print(y_pred.shape)
                    # print(W.shape)
                    print("Training loss after ", _, " iterations is : ", rmse, " | validation loss is : ", rmse_val)
            else:
                if(_%500 ==0):
                    print("Training loss after ", _, " iterations is ", rmse) 

        self.rmse_train_history = rmse_train_history
        self.mae_train_history = mae_train_history
        self.rmse_val_history = rmse_val_history
        self.mae_val_history = mae_val_history
        self.W = W
        self.b = b
        # fit function has to return an instance of itself or else it won't work with test.py
        return self

    def predict(self, X):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        """

        # return the numpy array y which contains the predicted values
        W  = self.W
        b = self.b
        # print(W)
        # print(b)
        y_pred =  np.dot(X,W)  + b


        return y_pred

    def loss(self,y_hat, y):
        """ Calculates Mean Squared Error 

        Parameters
        -----------
        y_hat : numpy array of predicted values
        y : numpy array of ground truth values

        Returns
        -------
        Mean squared error
        """
        m = y.shape[0]
        mae  = abs(y - y_hat)
        rmse = (np.sum(((mae)**2))/m)**0.5
        return rmse, mae





class MyLogisticRegression():
    """
    My implementation of Logistic Regression.
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        """
        Fitting (training) the logistic model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.
        
        Returns
        -------
        self : an instance of self
        """
        m,n = X.shape           #number of training examples, and features
        W = np.zeros((n,1))     #weights
        b = 0                   #bias

        y = y.reshape((m,1))
        epochs = 10000

        rmse_train_history = []
        mae_train_history = []
        rmse_val_history = []
        mae_val_history = []
        for _ in range(epochs):

            #forward propagation
            y_hat = np.dot(X,W)  + b
            rmse, mae = self.loss(y_hat,y)

            rmse_train_history.append(rmse)
            mae_train_history.append(mae)

            #backward propagation and calculation of gradients
            error = y-y_hat
            dW = (-1/m)*(np.sum(np.dot(X.T,error),axis = 1)/np.sum(((1/m)*(error)**2)**0.5))
            db = (-1/m)*(np.sum(error)/np.sum(((1/m)*(error)**2)**0.5))

            dW = dW.reshape((n,1))

            # print(dW.shape)
            # print(db.shape)
            W = W - learning_rate*dW
            b = b - learning_rate*db

            self.W = W
            self.b = b

            if(X_test is not None):
                y_pred = self.predict(X_test)
                # y_pred = y_pred.reshape()
                error = y_test - y_pred
                test_n = y_pred.shape[0]
                rmse_val = np.sum(((error)**2)**0.5)/test_n
                mae_val = np.sum(np.abs(error))/test_n
                rmse_val_history.append(rmse_val)
                mae_val_history.append(mae_val)


                if(_%500 ==0):
                    # print(y_pred.shape)
                    # print(W.shape)
                    print("Training loss after ", _, " iterations is : ", rmse, " | validation loss is : ", rmse_val)
            else:
                if(_%500 ==0):
                    print("Training loss after ", _, " iterations is ", rmse) 

        self.rmse_train_history = rmse_train_history
        self.mae_train_history = mae_train_history
        self.rmse_val_history = rmse_val_history
        self.rmse_val_history = rmse_val_history
        self.W = W
        self.b = b
        # fit function has to return an instance of itself or else it won't work with test.py
        return self

    def predict(self, X):
        """
        Predicting values using the trained logistic model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        """

        # return the numpy array y which contains the predicted values
        W  = self.W
        b = self.b
        y_pred =  sigmoid(np.dot(X,W)  + b)
        for i in range(y_pred.shape[0]):
            if(y_pred[i,1] <0.5):
                y_pred[i,1] = 0
            else:
                y_pred[i,1] = 1

        return y_pred

    def softmax(Z):
        A = 1/(1+np.exp(-Z))
        return A

Xtrain = np.array([[1,2,3], [4,5,6]])
ytrain = np.array([1,2])

Xtest = np.array([[7,8,9],[7,8,9]])
ytest = np.array([3])

print('Linear Regression')

linear = MyLinearRegression()
linear.fit(Xtrain, ytrain)

ypred = linear.predict(Xtest)

print('Predicted Values:', ypred)

print('True Values:', ytest)

print(linear.rmse_train_history)