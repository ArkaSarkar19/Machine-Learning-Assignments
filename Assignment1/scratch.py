import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random 

class MyPreProcessor():
    """
    My steps for pre-processing for the three datasets.
    """

    def __init__(self):
        pass

    def pre_process(self, dataset):
        """
        Reading the file and preprocessing the input and output.
        Note that you will encode any string value and/or remove empty entries in this function only.
        Further any pre processing steps have to be performed in this function too. 

        Parameters
        ----------

        dataset : integer with acceptable values 0, 1, or 2
        0 -> Abalone Dataset
        1 -> VideoGame Dataset
        2 -> BankNote Authentication Dataset
        
        Returns
        -------
        X : 2-dimensional numpy array of shape (n_samples, n_features)
        y : 1-dimensional numpy array of shape (n_samples,)
        """

        # np.empty creates an empty array only. You have to replace this with your code.
        X = np.empty((0,0))
        y = np.empty((0))

        if dataset == 0:
            # Implement for the abalone dataset
            df = pd.read_csv("LR_dataset/abalone/Dataset.data",sep="\s+", 
                 skiprows=1,  usecols=[0,1,2,3,4,5,6,7,8], 
                 names=['sex','length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight', 'rings' ])
            df = df.sample(frac = 1)
            train_columns = ['sex','length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight']
            test_columns = ['rings']

            m = X.shape[0]
            df['sex'].replace(to_replace = 'M', value = 1,inplace=True)
            df['sex'].replace(to_replace = 'F', value = 2,inplace=True)
            df['sex'].replace(to_replace = 'I', value = 3,inplace=True)
            X = np.array(df[train_columns])
            y = np.array(df[test_columns])
            X = (X-X.mean())/X.std()

        elif dataset == 1:
            # Implement for the video game dataset
            df = pd.read_csv("LR_dataset/VideoGameDataset - Video_Games_Sales_as_at_22_Dec_2016.csv",  usecols=['Critic_Score','Global_Sales','User_Score'])
            mean_critic_score = df["Critic_Score"].mean()
            df["Critic_Score"] = df["Critic_Score"].fillna(mean_critic_score)
            df['User_Score'].replace(to_replace = 'tbd', value = np.nan ,inplace=True)
            df["User_Score"] = pd.to_numeric(df["User_Score"], downcast="float")
            mean_user_score = df["User_Score"].mean()
            df["User_Score"] = df["User_Score"].fillna(mean_user_score)
            df = df.sample(frac = 1)
            train_columns = ["User_Score", "Critic_Score"]
            X = np.array(df[train_columns])
            Y = np.array(df["Global_Sales"])
            y = Y.reshape((Y.shape[0],1))
            X = (X - X.mean())/X.std()

        elif dataset == 2:
            # Implement for the banknote authentication dataset
            df = pd.read_csv("LoR_dataset/data_banknote_authentication.txt",sep="," , names = ["col1", "col2", "col3", "col4" , "val"])
            df = df.sample(frac = 1)
            train_columns = ["col1", "col2", "col3", "col4"]
            X = np.array(df[train_columns])
            Y = np.array(df["val"])
            X = (X - X.mean())/X.std()
            y = Y.reshape((Y.shape[0],1))


        return X, y
    def get_analytical_sol(self,dataset = 0):
        X,y = self.pre_process(dataset)
        bias = np.zeros((X.shape[0],1))
        bias.fill(1)
        X = np.append(X,bias, axis = 1)
        print(X)
        W = np.dot(np.linalg.inv(np.dot(X.T,X)), np.dot(X.T,y))
        return W




class MyLinearRegression():
    """
    My implementation of Linear Regression.
    """

    def __init__(self):
        pass

    def fit(self, X, y, X_test = None, y_test = None, epochs = 5000,learning_rate = 0.005, loss = "rmse"):
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
        self.W = W
        self.b = b
        y = y.reshape((m,1))
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
            epsilon = 10**-7

            if(loss == "rmse"):
                #gradient for rmse loss
                dW = (-1/m)*(np.sum(np.dot(X.T,error),axis = 1)/(np.sum(((1/m)*(error + 10**-7)**2)))**0.5)
                db = (-1/m)*(np.sum(error)/(np.sum(((1/m)*(error + 10**-7)**2))**0.5))
            else:
                epsilon = 10**-7
                dW = (-1/m)*(np.sum(np.dot(X.T,abs(error)/(error+ epsilon) ), axis = 1))
                db = (-1/m)*(np.sum(abs(error)/(error+ epsilon)))

            dW = dW.reshape((n,1))

            # print(dW.shape)
            # print(db.shape)


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
                    if(loss == "rmse"):
                        print("Training loss after ", _, " iterations is : ", rmse, " | validation loss is : ", rmse_val)
                    else:
                        print("Training loss after ", _, " iterations is : ", mae, " | validation loss is : ", mae_val)
            else:
                if(_%500 ==0):
                    print("Training loss after ", _, " iterations is ", rmse) 

            W = W - learning_rate*dW
            b = b - learning_rate*db

            self.W = W
            self.b = b

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
        return rmse, np.sum(mae)/m





class MyLogisticRegression():
    """
    My implementation of Logistic Regression.
    """

    def __init__(self):
        pass

    def fit(self, X, y, X_test = None, y_test = None, epochs = 5000,learning_rate = 0.005, grad_type = "bgd"):
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

        if(grad_type == "bgd"):
            m,n = X.shape           #number of training examples, and features
            W = np.zeros((n,1))     #weights
            b = 0                   #bias
            self.W = W
            self.b = b;
            y = y.reshape((m,1))

            loss_history = []
            val_loss_history = []
            train_acc_history = []
            val_acc_history = []
            for _ in range(epochs):

                #forward propagation
                Z = np.dot(X,W)  + b
                A = self.sigmoid(Z)
                y_train_pred = self.predict(X)
                train_acc = self.accuracy(y,y_train_pred)
                loss = self.cross_entropy_loss(y,A)
                loss_history.append(loss)
                train_acc_history.append(train_acc)

                #backward propagation and calculation of gradients
                dW = np.dot(X.T,(A-y))/m
                db = np.sum(A-y)/m
                dW = dW.reshape((n,1))

                # print(dW.shape)
                # print(db.shape)

                if(X_test is not None):
                    A_pred = self.sigmoid(np.dot(X_test, W) + b)
                    y_val_pred = self.predict(X_test)
                    val_acc = self.accuracy(y_test,y_val_pred)
                    val_loss = self.cross_entropy_loss(y_test,A_pred)
                    val_loss_history.append(val_loss)
                    val_acc_history.append(val_acc)

                    if(_%500 ==0):
                        # print(y_pred.shape)
                        # print(W.shape)
                        print("\nTraining loss after ", _, " iterations is : ", loss, " | validation loss is : ", val_loss)
                        print("Training accuracy after ", _, " iterations is : ", train_acc*100, "%" ," | validation accuracy is : ", val_acc*100,"%" )
                else:
                    if(_%500 ==0):
                        print("\nTraining loss after ", _, " iterations is ", loss) 
                        print("Training accuracy after ", _, " iterations is : ", train_acc*100,"%" )
                W = W - learning_rate*dW
                b = b - learning_rate*db

                self.W = W
                self.b = b

            self.loss_history = loss_history
            self.val_loss_history = val_loss_history
            self.train_acc_history = train_acc_history
            self.val_acc_history = val_acc_history
            self.W = W
            self.b = b

        else:

            m,n = X.shape           #number of training examples, and features
            print(X.shape, m , n)
            W = np.zeros((n,1))     #weights
            b = 0                   #bias
            self.W = W
            self.b = b;
            y = y.reshape((m,1))

            loss_history = []
            val_loss_history = []
            train_acc_history = []
            val_acc_history = []
            # iterations = 0
            for _ in range(epochs):

                # for i in range(m):
                #forward propagation
                i = random.randint(0,m-1)
                x_i = X[i,:]
                x_i = x_i.reshape((1,n))
                y_i = y[i,:]
                z = np.dot(x_i,W)  + b
                a = self.sigmoid(z)

                #For plotting the loss cure 
                A = self.sigmoid(np.dot(X,W) + b )
                loss = self.cross_entropy_loss(y,A)


                #backward propagation and calculation of gradients
                dW = x_i*(a-y_i)
                db = a-y_i
                dW = dW.reshape((n,1))
                # print(dW.shape)
                # print(db.shape)

                y_train_pred = self.predict(X)
                train_acc = self.accuracy(y,y_train_pred)
                train_acc_history.append(train_acc)
                loss_history.append(loss)
                if(X_test is not None):
                    A_pred = self.sigmoid(np.dot(X_test, W) + b)
                    y_val_pred = self.predict(X_test)
                    val_acc = self.accuracy(y_test,y_val_pred)
                    val_loss = self.cross_entropy_loss(y_test,A_pred) 
                    val_loss_history.append(val_loss)
                    val_acc_history.append(val_acc)
                    if(_%500 ==0):

                        print("\nTraining loss after ", _, " sgd steps is : ", loss, " | validation loss is : ", val_loss)
                        print("Training accuracy after ", _, " sgd steps is : ", train_acc*100, "%" ," | validation accuracy is : ", val_acc*100,"%" )
                else:
                    if(_%500 ==0):
                        print("\nTraining loss after ",_, " sgd steps is : ", loss) 
                        print("Training accuracy after ",_, " sgd steps is : ", train_acc*100,"%" )

                # iterations=iterations + 1
                W = W - learning_rate*dW
                b = b - learning_rate*db

                self.W = W
                self.b = b


        self.loss_history = loss_history
        self.val_loss_history = val_loss_history
        self.train_acc_history = train_acc_history
        self.val_acc_history = val_acc_history
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
        y_pred =  self.sigmoid(np.dot(X,W)  + b)
        y_pred[y_pred < 0.5] = 0
        y_pred[y_pred >= 0.5] = 1
        return y_pred

    def sigmoid(self ,Z):
        """
        Calculate sigmoid

        Parameters 
        ----------
        Z : 2-dimensional numpy array of shape (n_samples, 1) .

        Returns
        -------
        A : 2-dimensional numpy array of shape (n_samples, 1) which acts as activation .
        """
        A = 1/(1+np.exp(-Z))
        return A

    def cross_entropy_loss(self, y, A):
        """
        Calculate Binary cross entropy loss. 

        Parameters 
        ----------
        y : 2-dimensional numpy array of shape (n_samples, 1) which are the ground true labels .
        A : 2-dimensional numpy array of shape (n_samples, 1) which acts as activation .
        Returns
        -------
        loss : Binary cross entropy Loss  
        """
        epsilon = 10**-7
        m = y.shape[0]
        loss = (-1/m)*(np.sum(y*np.log(A + epsilon) + (1-y)*np.log(1-A +  epsilon))) 
        loss = np.squeeze(loss)
        return loss

    def accuracy(self,y, y_hat):
        """
        Calculate accuracy . 

        Parameters 
        ----------
        y : 2-dimensional numpy array of shape (n_samples, 1) which are the ground true labels .
        y_hat : 2-dimensional numpy array of shape (n_samples, 1) which acts as the predicted labels .
        Returns
        -------
        acc : accuracy  
        """

        count = 0
        m = y.shape[0]
        err = np.sum(abs(y - y_hat))/m
        acc = 1 - err
        return acc




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
