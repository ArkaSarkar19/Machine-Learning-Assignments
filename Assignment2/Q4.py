import numpy as np
import scipy as sp 
import pandas as pd
import h5py as h5py
import math
from sklearn.model_selection import train_test_split
import sklearn.metrics as met
from tabulate import tabulate
from sklearn.naive_bayes import GaussianNB
def load_dataset(dataset = 0):

    """ Returns the shuffled dataset from the .h5 file and calculates the class frequences. 

    Parameters
    ----------
    dataset : Integer to denoted the type of dataset to be loaded, default = 0

    Returns
    --------
    X :  2-dimensional numpy array of shape (n_samples, n_features) which acts as data.
    Y : 1-dimensional numpy array of shape (n_samples,) which acts as labels.

    """
    ## Dataset A
    if (dataset == 0):
        hf = h5py.File('Dataset/part_A_train.h5', 'r') #read the dataset 
        
        X = np.array(hf['X']) # X data  
        Y = np.array(hf['Y']) # class labels of the form [0,0,0,1,0,,..]
        print(X.shape,Y.shape)

        """ To calculate the class frequencies """

        print("The class frequencies are : ")

        for i in range(Y.shape[1]):
            freq = np.sum(Y[:,i])
            print("The frequency of class " + str(i) + " is " + str(freq) + " / " + str(Y.shape[0]) )

        """ Converting the binary class labels into single valued blabels """
        y = []
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                if(Y[i,j] == 1):
                    y.append(j)
        
        y = np.array(y)
        Y = y.reshape(-1,1)
        Y = np.squeeze(Y)

        """ Shuffling the dataset """
        np.random.seed(123)
        index = np.random.permutation(X.shape[0])
        np.take(X, index, axis = 0, out = X)
        np.take(Y, index, axis = 0, out = Y)

    ## dataset B
    elif(dataset == 1):

        hf = h5py.File('Dataset/part_B_train.h5', 'r') #read the dataset 

        X = np.array(hf['X']) # X data  
        Y = np.array(hf['Y']) # class labels of the form [0,0,0,1,0,,..]
        print(X.shape,Y.shape)

        """ To calculate the class frequencies """

        print("The class frequencies are : ")

        for i in range(Y.shape[1]):
            freq = np.sum(Y[:,i])
            print("The frequency of class " + str(i) + " is " + str(freq) + " / " + str(Y.shape[0]) )

        """ Converting the binary class labels into single valued blabels """
        y = []
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                if(Y[i,j] == 1):
                    y.append(j)

        y = np.array(y)
        Y = y.reshape(-1,1)
        Y = np.squeeze(Y)

        """ Shuffling the dataset """
        np.random.seed(123)
        index = np.random.permutation(X.shape[0])
        np.take(X, index, axis = 0, out = X)
        np.take(Y, index, axis = 0, out = Y)

    return X,Y


class MyGaussianNaiveBayes():
    
    def __init__(self):
        pass
    
    def fit(self, X,y):
        self.X = X
        self.y = y
        self.num_classes = np.unique(y) #number of unique classes in y
        model_parameters = {} #mean and variance of each feature for each class
        
        for idx,class_i in enumerate(self.num_classes):
            X_c = X[np.where(y==class_i)] #all training samples when label = class_i
            
            class_parameters = [] #storing mean and variance of each feature of class_i
            for i in range(X_c.shape[1]):
                feature_mean = X_c[:,i].mean()
                feature_var = X_c[:,i].var()
                class_parameters.append({"mean" : feature_mean, "var" : feature_var})
            model_parameters[class_i] = class_parameters
        self.model_parameters = model_parameters
    
    
    def predict(self, X):
        
        y_pred = []
        
        for k in range(X.shape[0]):
            sample = X[k,:]
            probabilities = []
            for idx,class_i in enumerate(self.num_classes):

                p = np.log(list(self.y).count(class_i)/len(self.y))
                feature_list = self.model_parameters[class_i]
                epsilon = 10**-4
                for i in range(len(feature_list)):
                    curr_feature = feature_list[i]
                    curr_mean = curr_feature["mean"]
                    curr_var = curr_feature["var"]
                    curr_x = sample[i]
                    if(curr_var <= 10**-7):
                        continue
                    gaussian_estimate_c = 1/math.sqrt(2*math.pi*curr_var + epsilon)
                    gaussian_estimate_exp = math.exp((-(curr_x - curr_mean)**2/(2*curr_var + epsilon)))
                    gauss = gaussian_estimate_c*gaussian_estimate_exp
                    p+=np.log(gauss)

                probabilities.append(p)
            y_pred.append(np.argmax(probabilities))
        
        return y_pred
        

if __name__ == "__main__":

	print("--------------------Dataset A----------------------------------------------------")
	X,y = load_dataset(0)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
	print("Running MyGaussianNaiveBayes---------------")
	nb = MyGaussianNaiveBayes()
	nb.fit(X_train,y_train)
	y_pred = nb.predict(X_test)
	accuracy = met.accuracy_score(y_test,y_pred)
	precision = met.precision_score(y_test,y_pred, average = "macro")
	recall = met.recall_score(y_test,y_pred,average = "macro")
	f1 = met.f1_score(y_test,y_pred, average = "macro")
	l = [[ accuracy, precision, recall, f1]]
	table = tabulate(l, headers=["Accuracy", "Precision ", " Recall", "f1"], tablefmt='orgtbl')
	print("\n")
	print(table)


	print("Running SklearnGaussianNaiveBayes---------------")
	gnb = GaussianNB()
	y_pred = gnb.fit(X_train, y_train).predict(X_test)
	accuracy = met.accuracy_score(y_test,y_pred)
	precision = met.precision_score(y_test,y_pred, average = "macro")
	recall = met.recall_score(y_test,y_pred,average = "macro")
	f1 = met.f1_score(y_test,y_pred, average = "macro")
	l = [[ accuracy, precision, recall, f1]]
	table = tabulate(l, headers=["Accuracy", "Precision ", " Recall", "f1"], tablefmt='orgtbl')
	print("\n")
	print(table)

	print("--------------------Dataset B----------------------------------------------------")


	X,y = load_dataset(1)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

	print("Running MyGaussianNaiveBayes---------------")

	nb = MyGaussianNaiveBayes()
	nb.fit(X_train,y_train)
	y_pred = nb.predict(X_test)

	accuracy = met.accuracy_score(y_test,y_pred)
	precision = met.precision_score(y_test,y_pred, average = "macro")
	recall = met.recall_score(y_test,y_pred,average = "macro")
	f1 = met.f1_score(y_test,y_pred, average = "macro")
	l = [[ accuracy, precision, recall, f1]]
	table = tabulate(l, headers=["Accuracy", "Precision ", " Recall", "f1"], tablefmt='orgtbl')
	print("\n")
	print(table)

	print("Running SklearnGaussianNaiveBayes---------------")

	gnb = GaussianNB()
	y_pred = gnb.fit(X_train, y_train).predict(X_test)
	accuracy = met.accuracy_score(y_test,y_pred)
	precision = met.precision_score(y_test,y_pred, average = "macro")
	recall = met.recall_score(y_test,y_pred,average = "macro")
	f1 = met.f1_score(y_test,y_pred, average = "macro")
	l = [[ accuracy, precision, recall, f1]]
	table = tabulate(l, headers=["Accuracy", "Precision ", " Recall", "f1"], tablefmt='orgtbl')
	print("\n")
	print(table)