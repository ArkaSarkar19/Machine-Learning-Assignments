import numpy as np 
import numpy as np
import scipy as sp 
import pandas as pd
import h5py as h5py
import math

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
            if(k%10 == 0):
                print("Total samples done : ", k)
            y_pred.append(np.argmax(probabilities))
        
        return y_pred
        