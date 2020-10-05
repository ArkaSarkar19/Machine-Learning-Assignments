from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score
from sklearn import decomposition, datasets
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import ParameterGrid
import h5py


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


class MyGridSearchCV():

    """ Performs Grid search on the estimator and uses k fold cross validation to validate it.

    """

    def __init__(self, estimator, parameters, test_score = "accuracy", cv = 5):

        self.estimator = estimator
        self.parameters = parameters
        self.test_score = test_score
        self.parameterGrid = ParameterGrid(parameters)
        self.cv = cv


    def fit(self, X, y):

        best_estimator_ = None
        best_score = 0
        best_avg = 0
        best_param = -1
        parameterGrid = self.parameterGrid

        self.training_accuracy_history = []
        self.validation_accuracy_history = []
        for i in range(len(parameterGrid)):
            curr_param = parameterGrid[i]
            dec_tree  = self.estimator.set_params(**curr_param)

            print("Current parameters : ", curr_param)
            curr_best_model, curr_avg_score, curr_best_val_score, curr_best_train_score = self.kFoldCrossValidation(X,y, dec_tree)
            self.training_accuracy_history.append(curr_best_train_score)
            self.validation_accuracy_history.append(curr_best_val_score)
            print("Best validation score achieved : ", curr_best_val_score, " Average validation score achieved : ", curr_avg_score, "\n")
            if(curr_best_val_score > best_score):
                best_score = curr_best_val_score
                best_estimator_ = dec_tree
                best_avg = curr_avg_score
                best_param = curr_param

            elif(curr_best_val_score == best_score):
                if(curr_avg_score > best_avg):
                    best_score = curr_best_val_score
                    best_estimator_ = dec_tree
                    best_avg = curr_avg_score
                    best_param = curr_param


        for i in range(len(best_param)):
            attr = list(best_param.keys())[i]
            val = best_param[attr]
            print("Best ", attr , " : ", val)

        print(type(best_estimator_).__name__, "(", best_estimator_.get_params(), ")")



    def kFoldCrossValidation(self, X, y, model):
        """ Performs K fold cross validation of the model and the dataset provided, and returns the best model with the fold
        
        Parameters 
        -----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as data
        Y : 1-dimensional numpy array of shape (n_samples,) which acts as labels.
        k : number of folds, default = 5.
        
        Returns
        --------
        optimal_model : best model determined.
        fold : dictionary that contains the best fold.
        """
        k = self.cv
        m = X.shape[0]  #number of examples
        fold_size = int(m/k)
        start = 0
        end = fold_size
        models = {}
        folds = {}
        for i in range(k):
            Xtrain_i = np.concatenate((X[0:start], X[end+1:]))
            ytrain_i = np.concatenate((y[0:start],y[end+1:]))
            X_test =  X[start:end]
            y_test = y[start:end]
            folds[i] = (start,end+1,Xtrain_i,ytrain_i,X_test,y_test)
            model = model.fit(Xtrain_i, ytrain_i)
            y_pred = model.predict(X_test)
            train_score = accuracy_score(ytrain_i,model.predict(Xtrain_i))
            test_score = accuracy_score(y_test,y_pred)
            models[i] = (model, train_score, test_score)
            start+=fold_size
            end+=fold_size
        
        best_model = -1
        best_val_score = 0
        best_train_score = 0
        avg_val_score=0
        for i in range(len(models)):
            avg_val_score+=models[i][2]
            if(best_val_score < models[i][2]):
                best_val_score = models[i][2]
                best_model = i
            if(best_train_score < models[i][1]):
                best_train_score = models[i][1]
                
        avg_val_score = avg_val_score/k
        
        return models[best_model][0], avg_val_score, best_val_score, best_train_score



# class MyEvaluationMetric():

#     def __init__(self):
#         pass

#     def accuracy_score(y_true, y_pred):
        

if __name__ == "__main__":
    X,Y = load_dataset()
    std_slc = StandardScaler()

    X = std_slc.fit_transform(X)

    dec_tree = tree.DecisionTreeClassifier()

    max_depth = list([i for i in range(1,30,1)])

    parameters = dict( max_depth=max_depth)

    clf_GS = MyGridSearchCV(dec_tree, parameters)
    clf_GS.fit(X, Y)
