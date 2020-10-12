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
import sklearn.metrics as met
import matplotlib.pyplot as plt

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
        best_val = 0
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
            if(curr_avg_score > best_score):
                    best_score = curr_avg_score
                    best_estimator_ = dec_tree
                    best_val = curr_best_val_score
                    best_param = curr_param

            elif(curr_avg_score == best_score):
                if(curr_best_val_score > best_val):
                    best_score = curr_avg_score
                    best_estimator_ = dec_tree
                    best_val = curr_best_val_score
                    best_param = curr_param
                # if(curr_avg_score > best_avg):
                #     best_score = curr_best_val_score
                #     best_estimator_ = dec_tree
                #     best_avg = curr_avg_score
                #     best_param = curr_param


        for i in range(len(best_param)):
            attr = list(best_param.keys())[i]
            val = best_param[attr]
            print("Best ", attr , " : ", val)

        print(type(best_estimator_).__name__, "(", best_estimator_.get_params(), ")")

        self.best_estimator_ = best_estimator_




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



class MyEvaluationMetric():

    def __init__(self):
        pass

    def accuracy_score(self,y_true, y_pred):
            m = y_true.shape[0]
            err = 0
            for i in range(m):

                if(y_true[i]!=y_pred[i]):
                    err+=1
            acc = err/m
            return acc

    def confusion_matrix(self,y_true, y_pred):
        y_true = np.squeeze(y_true)
        y_pred = np.squeeze(y_pred)
        n = len(set(y_true))
        cm = np.zeros((n,n),dtype = 'int')
        if(len(y_true)!=len(y_pred)):
            raise "dimension of label arrays don't match"
        for label in set(y_true):
            for i in range(len(y_true)):
                if(label == y_true[i]):
                    if(y_pred[i] == label):
                        cm[label,label]+=1
                    if(y_pred[i]!=label):
                        cm[label,y_pred[i]]+=1


        return cm

    def precision_score(self, y_true, y_pred, average = "micro"):
        cm = self.confusion_matrix(y_true,y_pred)
        n = cm.shape[0]
        TP = []
        FP = []
        class_precisions = []
        for i in range(n):
            tp = cm[i,i]
            fp = np.sum(cm[:,i]) - cm[i,i]
            TP.append(tp)
            FP.append(fp)
            pres_score = tp/(tp + fp)
            class_precisions.append(pres_score)

        if (average == "micro"):
            precision_score = np.sum(TP)/ (np.sum(TP) + np.sum(FP))
        elif(average == "macro"):
            precision_score = np.sum(class_precisions)/ (len(class_precisions))
        else:
            raise "Wrong average"
        return precision_score

    def recall_score(self, y_true, y_pred, average = "micro"):
        cm = self.confusion_matrix(y_true,y_pred)
        n = cm.shape[0]
        TP = []
        FN = []
        class_recalls = []
        for i in range(n):
            tp = cm[i,i]
            fn = np.sum(cm[i,:]) - cm[i,i]
            TP.append(tp)
            FN.append(fn)
            rec_score = tp/(tp + fn)
            class_recalls.append(rec_score)

        if (average == "micro"):
            recall_score = np.sum(TP)/ (np.sum(TP) + np.sum(FN))
        elif(average == "macro"):
            recall_score = np.sum(class_recalls)/ (len(class_recalls))
        else:
            raise "Wrong average"
        return recall_score

    def f1_score(self,y_true,y_pred,average = "micro"):

        precision = self.precision_score( y_true, y_pred, average)
        recall = self.recall_score( y_true, y_pred, average)

        f1_score = 2*precision*recall/(precision + recall)

        return f1_score

    def calculate_rates(self,y_true, predictions, threshold = 0.5, average = "micro"):
        y_true = np.squeeze(y_true)
        y_pred = np.squeeze(predictions)
        n = len(set(y_true))
        if(len(y_true)!=len(predictions)):
            raise "dimension of label arrays don't match"
        TP = []
        FP = []
        TN = []
        FN = []
        TPR = []
        FPR = []

        for label in set(y_true):
            tp=0
            fp=0
            tn=0
            fn=0
            for i in range(len(y_true)):
                if(label == y_true[i]):
                    if(predictions[i] >=threshold):
                        tp+=1
                    else:
                        fn+=1

                else:
                    if(predictions[i] >=threshold):
                        fp+=1
                    else:
                        tn+=1
            TP.append(tp)
            FN.append(fn)
            FP.append(fp)
            TN.append(tn)
            tpr = tp/(tp+fn)
            fpr = fp/(fp+tn)
            TPR.append(tpr)
            FPR.append(FPR)


        TP = np.array(TP)
        FN = np.array(FN)
        FP = np.array(FP)
        TN = np.array(TN)
        if (average == "micro"):
            tpr_fin = np.sum(TP)/ (np.sum(TP) + np.sum(FN))
            fpr_fin = np.sum(FP)/ (np.sum(FP) + np.sum(TN))
        elif(average == "macro"):
            tpr_fin = np.sum(TPR)/ len(TPR)
            fpr_fin = np.sum(FPR)/ len(FPR)
        else:
            raise "Wrong average"
        return tpr_fin,fpr_fin


    def plot_roc_curve(self,y_true,predictions, average = "macro"):
        TPR = []
        FPR = []
        thresholds = []
        n_classes = predictions.shape[1]

        threshold = 0
        inc = 1/100
        while(threshold<=1):
            TPR_i = []
            FPR_i = []
            for _ in range(n_classes):
                predictions_i = np.squeeze(predictions[:,_])
                tpr_fin,fpr_fin = self.calculate_rates(y_true,predictions_i,threshold,average)
                TPR_i.append(tpr_fin)
                FPR_i.append(fpr_fin)
            thresholds.append(threshold)
            TPR.append(TPR_i)
            FPR.append(FPR_i)
            threshold+=inc

        TPR = np.array(TPR)
        FPR = np.array(FPR)
        print(TPR)
        print(FPR)
        for i in range(n_classes):
            plt.plot(FPR[:,i],TPR[:,i]);
            plt.plot(thresholds,thresholds,'--')
            plt.xlabel("False Positive Rate (FPR)--->")
            plt.ylabel("True Positive Rate (TPR)--->")
            plt.title("ROC Curve")
            plt.show()



        

if __name__ == "__main__":
    # X,Y = load_dataset()
    # std_slc = StandardScaler()

    # X = std_slc.fit_transform(X)

    # dec_tree = tree.DecisionTreeClassifier()

    # max_depth = list([i for i in range(1,30,1)])

    # parameters = dict( max_depth=max_depth)

    # clf_GS = MyGridSearchCV(dec_tree, parameters)
    # clf_GS.fit(X, Y)
    # import matplotlib.pyplot as plt 
    # plt.plot([x for x in range(1,30,1)], clf_GS.training_accuracy_history, label = "Training  accuracy " )
    # plt.plot([x for x in range(1,30,1)], clf_GS.validation_accuracy_history, label = "Validation  accuracy")
    # plt.xlabel('max_depth')
    # plt.ylabel('accuracy')
    # plt.legend()
    # plt.show()






    # metrics = MyEvaluationMetric()
    # y_true = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2,3,1,3,3,2,0,2,1,0,3,1,1,1,2,0,0]
    # y_pred = [0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2,3,1,3,3,2,0,1,2,0,3,1,2,3,2,1,1]
    # # y_true = [1,1,1,1,1,0,0,0,1,1,0,1,0]
    # # y_pred = [1,1,0,1,0,0,0,1,1,1,0,1,1]
    # cm_1 = met.confusion_matrix(y_true, y_pred)
    # pres_score_micro = met.precision_score(y_true,y_pred,average = "micro")
    # pres_score_macro = met.precision_score(y_true,y_pred,average = "macro")
    # recall_score_micro = met.recall_score(y_true,y_pred,average = "micro")
    # recall_score_macro = met.recall_score(y_true,y_pred,average = "macro")
    # f1_score_micro = met.f1_score(y_true,y_pred,average = "micro")
    # f1_score_macro = met.f1_score(y_true,y_pred,average = "macro")
    # print(cm_1,pres_score_micro,pres_score_macro,recall_score_micro,recall_score_macro,f1_score_micro,f1_score_macro)
    # print("\n\n")
    # cm_2 = metrics.confusion_matrix(y_true  , y_pred)
    # pres_score_micro = metrics.precision_score(y_true,y_pred,average = "micro")
    # pres_score_macro = metrics.precision_score(y_true,y_pred,average = "macro")
    # recall_score_micro = metrics.recall_score(y_true,y_pred,average = "micro")
    # recall_score_macro = metrics.recall_score(y_true,y_pred,average = "macro")
    # f1_score_micro = metrics.f1_score(y_true,y_pred,average = "micro")
    # f1_score_macro = metrics.f1_score(y_true,y_pred,average = "macro")
    # print(cm_2,pres_score_micro,pres_score_macro,recall_score_micro,recall_score_macro,f1_score_micro,f1_score_macro)
    # metrics.plot_roc_curve(y_true,y_pred)


    X,y = load_dataset(0)
    # std_slc = StandardScaler()
    # X = std_slc.fit_transform(X)
    dec_tree = tree.DecisionTreeClassifier(max_depth = 10)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)
    dec_tree.fit(X_train, y_train)
    prob = dec_tree.predict_proba(X_test)
    print(prob)
    metric = MyEvaluationMetric()
    metric.plot_roc_curve(y_test, prob)
