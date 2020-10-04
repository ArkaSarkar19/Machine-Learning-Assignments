from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score
from sklearn import decomposition, datasets
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

def kFoldCrossValidation(model, X, Y, k = 5):
    """ Performs K fold cross validation of the model and the dataset provided, and returns the best model with the fold
    
    Parameters 
    -----------
    model : model to be used for training from the set .
    X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as data
    Y : 1-dimensional numpy array of shape (n_samples,) which acts as labels.
    k : number of folds, default = 5.
    
    Returns
    --------
    optimal_model : best model determined.
    fold : dictionary that contains the best fold.
    """
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
        score = accuracy_score(y_test,y_ped)
        print("For fold : ", i , "/", k)
        print("Accuracy recieved is : ", score , "\n")
        models[i] = (model, score)
        start+=fold_size
        end+=fold_size
    
    best_model = -1
    best_score = 0
    avg_score=0
    for i in range(len(models)):
        avg_score+=models[i][1]
        if(best_score < models[i][1]):
            best_score = models[i][1]
            best_model = i
            
    avg_score = avg_score/k
    
    return models[best_model][0], folds[best_model]


def load_dataset():
    hf = h5py.File('Dataset/part_A_train.h5', 'r')
    X = np.array(hf['X'])
    Y = np.array(hf['Y'])
    print(X.shape,Y.shape)

    """ To calculate the class frequencies """

    print("The class frequencies are : ")

    for i in range(Y.shape[1]):
        freq = np.sum(Y[:,i])
        print("The frequency of class " + str(i) + " is " + str(freq) + " / " + str(Y.shape[0]) )

    y = []
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if(Y[i,j] == 1):
                y.append(j)
    y = np.array(y)
    Y = y.reshape(-1,1)
    Y = np.squeeze(Y)

    return X,Y



if __name__ == "__main__":
    X,Y = load_dataset()
    std_slc = StandardScaler()
    pca = decomposition.PCA()
    dec_tree = tree.DecisionTreeClassifier()
    pipe = Pipeline(steps=[('std_slc', std_slc), ('pca', pca), ('dec_tree', dec_tree)])
    n_components = list(range(1,X.shape[1]+1,1))

    criterion = ['gini', 'entropy']
    max_depth = [2,4,6,8,10,12]

    parameters = dict(pca__n_components=n_components, dec_tree__criterion=criterion, dec_tree__max_depth=max_depth)

    clf_GS = GridSearchCV(pipe, parameters)
    clf_GS.fit(X, y)

    print('Best Criterion:', clf_GS.best_estimator_.get_params()['dec_tree__criterion'])
    print('Best max_depth:', clf_GS.best_estimator_.get_params()['dec_tree__max_depth'])
    print('Best Number Of Components:', clf_GS.best_estimator_.get_params()['pca__n_components'])
    print(); print(clf_GS.best_estimator_.get_params()['dec_tree'])