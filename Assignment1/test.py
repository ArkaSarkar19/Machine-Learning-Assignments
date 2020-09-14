from scratch import MyLinearRegression, MyLogisticRegression, MyPreProcessor
import numpy as np

preprocessor = MyPreProcessor()

Xtrain = np.array([[1,2,3], [4,5,6]])
Xtrain = preprocessor.pre_process_x(Xtrain, 0)
ytrain = np.array([1,2])
ytrain = preprocessor.pre_process_y(ytrain, 0)

Xtest = np.array([[7,8,9]])
Xtest = preprocessor.pre_process_x(Xtest, 0)
ytest = np.array([3])
ytest = preprocessor.pre_process_y(ytest, 0)

print('Linear Regression')

linear = MyLinearRegression()
linear.fit(Xtrain, ytrain)

ypred = linear.predict(Xtest)

print('Predicted Values:', ypred)
print('True Values:', ytest)

Xtrain = np.array([[1,2,3], [4,5,6]])
Xtrain = preprocessor.pre_process_x(Xtrain, 2)
ytrain = np.array([1,2])
ytrain = preprocessor.pre_process_y(ytrain, 2)

Xtest = np.array([[7,8,9]])
Xtest = preprocessor.pre_process_x(Xtest, 2)
ytest = np.array([3])
ytest = preprocessor.pre_process_y(ytest, 2)

print('Logistic Regression')

logistic = MyLogisticRegression()
logistic.fit(Xtrain, ytrain)

ypred = logistic.predict(Xtest)

print('Predicted Values:', ypred)
print('True Values:', ytest)