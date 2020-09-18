from scratch import MyLinearRegression, MyLogisticRegression, MyPreProcessor
import numpy as np

preprocessor = MyPreProcessor()

print('Linear Regression')

X, y = preprocessor.pre_process(0)

# Create your k-fold splits or train-val-test splits as required

Xtrain = np.empty((0,0))
ytrain = np.empty((0))
Xtest = np.empty((0,0))
ytest = np.empty((0))

linear = MyLinearRegression()
linear.fit(Xtrain, ytrain)

ypred = linear.predict(Xtest)

print('Predicted Values:', ypred)
print('True Values:', ytest)

print('Logistic Regression')

X, y = preprocessor.pre_process(2)

# Create your k-fold splits or train-val-test splits as required

Xtrain = np.empty((0,0))
ytrain = np.empty((0))
Xtest = np.empty((0,0))
ytest = np.empty((0))

logistic = MyLogisticRegression()
logistic.fit(Xtrain, ytrain)

ypred = logistic.predict(Xtest)

print('Predicted Values:', ypred)


print('True Values:', ytest)
