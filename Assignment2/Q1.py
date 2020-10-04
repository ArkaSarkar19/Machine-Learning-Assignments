import numpy as np 
import pandas as pd 
import os 
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD

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

def run_logistic_regression(X_train, X_test, y_train, y_test, iter = 5000):
	logistic = LogisticRegression(solver = "saga", max_iter  = iter)
	logistic.fit(X_train, y_train)

	test_score = logistic.score(X_test, y_test)
	y_pred = logistic.predict(X_test)
	acc_score = accuracy_score(y_test, y_pred)

	pres_score = precision_score(y_test, y_pred, average='macro')
	f1 = f1_score(y_test, y_pred, average='macro')


	print("Accuracy :",acc_score)
	print("Precision : ",pres_score)
	print("F1 Score : ", f1)


if __name__ ==  "__main__":

	print(" --------------------------------------------------------- PCA -----------------------------------------------------------")
	
	X,Y = load_dataset()
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

	standardScalar = StandardScaler()
	X_train = standardScalar.fit_transform(X_train)
	X_test = standardScalar.transform(X_test)

	pca = PCA(random_state = 123,n_components = 50)
	X_train = pca.fit_transform(X_train)
	X_test = pca.transform(X_test)

	print("Running Logistic Regression 	")
	run_logistic_regression(X_train, X_test, y_train, y_test)

	tsne_em = TSNE(n_components=2, perplexity=30.0, n_iter=1000, verbose=1).fit_transform(X_train)

	df_tsne_pca_2d = pd.DataFrame(columns = ["x", "y", "label"] )
	df_tsne_pca_2d["x"] = tsne_em[:,0]
	df_tsne_pca_2d["y"] = tsne_em[:,1]
	df_tsne_pca_2d["label"] = y_train

	plt.figure(figsize = (10,10))
	# plt.scatter(tsne_em[:,0], tsne_em[:,1],color = "green", hue = y_train);
	sns.scatterplot(x = df_tsne_pca_2d["x"], y = df_tsne_pca_2d["y"], hue = df_tsne_pca_2d["label"], palette = sns.color_palette("husl", 10) )
	plt.show()

	print(" --------------------------------------------------------------------------------------------------------------------")


	print(" --------------------------------------------------------- SVD -----------------------------------------------------------")
	
	X,Y = load_dataset()
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

	standardScalar = StandardScaler()
	X_train = standardScalar.fit_transform(X_train)
	X_test = standardScalar.transform(X_test)

	svd  = TruncatedSVD(random_state = 123,n_components = 50)
	svd.fit(X_train)
	X_train = svd.transform(X_train)
	X_test = svd.transform(X_test)

	print("Running Logistic Regression 	")
	run_logistic_regression(X_train, X_test, y_train, y_test)

	tsne_em = TSNE(n_components=2, perplexity=30.0, n_iter=1000, verbose=1).fit_transform(X_train)

	df_tsne_svd_2d = pd.DataFrame(columns = ["x", "y", "label"] )
	df_tsne_svd_2d["x"] = tsne_em[:,0]
	df_tsne_svd_2d["y"] = tsne_em[:,1]
	df_tsne_svd_2d["label"] = y_train

	plt.figure(figsize = (10,10))
	# plt.scatter(tsne_em[:,0], tsne_em[:,1],color = "green", hue = y_train);
	sns.scatterplot(x = df_tsne_svd_2d["x"], y = df_tsne_svd_2d["y"], hue = df_tsne_svd_2d["label"], palette = sns.color_palette("husl", 10) )
	plt.show()

	print(" --------------------------------------------------------------------------------------------------------------------")