#! /usr/bin/python
# encoding: utf-8

import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn import tree
from skfeature.function.information_theoretical_based import JMI
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import math

def main():

	# Reading the input data from the source
	data = pd.read_csv("Tic_Toc.csv")

	# Reads the csv file that was given in the input, as a Matrix
	csv = data.as_matrix()

	# Defining number of samples
	X = csv[:,:-1]

	# Defining number of labels
	y = csv[:,-1]

	# Calculating the features' dimensions
	n, m = X.shape 

	# Dividing the dataset into training and testing sets
	ss = cross_validation.KFold(n , n_folds= 10, shuffle = True)
	correct_1 = 0

	# Selecting a decision tree classifier
	clf = tree.DecisionTreeRegressor()

	# Selecting the testing and training data
	for train, test in ss:

		# Selects all the features
		for i in range(75):
			idx = JMI.jmi(X[train], y[train], n_selected_features = m-1)

		# To get the number of features
		features = X[:, idx[0:m-1]]

		# Train the classifier model with the seleced features
		clf.fit(features[train], y[train])

		# Predict the class labels of test data
		y_predict = clf.predict(features[test])
		acc = accuracy_score(y[test], y_predict)
		correct_1 = correct_1 + acc
	print "Accuracy:"+str(correct_1/10)

if __name__ == "__main__":
	main()