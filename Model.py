import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3,random_state= 10)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)    

# Evaluate the model
accuracy = knn.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Save the trained model using joblib
joblib.dump(knn, 'iris_classifier.pkl')
