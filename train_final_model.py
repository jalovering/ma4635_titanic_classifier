import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import joblib
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# Train and save model on best parameters
def train(X, y, classifier, params, num_pc):
    if classifier == "SVC":
        clf = SVC(**params)
    elif classifier == 'kNN':
        clf = KNeighborsClassifier(**params)
    elif classifier == 'LR':
        clf = LogisticRegression(**params)
    if classifier == "RF":
        clf = RandomForestClassifier(**params)

    model = clf.fit(X, y) # train model
    joblib.dump(model, 'trained_models/best_' + classifier + '_' + str(num_pc) + 'pc.pkl', compress = 1)

# PCA Dimensionality Reduction
def pca(X, num_pc):
    pca = PCA(n_components=num_pc)
    pca = pca.fit(X)
    X = pca.transform(X)
    X = pd.DataFrame(X)
    return X

def main():
    # Import Data
    data = pd.read_csv("data/train_clean.csv").sample(frac=1).reset_index(drop=True)
    # data = data.iloc[:1000,]
    data = data.drop(columns=["PassengerId", "Name", "Cabin"])
    X = data.drop(columns="Survived").to_numpy() # Predictors
    y = data['Survived'].to_numpy() # Response

    # Normalization
    X_normalized = preprocessing.normalize(X, axis=0)
    X_normalized = pd.DataFrame(X_normalized) # to dataframe

    # dimensionality reduction
    X_selected = pca(X_normalized, num_pc)

    # Train and Save Final Model
    train(X_selected, y, classifier, params, num_pc)

if __name__ == "__main__":
    r = 4635 # random seed
    np.random.seed(r)

    # Pick Best Parameters
    # classifier = 'kNN'
    # params = {'algorithm': 'auto', 'metric': 'manhattan', 'n_neighbors': 15, 'weights': 'uniform'}
    # num_pc = 3
    classifier = 'SVC'
    params = {'C': 1000, 'kernel': 'rbf'}
    num_pc = 3
    
    main()