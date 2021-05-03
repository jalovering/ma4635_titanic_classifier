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

# PCA Dimensionality Reduction
def pca(X, num_pc):
    pca = PCA(n_components=num_pc)
    pca = pca.fit(X)
    X = pca.transform(X)
    X = pd.DataFrame(X)
    return X

def main():
    # Import Test Data
    data = pd.read_csv("data/test_clean.csv").reset_index(drop=True)
    data = data.iloc[:1000,]
    ids = data["PassengerId"]
    X = data.drop(columns=["PassengerId", "Name", "Cabin"]).to_numpy() # Predictors

    # Normalization
    X_normalized = preprocessing.normalize(X, axis=0)
    X_normalized = pd.DataFrame(X_normalized) # to dataframe

    # dimensionality reduction
    X_selected = pca(X_normalized, 3)

    # make predictions
    clf = joblib.load('trained_models/best_kNN_3pc.pkl')
    prediction = clf.predict(X_selected)

    # Compile Results
    results = pd.DataFrame(list(zip(ids, prediction)), columns=["PassengerId", "Survived"])
    results.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    r = 4635 # random seed
    np.random.seed(r)
    main()