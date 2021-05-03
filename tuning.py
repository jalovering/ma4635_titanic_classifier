import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import joblib
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# Grid Search uses 5-fold cross validation on X_train to determine best grid
# Best grid parameters are saved and is tested on X_test
def train(X_train, y_train, X_test, classifier, num_pc):
    if classifier == "SVC":
        clf = GridSearchCV(
            estimator = SVC(), param_grid = param_grid[classifier], n_jobs = -1
        )
    elif classifier == 'kNN':
        clf = GridSearchCV(
            estimator = KNeighborsClassifier(), param_grid = param_grid[classifier], n_jobs = -1
        )
    elif classifier == 'LR':
        clf = GridSearchCV(
            estimator = LogisticRegression(), param_grid = param_grid[classifier], n_jobs = -1
    )
    if classifier == "RF":
        clf = GridSearchCV(
            estimator = RandomForestClassifier(), param_grid = param_grid[classifier], n_jobs = -1
    )

    grid_search = clf.fit(X_train, y_train)

    best_grid = grid_search.best_params_
    result = grid_search.predict(X_test)
    return result, best_grid

# PCA Dimensionality Reduction
def pca(X_train, X_test, feature_names, num_pc):
    pca = PCA(n_components=num_pc)
    pca = pca.fit(X_train)

    # generate and save weights as csv 
    pca_weights = pd.DataFrame(pca.components_, columns=feature_names).transpose()
    pca_weights.to_csv('PCA/'+str(num_pc)+'PCs_weights.csv')
    
    # visualize variance coverage by PC
    var = pca.explained_variance_ratio_
    pca_vis(var, num_pc)
    
    X_train = pca.transform(X_train)
    X_train = pd.DataFrame(X_train)
    X_test = pca.transform(X_test)
    X_test = pd.DataFrame(X_test)
    return X_train, X_test

# Generate Variance Coverage Lineplot
def pca_vis(var, num_pc):
    fig = plt.figure()
    plt.plot(var)
    plt.title("Amount of Variance Covered by Each PC", fontsize = 15)
    plt.ylabel("Variance", fontsize = 12)
    plt.xlabel("Principal Component (PC)", fontsize = 12)
    plt.xticks(list(range(0, num_pc, 1)))
    plt.savefig('PCA/'+str(num_pc)+'PCs_var.png', bbox_inches = "tight")
    plt.close()

# Plot Metric Scores for Each Model
def plot_scores(results, num_pc_list):
    g = sns.PairGrid(results, 
                     y_vars=["accuracy", "f1"],
                     x_vars=["num_principal_components"], 
                     hue='classifier', 
                     height=4, aspect=2)
    g.map(sns.lineplot)
    g.fig.suptitle("Model Metrics for Each Number of Principal Components", fontsize=15, y=1.03)
    plt.xlabel("Number of Principal Components", fontsize=12)
    plt.xticks(num_pc_list)
    plt.legend(bbox_to_anchor=(1, 2), loc=2, borderaxespad=0.)
    plt.savefig("model_evaluation/comparison.png", dpi=160, bbox_inches='tight')
    plt.close()

def main():
    # Import Data
    data = pd.read_csv("data/train_clean.csv").sample(frac=1).reset_index(drop=True)
    data = data.iloc[:1000,]
    data = data.drop(columns=["PassengerId", "Name", "Cabin"])
    feature_names = data.drop(columns="Survived").columns
    X = data.drop(columns="Survived").to_numpy() # Predictors
    y = data['Survived'].to_numpy() # Response

    # Normalization
    X_normalized = preprocessing.normalize(X, axis=0)
    X_normalized = pd.DataFrame(X_normalized) # to dataframe

    # Cross-Val Split
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.33)

    # Train and Test Models Using Parameter Grid
    print("Running parameter grid...")
    # run hyperparameter tuning for each number of features
    for num_pc in num_pc_list:

        # dimensionality reduction
        X_train_selected, X_test_selected = pca(X_train, X_test, feature_names, num_pc)

        # tune all models in the parameter grid
        for classifier in param_grid.keys():
            print(str(num_pc) + ' Principal Component(s) |', classifier)
            result, best_grid = train(X_train_selected, y_train, X_test_selected, classifier, num_pc)

            num_pc_tracker.append(num_pc)
            classifiers.append(classifier)
            true.append(y_test)
            predicted.append(result)
            best_grids.append(best_grid)

    print("...Concluded")   

    # Compile Results
    results = pd.DataFrame(list(zip(num_pc_tracker, classifiers, best_grids, true, predicted)), 
                                columns=[
                                    "num_principal_components", 
                                    "classifier", 
                                    "best_grid", 
                                    "true_class", 
                                    "predicted_class",
                                ]
                            )

    # Evaluate Results
    accuracies = []
    f1s = []
    for i, row in results.iterrows(): 
        accuracies.append(accuracy_score(row.true_class, row.predicted_class))
        f1s.append(f1_score(row.true_class, row.predicted_class))
        confusion_matricies.append(confusion_matrix(row.true_class, row.predicted_class).ravel())

    results['accuracy'] = accuracies
    results['f1'] = f1s
    results['tn_fp_fn_tp'] = confusion_matricies
    results = results.drop(columns=["true_class", "predicted_class"])
    results.to_csv("model_evaluation/results.csv", index=False)

    # Plot Results
    plot_scores(results, num_pc_list)

if __name__ == "__main__":
    r = 4635 # random seed
    np.random.seed(r)
    num_pc_list = range(1,5)

    param_grid = {
        "SVC": {
            'kernel': ['linear', 'rbf'], 
            'C': [1, 10, 100, 1000],
        },
        "kNN": {
            "n_neighbors": [2, 3, 5, 7, 9, 11, 15],
            "weights": ["uniform", "distance"],
            "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
            "metric": ["manhattan"]
        },
        "LR": {
            "solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        },
        "RF": {
            "criterion": ['gini', 'entropy'],
            "max_depth": [80, 90, 100, 110],
            "max_features": ["auto", "sqrt", "log2"],
            "n_estimators": [10, 100, 200],
            },
    }
        
    classifiers = []
    true, predicted = [], []
    best_grids = []
    num_pc_tracker = []
    confusion_matricies = []

    main()