from dataclasses import dataclass
from scipy.stats import loguniform, randint, uniform 

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

@dataclass(frozen= True)
class Config:
    """Constants"""
    MODEL_PATH = "./models/heart_disease_model.pkl"
    MODEL_PATH_B = "./models/heart_disease_binarized_model.pkl"
    RESULTS_PATH = "./results/evaluation_metrics.txt"
    RANDOM_STATE = 42
    DATASET_ID = 45

    MISSIING_VALUES_COLUMNS = ["ca", "thal"]
    PRE_TRAIN_VISUALIZE_MODE = "histogram"
    N_PCA_COMPONENTS = 2

    #N_FEATURES_TO_SELECT = 8
    STEP = 1

    DATASET_NAME = "UCI_Heart_Disease_Dataset"

    CV = 5

    POST_TRAIN_VISUALIZE_MODE = "confusion_matrix"
    N_CLUSTERS = 2
    #==================== GRIDSEARCH PARAMATERS FOR SVP MODELS ====================#
    PARAM_GRID_SVP = {
        "LogisticRegression": {
            "model__solver": ["lbfgs", 'liblinear'],
            "model__C": [0.01, 0.1, 1, 10, 100]
        },
        "SVC": {
            "model__kernel": ["rbf", "poly", "linear"],
            "model__C": [0.01, 0.1, 1, 10, 100],
            "model__gamma": ["scale", "auto"]
        },
        "DecisionTree": {
            "model__criterion": ["gini", "entropy", "log_loss"],
            "model__max_depth": [None, 3, 5, 8, 12],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4]
        },
        "RandomForest": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 5, 8],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2"]
        }
    }

    #==================== RANDOMSEARCH PARAMATERS FOR SVP MODELS ====================#
    PARAM_DISTRIBUTIONS_SVP = {
        "LogisticRegression": {
            "model__C": uniform(0.001, 100),
            "model__solver": ["lbfgs", 'liblinear']
        },
        "SVC": {
            "model__kernel": ["rbf", "poly", "linear"],
            "model__C": loguniform(1e-3, 1e2),
            "model__gamma": ["scale", "auto"]
        },
        "DecisionTree": {
            "model__criterion": ["gini", "entropy", "log_loss"],
            "model__max_depth": randint(3, 30),
            "model__min_samples_split": randint(2, 15),
            "model__min_samples_leaf": randint(1, 6)
        },
        "RandomForest": {
            "model__n_estimators": randint(100, 1200),
            "model__max_depth": randint(5, 40),
            "model__min_samples_split": randint(2, 15),
            "model__min_samples_leaf": randint(1, 6),
            "model__max_features": ["sqrt", "log2"]
        }
    }

    #==================== GRIDSEARCH PARAMATERS FOR UNSVP MODELS ====================#
    PARAM_GRID_UNSVP = {
        "KMeans": {
            "model__n_clusters": [2, 3, 4, 5, 6, 7, 8, 10],
            "model__init": ["k-means++", "random"],
            "model__max_iter": [300, 500]
        },
        # For Agglomerative, ward requires metric='euclidean'; others can vary
        "AgglomerativeClustering": [
            {
                "model__n_clusters": [2, 3, 4, 5, 6, 8, 10],
                "model__linkage": ["ward"],
                "model__metric": ["euclidean"]
            },
            {
                "model__n_clusters": [2, 3, 4, 5, 6, 8, 10],
                "model__linkage": ["complete", "average", "single"],
                "model__metric": ["euclidean", "manhattan", "cosine"]
            }
        ]
    }

    #==================== RANDOM SEARCH PARAMETERS FOR UNSVP MODELS ====================#
    
    PARAM_DISTRIBUTIONS_UNSVP = {
        "KMeans": {
            "model__n_clusters": randint(2, 20),
            "model__init": ["k-means++", "random"],
            "model__max_iter": randint(100, 1000)
        },
        "AgglomerativeClustering": {
            "model__n_clusters": randint(2, 20),
            "model__linkage": ["ward", "complete", "average", "single"],
            "model__metric": ["euclidean", "manhattan", "cosine"]  # will be ignored if linkage="ward" (euclidean enforced)
        }
    }

    #==================== BASELINE MODELS ====================#
    # CLASSIFICATION_MODELS = {
    #     "LogisticRegression": LogisticRegression(solver= 'liblinear'),
    #     "SVC": SVC(kernel="rbf", C=1.0),
    #     "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE , max_depth=5),
    #     "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    # }

    # CLUSTERING_MODELS = {
    #     "KMeans": KMeans(n_clusters=13, random_state=42),
    #     "AgglomerativeClustering": AgglomerativeClustering(n_clusters=4, linkage='ward')
    # }

    #==================== OPTIMIZED MODELS ====================#
    CLASSIFICATION_MODELS = {
        "LogisticRegression": LogisticRegression(solver= 'lbfgs', C = 1, multi_class='multinomial', max_iter=500),
        "SVC": SVC(kernel="rbf", C=10.0, gamma= 'auto'),
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE , max_depth=3, min_samples_leaf=1, min_samples_split = 2),
        "RandomForest": RandomForestClassifier(max_depth = 5, max_features= 'log2', min_samples_leaf= 4, min_samples_split= 10, n_estimators= 100, random_state=RANDOM_STATE)
    }

    CLUSTERING_MODELS = {
        "KMeans": KMeans(init= 'random', max_iter= 300, n_clusters= 2, random_state=RANDOM_STATE),
        "AgglomerativeClustering": AgglomerativeClustering(linkage= 'average', metric= 'manhattan', n_clusters= 2)
    }

    ESTIMATOR = CLASSIFICATION_MODELS["DecisionTree"]    #Merely a sample model to select features and see feature importances 