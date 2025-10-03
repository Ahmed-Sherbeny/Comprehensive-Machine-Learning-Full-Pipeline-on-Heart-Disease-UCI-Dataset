#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import joblib
import os

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import silhouette_score


# In[ ]:

def silhouette_scorer(estimator, X):
    labels = estimator.fit_predict(X)
    # silhouette_score requires at least 2 labels; guard against degenerate cases
    if len(set(labels)) < 2:
        return -1.0
    return silhouette_score(X, labels)

def grid_search(X, y=None, estimator=None, param_grid=None, scoring=None, **kwargs):
    clf = GridSearchCV(estimator = estimator, param_grid= param_grid,scoring = scoring, **kwargs)
    if y is not None:
        clf.fit(X, y)
    else:
        clf.fit(X)   # unsupervised
    return clf


# In[ ]:


def random_search(X, y = None, estimator=None, param_distributions=None, random_state=None, scoring=None, **kwargs):
    clf = RandomizedSearchCV(estimator = estimator, param_distributions = param_distributions, random_state=random_state, scoring=scoring, **kwargs)
    if y is not None:
        clf.fit(X, y)
    else:
        clf.fit(X)   # unsupervised
    return clf


# In[ ]:


def save_model(model, path):
    joblib.dump(model, path)
    print(f"Model has been saved")


# In[ ]:


def load_model(path):
    if os.path.exists(path):
        print("Model has been loaded")
        return joblib.load(path)
    else:
        raise FileNotFoundError(f"No model found at {path}")