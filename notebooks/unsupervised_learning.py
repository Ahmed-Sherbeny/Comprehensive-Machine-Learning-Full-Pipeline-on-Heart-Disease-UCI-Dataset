#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.metrics import adjusted_rand_score
from config import Config as cfg
from sklearn.base import clone 

# In[ ]:


def train_model_unsvp(x_train):
    trained_models = {}
    for name, model in cfg.CLUSTERING_MODELS.items():
        print(f"Training {name}")
        trained_models[name] = clone(model).fit(x_train)

    print("UNSVP model have been trained")
    return trained_models


# In[ ]:


def evaluate_unsvp_models(trained_models_unsvp, X, y, dataset_name):

    ari_scores = {}
    for name, model in trained_models_unsvp.items():
        labels = model.fit_predict(X)
        ari = adjusted_rand_score(y, labels)
        ari_scores[name] = ari
        print(f"Adjusted random index for {name}: {(ari)}\n")

    print(f"{'='*60}")
    return ari_scores

