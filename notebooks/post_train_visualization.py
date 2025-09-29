#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import recall_score, confusion_matrix
from config import Config as cfg

# In[ ]:


def visualize_classification(x_test, y_test, trained_models_svp, mode):
    if mode == "confusion_matrix":
        plt.figure(figsize=(8, 6))
        for name, model in trained_models_svp.items():
            y_pred = model.predict(x_test)
            print(f"Confusion matrix for {name}")
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted labels')
            plt.ylabel('True labels')
            plt.show()

    if mode == "model_comparison":
        plt.figure(figsize=(12, 10))
        for name, model in trained_models_svp.items():
            y_pred = model.predict(x_test)
            acc_score = recall_score(y_test, y_pred)
            plt.bar(name, acc_score, color='skyblue')
            print(f"Confusion matrix for {name}")

        plt.xlabel('Model name')
        plt.ylabel('Recall')
        plt.title('Models comparison')
        plt.show()


# In[ ]:


def visualize_cluster(x, trained_models_unsvp, n_clusters):
    model = trained_models_unsvp["KMeans"]
    clusters = model.fit_predict(x)
    Xv = x.values if hasattr(x, "values") else x    #safety condition to check whether x has "value" attribute

    plt.figure(figsize=(10, 6))
    colors = ["red","green","blue","yellow","pink","black","orange","purple","beige","brown","gray","cyan","magenta"]  # cluster colors

    for i in range(n_clusters):
        cluster_points = Xv[clusters == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                c=colors[i], label=f'Cluster {i}', s=50)

    plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
            c='black', marker='x', s=200, linewidths=3, label='Centroids')

    plt.title(f'Heart disease clusters (k={n_clusters})')
    plt.xlabel('First Feature')
    plt.ylabel('Second Feature')
    plt.legend()
    plt.show()