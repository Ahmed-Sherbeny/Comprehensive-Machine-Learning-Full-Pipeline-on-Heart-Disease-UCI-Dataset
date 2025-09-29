#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# In[65]:


def pca_analysis(x, n_components):
    pca = PCA(n_components = n_components)
    x_pca = pca.fit_transform(x)
    return pca, x_pca


# In[ ]:


def optimal_num_of_components(pca):
    explained_var = pca.explained_variance_ratio_
    cum_explained_var = explained_var.cumsum()
    n_components_90 = np.argmax(cum_explained_var >= 0.90) + 1
    print("Optimal number of components for (>=90% variance):", n_components_90)


# In[ ]:


def pca_visualization(X_scaled, y):
    fig, axes = plt.subplots(1, 2, figsize=(12, 10))
    pca, x_pca = pca_analysis(X_scaled, n_components=2)
    sc = axes[0].scatter(x_pca[:, 0], x_pca[:, 1], c = y.values.ravel(), cmap = 'viridis')
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].set_title("PCA (2 components)")
    fig.colorbar(sc, ax=axes[0], label="Class")

    pca, _ = pca_analysis(X_scaled, n_components=None)
    cum = np.cumsum(pca.explained_variance_ratio_)
    axes[1].plot(np.arange(1, len(cum)+1), cum, marker='o')
    axes[1].axhline(0.90, ls='--', label='90% threshold')
    axes[1].set_xlabel("Number of Principal Components")
    axes[1].set_ylabel("Cumulative Explained Variance")
    axes[1].set_title("Cumulative Explained Variance VS Number of Principal Components")
    axes[1].legend()
    plt.show()

