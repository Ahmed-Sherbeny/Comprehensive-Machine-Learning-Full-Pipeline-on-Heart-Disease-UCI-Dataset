#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import chi2, SelectKBest


# In[ ]:


#sklearn built-in 'gini' importances.
def feature_importances(feature_names, model):
    importances = model.feature_importances_
    feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Gini Importance': importances}).sort_values('Gini Importance', ascending=False) 
    return feature_imp_df


# In[ ]:


def visualize_importances(feature_imp_df):
    plt.figure(figsize=(8, 4))
    plt.barh(feature_imp_df["Feature"], feature_imp_df["Gini Importance"], color='skyblue')
    plt.xlabel('Gini importances')
    plt.ylabel('Features')
    plt.title('Feature Importance - Gini Importance')
    plt.gca().invert_yaxis()  # Invert y-axis for better visualization
    plt.show()


# In[16]:


def RFE_select(estimator,step, X, y, cv):
    selector = RFECV(estimator=estimator, step = step, cv = cv)
    selector = selector.fit(X, y)
    selected_features_mask = selector.support_
    return selected_features_mask


# In[17]:


def get_selected_feature_names(X, selected_features):
    selected_feature_names = X.columns[selected_features]
    return selected_feature_names


# In[ ]:


def model_selected_features(X_train, X_test, selected_feature_names):
    X_train_selected = X_train[selected_feature_names]
    X_test_selected = X_test[selected_feature_names]
    return X_train_selected, X_test_selected


# In[20]:


def Chi_square_test(X, y):
    selector = SelectKBest(score_func=chi2, k=2)
    selector = selector.fit(X, y)
    feature_scores = selector.scores_
    selected_features = X.columns[selector.get_support()]
    return feature_scores, selected_features

