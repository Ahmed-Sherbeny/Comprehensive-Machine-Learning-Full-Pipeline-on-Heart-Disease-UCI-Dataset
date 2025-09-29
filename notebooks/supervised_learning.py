#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.metrics import classification_report, roc_curve, roc_auc_score, recall_score
from config import Config as cfg
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.base import clone  

# In[ ]:


def train_model_svp(x_train, y_train):
    trained_models = {}
    for name, model in cfg.CLASSIFICATION_MODELS.items():
        print(f"Training {name}")
        trained_models[name] = clone(model).fit(x_train, y_train)

    print("SVP models have been trained")
    return trained_models


# In[ ]:
def _score_matrix(model, X, classes):
    """
    Returns a 2D array of scores with shape (n_samples, n_classes) when possible:
    - predict_proba -> probabilities per class
    - decision_function -> per-class scores if available; 1D is reshaped to (n_samples, 1)
    Returns None if neither method exists.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 1:
            proba = proba[:, None]
        return proba

    if hasattr(model, "decision_function"):
        df = model.decision_function(X)
        if df.ndim == 1:
            df = df[:, None]
        return df

    return None

#This function was refined using an LLM due to decision_function, predict_proba missing attributes logic
def evaluate_svp_models(trained_models_svp, X, y, dataset_name):
    print("\n" + "=" * 60)
    print(f"Evaluating models on {dataset_name} dataset:")
    print("=" * 60)

    scores = {}
    recall_scores = {}  # I don't need this in theory since classification report
                        # already provides recall but i need it in the run_pipeline 
                        # function to select best model based on it
    ROC_scores = {}
    AUC_scores = {}

    y = np.asarray(y).ravel()
    classes = np.unique(y)
    n_classes = len(classes)
    y_bin = label_binarize(y, classes=classes) if n_classes > 2 else y

    for name, model in trained_models_svp.items():
        # --- classification report & recall ---
        y_pred = model.predict(X)
        report = classification_report(y, y_pred, zero_division=0)
        scores[name] = report

        if n_classes > 2:
            recall_scores[name] = recall_score(y, y_pred, average="macro", zero_division=0) #since it's a multiclass. Macro is preferred 
                                                                                            #in this case to not miss false negatives
        else:
            recall_scores[name] = recall_score(y, y_pred, zero_division=0)  #default is 'binary'

        # --- ROC/AUC (skip if we don't have usable continuous scores) ---
        S = _score_matrix(model, X, classes)

        if S is None:
            print(f"- {name}: no predict_proba/decision_function -> skipping ROC/AUC.")
            continue

        if n_classes == 2:
            # pick the column corresponding to the positive class
            if S.shape[1] == 1:
                s_pos = S[:, 0]
                pos_label = classes[1] if len(classes) == 2 else classes[-1]
            else:
                pos_label = classes[1] if len(classes) == 2 else classes[-1]
                pos_col = np.where(classes == pos_label)[0][0]
                s_pos = S[:, pos_col]

            fpr, tpr, thr = roc_curve(y, s_pos, pos_label=pos_label)
            auc_val = roc_auc_score(y, s_pos)
            ROC_scores[name] = [fpr, tpr, thr]
            AUC_scores[name] = auc_val
        else:
            # one-vs-rest per-class ROC
            if S.shape[1] != n_classes:
                print(f"- {name}: score shape {S.shape} != n_classes={n_classes}; skipping ROC/AUC.")
                continue
            per_class_roc = {}
            for i, cls in enumerate(classes):
                fpr, tpr, thr = roc_curve(y_bin[:, i], S[:, i])
                per_class_roc[int(cls)] = (fpr, tpr, thr)
            ROC_scores[name] = per_class_roc
            AUC_scores[name] = roc_auc_score(y_bin, S, average="macro", multi_class="ovr")

        print(f"Classification report for {name}:\n {report}\n")
        print(f"- {name}: recall(macro average if multi-class) = {recall_scores[name]:.3f}" +
              (f", AUC = {AUC_scores.get(name, np.nan):.3f}\n" if name in AUC_scores else ""))

    print("=" * 60)
    return scores, recall_scores, ROC_scores, AUC_scores

