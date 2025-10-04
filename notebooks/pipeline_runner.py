#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import data_preprocessing as dp
import pca_analysis as pc
import feature_selection as fs
import supervised_learning as svp
import unsupervised_learning as unsvp
import hyperparameter_tuning as hyp
import post_train_visualization as ptv

from .config import Config
from importlib import reload
from sklearn.pipeline import Pipeline
from sklearn.base import clone      #To avoid models overwriting each other while fitting
import warnings
from sklearn.metrics import classification_report
import os

# In[ ]:

reload(dp)
reload(pc)
reload(fs)
reload(svp)         # These reloads are to ignore previous runs as they are cached in a __pycache__ file
reload(unsvp)
reload(hyp)
reload(ptv)

def run_pipeline(cfg: Config):
    # ----------------------------- Preprocessing ----------------------------- #
    df = dp.load_data(cfg.DATASET_ID)                                  # raw
    df_binarized = dp.binarize_output(df.copy(deep= True))                             

    X, y = dp.split_features_target(df)

    X_binarized, y_binarized = dp.split_features_target(df_binarized)

    # fill missing
    X = dp.fill_missing(X, cfg.MISSIING_VALUES_COLUMNS)
    feature_names = X.columns

    # consistent splits for original and binarized targets
    x_train, x_test, y_train, y_test = dp.split_train_test(X, y, test_size=0.2, random_state=42)
    x_train_b, x_test_b, y_train_b, y_test_b = dp.split_train_test(X, y_binarized, test_size=0.2, random_state=42)

    # Standardize data (scale)
    x_train, x_test = dp.scale_data(x_train, x_test)
    # note: x_train_b == x_train after scaling since features are same; we keep separate names for clarity

    # correlation + quick pre-train viz
    corr_matrix = dp.correlation(df)
    print("Correlation matrix:\n", corr_matrix)
    print("=" * 30)
    dp.visualize_data(X, y, cfg.PRE_TRAIN_VISUALIZE_MODE)

    # ----------------------------- PCA (diagnostic only) ----------------------------- #
    pca, x_pca = pc.pca_analysis(x_train, cfg.N_PCA_COMPONENTS)
    pc.optimal_num_of_components(pca)
    print("=" * 30)
    pc.pca_visualization(x_train, y_train)

    # ----------------------------- Feature Selection ----------------------------- #
    # Important: clone to avoid overwriting the same estimator when fitting twice
    est_orig = clone(cfg.ESTIMATOR).fit(x_train, y_train)
    est_bin  = clone(cfg.ESTIMATOR).fit(x_train, y_train_b)

    feat_imp_orig = fs.feature_importances(feature_names, est_orig)
    feat_imp_bin  = fs.feature_importances(feature_names, est_bin)
    fs.visualize_importances(feat_imp_orig)
    print("---- Binarized features importances visualization ----")
    fs.visualize_importances(feat_imp_bin)

    # RFE masks for each target
    sel_mask_orig = fs.RFE_select(est_orig, cfg.STEP, x_train, y_train, cfg.CV)
    sel_names_orig = fs.get_selected_feature_names(x_train, sel_mask_orig)

    sel_mask_bin = fs.RFE_select(est_bin, cfg.STEP, x_train, y_train_b, cfg.CV)
    sel_names_bin = fs.get_selected_feature_names(x_train, sel_mask_bin)

    X_train_sel, X_test_sel = fs.model_selected_features(x_train, x_test, sel_names_orig)
    X_train_sel_b, X_test_sel_b = fs.model_selected_features(x_train, x_test, sel_names_bin)

    # ----------------------------- Train base models ----------------------------- #
    print("\n--- Training classification models (original target) ---")
    trained_svp = svp.train_model_svp(X_train_sel, y_train)
    print("\n--- Training classification models (binarized target) ---")
    trained_svp_b = svp.train_model_svp(X_train_sel_b, y_train_b)

    print("\n--- Training clustering models (original features) ---")
    trained_unsvp = unsvp.train_model_unsvp(X_train_sel) 
    print("\n--- Training clustering models (binarized features) ---")
    trained_unsvp_b = unsvp.train_model_unsvp(X_train_sel_b)

    # ----------------------------- Train-set evaluation (ONE pass each) ----------------------------- #
    print("\n--- Train set evaluation (original) ---")
    tr_scores, tr_recalls, tr_ROC, tr_AUC = svp.evaluate_svp_models(trained_svp, X_train_sel, y_train, cfg.DATASET_NAME)
    unsup_tr_scores = unsvp.evaluate_unsvp_models(trained_unsvp, X_train_sel, y_train, cfg.DATASET_NAME)

    print("\n--- Train set evaluation (binarized) ---")
    tr_scores_b, tr_recalls_b, tr_ROC_b, tr_AUC_b = svp.evaluate_svp_models(trained_svp_b, X_train_sel_b, y_train_b, cfg.DATASET_NAME)
    unsup_tr_scores_b = unsvp.evaluate_unsvp_models(trained_unsvp_b, X_train_sel_b, y_train_b, cfg.DATASET_NAME)

    # ----------------------------- Hyperparameter Tuning (Grid + Random) ----------------------------- #
    print("\n--- Hyperparameter tuning (GridSearchCV + RandomizedSearchCV) ---")
    optimal_grid_svp = {}
    optimal_rand_svp = {}
    optimal_grid_unsvp = {}
    optimal_rand_unsvp = {}
    optimal_grid_svp_b = {}
    optimal_rand_svp_b = {}
    optimal_grid_unsvp_b = {}
    optimal_rand_unsvp_b = {}

    # Fewer warnings during searches; raise real errors if they occur
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # SUPERVISED – ORIGINAL
        for name, base_model in trained_svp.items():
            search_est = Pipeline([("model", base_model)])

            gs = hyp.grid_search(
                X_train_sel, y_train,
                estimator=search_est,
                param_grid=cfg.PARAM_GRID_SVP[name],
                cv=cfg.CV,
                scoring="f1_macro"
            )
            optimal_grid_svp[name] = gs
            print(f"{name} (supervised) GridSearch best_score_: {gs.best_score_:.4f} | best_params_: {gs.best_params_}")

            rs = hyp.random_search(
                X_train_sel, y_train,
                estimator=search_est,
                param_distributions=cfg.PARAM_DISTRIBUTIONS_SVP[name],
                random_state=cfg.RANDOM_STATE,
                cv=cfg.CV,
                scoring="f1_macro"
            )
            optimal_rand_svp[name] = rs
            print(f"{name} (supervised) RandomizedSearch best_score_: {rs.best_score_:.4f} | best_params_: {rs.best_params_}")

        # SUPERVISED – BINARIZED
        print("\n")
        for name, base_model in trained_svp_b.items():
            search_est_b = Pipeline([("model", base_model)])

            gs_b = hyp.grid_search(
                X_train_sel_b, y_train_b,
                estimator=search_est_b,
                param_grid=cfg.PARAM_GRID_SVP[name],
                cv=cfg.CV,
                scoring="f1_macro"
            )
            optimal_grid_svp_b[name] = gs_b
            print(f"{name} (binarized supervised) GridSearch best_score_: {gs_b.best_score_:.4f} | best_params_: {gs_b.best_params_}")

            rs_b = hyp.random_search(
                X_train_sel_b, y_train_b,
                estimator=search_est_b,
                param_distributions=cfg.PARAM_DISTRIBUTIONS_SVP[name],
                random_state=cfg.RANDOM_STATE,
                cv=cfg.CV,
                scoring="f1_macro"
            )
            optimal_rand_svp_b[name] = rs_b
            print(f"{name} (binarized supervised) RandomizedSearch best_score_: {rs_b.best_score_:.4f} | best_params_: {rs_b.best_params_}")

        # UNSUPERVISED – ORIGINAL
        print("\n")
        for name, base_model in trained_unsvp.items():
            search_est_u = Pipeline([("model", base_model)])

            gs_u = hyp.grid_search(
                X_train_sel,                      
                estimator=search_est_u,
                param_grid=cfg.PARAM_GRID_UNSVP[name],
                cv=cfg.CV,
                scoring=hyp.silhouette_scorer
            )
            optimal_grid_unsvp[name] = gs_u
            print(f"{name} (unsupervised) GridSearch best_score_: {gs_u.best_score_:.4f} | best_params_: {gs_u.best_params_}")

            rs_u = hyp.random_search(
                X_train_sel,
                estimator=search_est_u,
                param_distributions=cfg.PARAM_DISTRIBUTIONS_UNSVP[name],
                random_state=cfg.RANDOM_STATE,
                cv=cfg.CV,
                scoring=hyp.silhouette_scorer
            )
            optimal_rand_unsvp[name] = rs_u
            print(f"{name} (unsupervised) RandomizedSearch best_score_: {rs_u.best_score_:.4f} | best_params_: {rs_u.best_params_}")

        # UNSUPERVISED – BINARIZED
        print("\n")
        for name, base_model in trained_unsvp_b.items():
            search_est_ub = Pipeline([("model", base_model)])

            gs_ub = hyp.grid_search(
                X_train_sel_b,
                estimator=search_est_ub,
                param_grid=cfg.PARAM_GRID_UNSVP[name],
                cv=cfg.CV,
                scoring=hyp.silhouette_scorer
            )
            optimal_grid_unsvp_b[name] = gs_ub
            print(f"{name} (binarized unsupervised) GridSearch best_score_: {gs_ub.best_score_:.4f} | best_params_: {gs_ub.best_params_}")

            rs_ub = hyp.random_search(
                X_train_sel_b,
                estimator=search_est_ub,
                param_distributions=cfg.PARAM_DISTRIBUTIONS_UNSVP[name],
                random_state=cfg.RANDOM_STATE,
                cv=cfg.CV,
                scoring=hyp.silhouette_scorer
            )
            optimal_rand_unsvp_b[name] = rs_ub
            print(f"{name} (binarized unsupervised) RandomizedSearch best_score_: {rs_ub.best_score_:.4f} | best_params_: {rs_ub.best_params_}")
    # ----------------------------- Test-set evaluation (ONE pass each) ----------------------------- #
    print("\n--- Test set evaluation (original) ---")
    test_scores, test_recalls, test_ROC, test_AUC = svp.evaluate_svp_models(trained_svp, X_test_sel, y_test, cfg.DATASET_NAME)
    unsup_test_scores = unsvp.evaluate_unsvp_models(trained_unsvp, X_test_sel, y_test, cfg.DATASET_NAME)

    print("\n--- Test set evaluation (binarized) ---")
    test_scores_b, test_recalls_b, test_ROC_b, test_AUC_b = svp.evaluate_svp_models(trained_svp_b, X_test_sel_b, y_test_b, cfg.DATASET_NAME)
    unsup_test_scores_b = unsvp.evaluate_unsvp_models(trained_unsvp_b, X_test_sel_b, y_test_b, cfg.DATASET_NAME)

    # ----------------------------- Visualizations (optional) ----------------------------- #
    print("\n--- Model visualization (original) ---")
    ptv.visualize_classification(X_test_sel, y_test, trained_svp, cfg.POST_TRAIN_VISUALIZE_MODE)
    ptv.visualize_cluster(X_test_sel, trained_unsvp, cfg.N_CLUSTERS)

    print("\n--- Model visualization (binarized) ---")
    ptv.visualize_classification(X_test_sel_b, y_test_b, trained_svp_b, cfg.POST_TRAIN_VISUALIZE_MODE)
    ptv.visualize_cluster(X_test_sel_b, trained_unsvp_b, cfg.N_CLUSTERS)

    # ----------------------------- Select and save best model ----------------------------- #
    best_model_name = max(test_recalls, key=test_recalls.get)   
    best_model = trained_svp[best_model_name]
    print(f"\nBest model (original target) based on Test recall: {best_model_name} "
          f"with 'macro' average recall = {test_recalls[best_model_name]:.4f}")
    hyp.save_model(best_model, cfg.MODEL_PATH)
    best_recall = test_recalls[best_model_name]

    y_pred = best_model.predict(X_test_sel)
    report = classification_report(y_test, y_pred, digits=4)

    with open(os.path.join("results", "best_model_results.txt"), "w") as f:
        f.write(f"Best supervised model (original target): {best_model_name}\n")
        f.write(f"Test macro recall: {best_recall:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print(f"Saved best model report to results/best_model_results.txt")

    #Best binarized model
    best_model_name_b = max(test_recalls_b, key=test_recalls_b.get)
    best_model_b = trained_svp_b[best_model_name_b]
    best_recall_b = test_recalls_b[best_model_name_b]
    hyp.save_model(best_model_b, cfg.MODEL_PATH_B)

    y_pred_b = best_model_b.predict(X_test_sel_b)
    report_b = classification_report(y_test_b, y_pred_b, digits=4)

    with open(os.path.join("results", "best_binarized_model_results.txt"), "w") as f:
        f.write(f"Best supervised model (binarized target): {best_model_name_b}\n")
        f.write(f"Test macro recall: {best_recall_b:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report_b)
    
    print(f"Saved best binarized model report to results/best_binarized_model_results.txt")