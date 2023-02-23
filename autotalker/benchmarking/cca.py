"""
This module contains the Cell Classification Accuracy (CCA) benchmark for
testing how accurately the latent feature space can predict cell categories
(e.g. cell-type).
"""

from typing import Literal, Optional, Union

import numpy as np
from anndata import AnnData
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def compute_cca(
        adata: AnnData,
        cell_cat_key: str="cell_type",
        active_gp_names_key: str="autotalker_active_gp_names",
        latent_key: str="autotalker_latent",
        classifier: Literal["baseline", "knn", "svm"]="mlp",
        selected_gps: Optional[Union[str,list]]=None,
        selected_cats: Optional[Union[str,list]]=None,
        n_neighbors: int=3,
        n_features_gt_n_samples: bool=False,
        seed: int=0,
        verbose: bool=False) -> float:
    """
    Use the latent representation / active gene program scores of a trained
    deep generative model for cell category (e.g. cell-type) classification 
    using a benchmark classifier. Compute the accuracy between the predicted 
    cell categories and the ground truth cell categories for the entire dataset.
    A higher value indicates that the latent space can more accurately predict
    cell categories.

    Parameters
    ----------
    adata:
        AnnData object with cell categories for classification stored in
        ´adata.obs[cell_cat_key]´, active gene program names stored in
        ´adata.uns[active_gp_names_key]´ and the latent representation stored in
        adata.obsm[latent_key].
    cell_cat_key:
        Key under which the cell categories that serve as classification labels
        are stored in ´adata.obs´.
    active_gp_names_key:
        Key under which the active gene program names are stored in ´adata.uns´.
    latent_key:
        Key under which the latent representation from the model is stored in
        ´adata.obsm´.
    classifier:
        Model algorithm used for cell category classification. If ´baseline´,
        predict the majority class for all cells.
    selected_gps:
        List of active gene program names which will be used for the
        classification task. If ´None´, uses all active gene programs.
    selected_cats:
        List of cell categories which will be included as separate labels in the
        classification task. If ´None´, uses all cell categories as separate
        labels.
    n_neighbors:
        Only relevant if ´classifier == knn´. Number of neighbors used for knn
        classification.
    n_features_gt_n_samples:
        Only relevant if ´classifier == svm´. If ´True´, select svm to solve
        dual optimization problem. Only set this to ´True´ if the number of
        features is greater than the number of samples.
    seed:
        Random seed for reproducibility.
    verbose:
        If ´True´, display ground truth label information for the classification
        task.

    Returns
    ----------
    cca:
        Cell classification accuracy.
    """
    # Assign classification labels. All categories that are not in
    # ´selected_cats´ will be assigned the label ´0´. All categories in
    # ´selected_cats´ will get their own label for multiclass classification
    cell_cat_values = adata.obs[cell_cat_key]
    if selected_cats is None:
        selected_cats = list(cell_cat_values.unique())
    elif isinstance(selected_cats, str):
        selected_cats = [selected_cats]
    cell_labels = np.zeros_like(cell_cat_values, dtype=np.int32)
    for i, cat in enumerate(selected_cats):
        cat_mask = cell_cat_values == cat
        cell_labels[cat_mask] = i + 1

    # Display classification label information
    if verbose:
        if len(selected_cats) < len(list(cell_cat_values.unique())):
            selected_cats = ["Other"] + selected_cats
        print("Cell categories used for classification and corresponding number"
              " of cells:")
        for i, cat in enumerate(selected_cats):
            print(f"{cat}: {np.unique(cell_labels, return_counts=True)[1][i]}")

    # Use classifier to predict cell labels and get accuracy
    if classifier == "baseline":
        # Predict majority class
        cell_labels_pred = np.full_like(cell_labels, 
                                        np.bincount(cell_labels).argmax())
        cca = accuracy_score(cell_labels, cell_labels_pred)
    else:
        # Get selected gps and their index in all active gps
        active_gps = list(adata.uns[active_gp_names_key])
        if selected_gps is None:
            selected_gps = active_gps
        else:
            if isinstance(selected_gps, str):
                selected_gps = [selected_gps]
        for gp in selected_gps:
            if gp not in active_gps:
                raise ValueError(f"GP {gp} is not an active gene program. "
                                 "Please only select active gene programs. ")
        selected_gps_idx = np.array([active_gps.index(gp) for gp in
                                     selected_gps])
        
        # Get latent representation / active gene program scores for selected
        # gene programs
        gp_scores = adata.obsm[latent_key][:, selected_gps_idx]

        # Predict cell categories using classifier
        if classifier == "mlp":
            clf = MLPClassifier(
                hidden_layer_sizes=(int(gp_scores.shape[1] / 2),
                                    int(gp_scores.shape[1] / 2)),
                random_state=seed,
                max_iter=500)
        if classifier == "knn":
            clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        elif classifier == "svm":
            clf = make_pipeline(StandardScaler(),
                                LinearSVC(random_state=seed,
                                          tol=1e-5,
                                          dual=n_features_gt_n_samples))
        clf.fit(X=gp_scores, y=cell_labels)
        cca = clf.score(X=gp_scores, y=cell_labels)
    return cca