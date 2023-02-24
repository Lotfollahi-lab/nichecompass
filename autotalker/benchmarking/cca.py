"""
This module contains the Cell Classification Accuracy (CCA) benchmark for
testing how accurately the latent feature space can linearly recover a cell
category, e.g. cell type.
"""

from typing import Literal, Optional, Union

import numpy as np
from anndata import AnnData
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


def compute_cca(
        adata: AnnData,
        cell_cat_key: str="cell_type",
        latent_key: str="autotalker_latent",
        classifier: Literal["baseline", "mlp"]="mlp",
        selected_cats: Optional[Union[str,list]]=None,
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
        ´adata.obs[cell_cat_key]´ and the latent representation from a model
        stored in adata.obsm[latent_key].
    cell_cat_key:
        Key under which the cell categories that serve as classification labels
        are stored in ´adata.obs´.
    latent_key:
        Key under which the latent representation from a model is stored in
        ´adata.obsm´.
    classifier:
        Model algorithm used for cell category classification. If ´baseline´,
        predict the majority class for all cells.
    selected_cats:
        List of cell categories which will be included as separate labels in the
        classification task. If ´None´, uses all cell categories as separate
        labels.
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
        # Get latent representation from a model
        latent = adata.obsm[latent_key]

        # Predict cell categories using classifier
        # Train classifier and use it for scoring
        if classifier == "mlp":
            clf = MLPClassifier(
                hidden_layer_sizes=(),
                random_state=seed,
                max_iter=500)
        clf.fit(X=latent, y=cell_labels)
        cca = clf.score(X=latent, y=cell_labels)
    return cca