"""
This module contains a benchmark for testing how informative the gene program
scores are for cell category / annotation classification.
"""

from typing import Optional, Union

import numpy as np
from anndata import AnnData
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def compute_cell_cat_cls_accuracy(
        adata: AnnData,
        cat_key: str="celltype_mapped_refined",
        gp_names_key: str="autotalker_gp_names",
        latent_key: str="autotalker_latent",
        selected_gps: Optional[Union[str,list]]=None,
        selected_cats: Optional[Union[str,list]]=None,
        n_features_gt_n_samples: bool=False,
        seed: int=0) -> float:
    """
    Use the gene program / latent scores of a trained Autotalker model for cell
    category (e.g. cell type) classification using a support vector machine 
    classifier.

    Parameters
    ----------
    adata:
        AnnData object with cell categories / annotations stored in 
        ´adata.obs[cat_key]´, gene program names stored in
        ´adata.uns[gp_names_key]´ and the latent representation from the model
        stored in adata.obsm[latent_key].
    cat_key:
        Key under which the cell categories / annotations which serve as 
        classification label are stored in ´adata.obs´.
    gp_names_key:
        Key under which the gene program names are stored in ´adata.uns´
    latent_key:
        Key under which the latent representation from the model is stored in 
        ´adata.obsm´.
    selected_gps:
        List of gene program names which will be used for the classification 
        task. If ´None´, uses all gene programs.
    selected_cats:
        List of category labels which will be included as separate labels in the
        classification task. If ´None´, uses all category labels as separate
        labels.
    n_features_gt_n_samples:
        If ´True´, select algorithm to solve dual optimization problem. Only set
        this to ´True´ if the number of features is greater than the number of
        samples.
    seed:
        Random seed for reproducibility.

    Returns
    ----------
    accuracy:
        Cell category SVM classification accuracy.        
    """
    # Get index of selected gps
    if selected_gps is None:
        selected_gps = list(adata.uns[gp_names_key])
        selected_gps_idx = np.arange(len(selected_gps))
    else:
        if isinstance(selected_gps, str):
            selected_gps = [selected_gps]
        selected_gps_idx = np.array([list(adata.uns[gp_names_key]).index(gp) for
                                     gp in selected_gps])

    # Assign classification labels. All categories that are not in 
    # ´selected_cats´ will be assigned the label ´0´. All categories in
    # ´selected_cats´ will get their own label for multiclass classification
    cell_cat_values = adata.obs[cat_key]
    if selected_cats is None:
        selected_cats = list(cell_cat_values.unique())
    elif isinstance(selected_cats, str):
        selected_cats = [selected_cats]
    cell_cat_codes = np.zeros_like(cell_cat_values, dtype=np.int32)
    for i, cat in enumerate(selected_cats):
        cat_mask = cell_cat_values == cat
        cell_cat_codes[cat_mask] = i + 1

    # Display classification label information
    if len(selected_cats) < len(list(cell_cat_values.unique())):
        selected_cats = ["Other"] + selected_cats
    print("Cell type labels used for classification and corresponding number of"
          " cells:")
    for i, cat in enumerate(selected_cats):
        print(f"{cat}: {np.unique(cell_cat_codes, return_counts=True)[1][i]}")
    
    # Get gene program / latent scores for selected gene programs
    gp_scores = adata.obsm[latent_key][:, selected_gps_idx]

    # Train SVM classifier and use it for scoring
    clf = make_pipeline(StandardScaler(),
                        LinearSVC(random_state=seed,
                                  tol=1e-5,
                                  dual=n_features_gt_n_samples))
    clf.fit(X=gp_scores, y=cell_cat_codes)
    accuracy = clf.score(X=gp_scores, y=cell_cat_codes)
    return accuracy