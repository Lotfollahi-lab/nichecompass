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
        List of category labels which will be included in the classification
        task. If ´None´, uses all category labels. 
    seed:
        Random seed for reproducibility.

    Returns
    ----------
    accuracy:
        Cell category classification accuracy.        
    """
    # Get index of selected gps
    if selected_gps is None:
        selected_gps = adata.uns[gp_names_key]
        selected_gps_idx = np.arange(len(selected_gps))
    else: 
        if isinstance(selected_gps, str):
            selected_gps = [selected_gps]
        selected_gps_idx = [adata.uns[gp_names_key].index(gp) 
                            for gp in selected_gps]

    # Get mask of selected categories / annotations
    cell_cat_values = adata.obs[cat_key]
    if selected_cats is None:
        selected_cats = cell_cat_values.unique()
    elif isinstance(selected_cats, str):
        selected_cats = [selected_cats]
    selected_cats_mask = cell_cat_values.isin(selected_cats)
    
    # Get gene program / latent scores and cell categories / annotations for
    # the selected cell categories and gene programs
    gp_scores = adata.obsm[latent_key][selected_cats_mask][:, selected_gps_idx]
    cell_cat_values = cell_cat_values[selected_cats_mask]

    # Train SVM classifier and use it for scoring
    clf = make_pipeline(StandardScaler(),
                        LinearSVC(random_state=seed, tol=1e-5))
    clf.fit(X=gp_scores, y=cell_cat_values)
    accuracy = clf.score(X=gp_scores, y=cell_cat_values)
    return accuracy