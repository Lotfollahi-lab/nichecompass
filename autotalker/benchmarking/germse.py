"""
This module contains a benchmark for testing how informative the gene program
scores are for gene expression regression.
"""

from typing import Optional, Union

import numpy as np
from anndata import AnnData
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def compute_gene_expression_regression_mse(
        adata: AnnData,
        cat_key: str="celltype_mapped_refined",
        active_gp_names_key: str="autotalker_active_gp_names",
        latent_key: str="autotalker_latent",
        selected_gps: Optional[Union[str,list]]=None,
        selected_genes: Optional[Union[str,list]]=None,
        n_features_gt_n_samples: bool=False,
        seed: int=0) -> float:
    """
    Use the gene program / latent scores of a trained Autotalker model for gene
    expression regression using a support vector machine regressor.

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
    active_gp_names_key:
        Key under which the active gene program names are stored in ´adata.uns´.
    latent_key:
        Key under which the latent representation from the model is stored in 
        ´adata.obsm´.
    selected_gps:
        List of gene program names which will be used for the classification 
        task. If ´None´, uses all active gene programs.
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
    active_gps = adata.uns[active_gp_names_key]
    if selected_gps is None:
        selected_gps = list(active_gps)
    else:
        if isinstance(selected_gps, str):
            selected_gps = [selected_gps]
    selected_gps_idx = np.array([list(active_gps).index(gp) for gp in 
                                 selected_gps])

    # Get selected genes
    if selected_genes is None:
        selected_genes = list(adata.var_names)
    else:
        if isinstance(selected_genes, str):
            selected_genes = [selected_genes]

    adata_selected_genes = adata[:, adata.var_names.isin(selected_genes)]
    gene_expr = adata_selected_genes.X.toarray()

    # Get gene program / latent scores for selected gene programs
    gp_scores = adata.obsm[latent_key][:, selected_gps_idx]

    # Train SVM classifier and use it for scoring
    clf = make_pipeline(StandardScaler(),
                        MultiOutputRegressor(SVR(C=1.0, epsilon=0.2)))
    clf.fit(X=gp_scores, y=gene_expr)
    gene_expr_preds = clf.predict(X=gp_scores)
    mse = mean_squared_error(gene_expr, gene_expr_preds)
    return mse