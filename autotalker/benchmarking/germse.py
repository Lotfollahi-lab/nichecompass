"""
This module contains the Gene Expression Regression Mean Squared Error (GERMSE)
benchmark for testing how accurately the latent feature space can predict the 
gene expression of a cell.
"""

from typing import Literal, Optional, Union

import numpy as np
from anndata import AnnData
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


def compute_germse(
        adata: AnnData,
        active_gp_names_key: str="autotalker_active_gp_names",
        latent_key: str="autotalker_latent",
        regressor: Literal["baseline", "mlp", "tree", "svm"]="mlp",
        selected_gps: Optional[Union[str,list]]=None,
        selected_genes: Optional[Union[str,list]]=None,
        seed: int=0) -> float:
    """
    Use the latent representation / active gene program scores of a trained
    deep generative model for gene expression regression using a benchmark
    regressor. Compute the mean squared error between the predicted gene
    expression and the ground truth gene expression for the entire dataset. A
    lower value indicates that the latent space can more accurately predict
    gene expression.

    Parameters
    ----------
    adata:
        AnnData object with active gene program names stored in
        ´adata.uns[active_gp_names_key]´, and the latent representation stored
        in ´adata.obsm[latent_key]´.
    active_gp_names_key:
        Key under which the active gene program names are stored in ´adata.uns´.
    latent_key:
        Key under which the latent representation from the model is stored in
        ´adata.obsm´.
    regressor:
        Model algorithm used for gene expression regression. If ´baseline´,
        predict the average gene expression of a gene across all cells for all
        genes.
    selected_gps:
        List of active gene program names which will be used for the regression
        task. If ´None´, uses all active gene programs.
    selected_genes:
        List of genes used as regression target in the regression task. If
        ´None´, use all probed genes.
    seed:
        Random seed for reproducibility.

    Returns
    ----------
    germse:
        Gene expression regression mean squared error.
    """
    # Get selected genes
    if selected_genes is None:
        selected_genes = list(adata.var_names)
    else:
        if isinstance(selected_genes, str):
            selected_genes = [selected_genes]
    selected_genes_idx = np.array([list(adata.var_names).index(gene) for gene in
                                   selected_genes])

    gene_expr = adata.X.toarray()[:, selected_genes_idx]

    # Use regressor to get gene expression predictions
    if regressor == "baseline":
        # Predict average gene expression across cells
        gene_expr_preds = np.repeat(gene_expr.mean(0)[np.newaxis, :],
                                    gene_expr.shape[0],
                                    axis=0)
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

        # Predict gene expression using regressor
        # Train regressor and use it for scoring
        if regressor == "mlp":
            regr = MLPRegressor(
                hidden_layer_sizes=(int(gp_scores.shape[1] / 2),
                                    int(gp_scores.shape[1] / 2)),
                random_state=seed,
                max_iter=500)
        elif regressor == "tree":
            regr = DecisionTreeRegressor(random_state=seed)
        elif regressor == "svm":
            regr = make_pipeline(StandardScaler(),
                                 MultiOutputRegressor(SVR(C=1.0,
                                                          epsilon=0.2)))
        regr.fit(X=gp_scores, y=gene_expr)
        gene_expr_preds = regr.predict(X=gp_scores)

    # Compute mse between ground truth and predicted gene expression
    germse = mean_squared_error(gene_expr, gene_expr_preds)
    return germse