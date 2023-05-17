"""
This module contains the Gene Expression Regression R-squared (gerr2) benchmark
for testing how accurately the latent feature space can linearly reconstruct the
gene expression of a cell.
"""

from typing import Literal, Optional, Union

import numpy as np
from anndata import AnnData
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor


def compute_gerr2(
        adata: AnnData,
        latent_key: str="nichecompass_latent",
        regressor: Literal["baseline", "mlp"]="mlp",
        selected_genes: Optional[Union[str,list]]=None,
        seed: int=0) -> float:
    """
    Use the latent representation of a trained deep generative model for linear
    gene expression regression using a single layer perceptron regressor and
    compute the R-squared between the predicted gene expression and the
    ground truth gene expression for all genes across the entire dataset. A 
    higher value indicates that the latent space can more accurately reconstruct
    gene expression in a linear way.

    Parameters
    ----------
    adata:
        AnnData object with the latent representation from a model stored in
        ´adata.obsm[latent_key]´.
    latent_key:
        Key under which the latent representation from the model is stored in
        ´adata.obsm´.
    regressor:
        Model algorithm used for gene expression regression. If ´baseline´,
        predict the average gene expression of a gene across all cells for all
        genes.
    selected_genes:
        List of genes used as regression target in the regression task. If
        ´None´, uses all probed genes.
    seed:
        Random seed for reproducibility.

    Returns
    ----------
    gerr2:
        Gene expression regression R-squared.
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
        # Get latent representation from a model
        latent = adata.obsm[latent_key]

        # Predict gene expression using regressor
        # Train regressor and use it for scoring
        if regressor == "mlp":
            regr = MLPRegressor(
                hidden_layer_sizes=(),
                random_state=seed,
                max_iter=500)
        regr.fit(X=latent, y=gene_expr)
        gene_expr_preds = regr.predict(X=latent)

    # Compute r2 between ground truth and predicted gene expression
    gerr2 = r2_score(gene_expr,
                     gene_expr_preds,
                     multioutput="uniform_average")
    return gerr2