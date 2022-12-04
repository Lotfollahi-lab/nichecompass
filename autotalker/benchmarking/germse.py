"""
This module contains a benchmark for testing how informative the (active) gene 
program scores are for gene expression regression.
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

from autotalker.data import SpatialAnnTorchDataset
from autotalker.models import Autotalker
from autotalker.nn import OneHopGCNNormNodeLabelAggregator
from autotalker.nn import SelfNodeLabelNoneAggregator


def compute_gene_expression_regression_mse(
        adata: AnnData,
        counts_key: str="counts",
        adj_key: str="spatial_connectivities",
        active_gp_names_key: str="autotalker_active_gp_names",
        latent_key: str="autotalker_latent",
        node_label_method: Literal["self", "one-hop-agg"]="one-hop-agg",
        model: Optional[Autotalker]=None,
        regressor: Literal["baseline", "decoder", "mlp", "tree", "svm"]="mlp",
        selected_gps: Optional[Union[str,list]]=None,
        selected_genes: Optional[Union[str,list]]=None,
        seed: int=0) -> float:
    """
    Use the gene program / latent scores of a trained Autotalker model for gene 
    expression regression using different benchmark regressors. Compute the
    mean squared error between the predicted gene expression and the ground 
    truth gene expression for the entire dataset.

    Parameters
    ----------
    adata:
        AnnData object with raw counts stored in ´adata.layers[counts_key]´, 
        sparse adjacency matrix stored in ´adata.obsp[adj_key]´, active gene 
        program names stored in ´adata.uns[active_gp_names_key]´, and gp scores
        stored in ´adata.obsm[latent_key]´.
    counts_key:
        Key under which the raw counts are stored in ´adata.layer´.
    adj_key:
        Key under which the sparse adjacency matrix is stored in ´adata.obsp´.
    active_gp_names_key:
        Key under which the active gene program names are stored in ´adata.uns´.
    latent_key:
        Key under which the latent representation from the model is stored in 
        ´adata.obsm´.
    node_label_method:
        Node label method used to determine the regression target. Should be 
        ´self´ if the node label method ´self´ was used for model training and 
        ´one-hop-agg´ otherwise. Check the ´Autotalker´ class for more 
        information.
    model:
        Only relevant if ´regressor == decoder´. A trained Autotalker model.
    regressor:
        Model algorithm used for gene expression regression. If ´baseline´, 
        predict the average gene expression of a gene across all cells for all
        genes.
    selected_gps:
        List of gene program names which will be used for the regression task. 
        If ´None´, uses all active gene programs.
    selected_genes:
        List of genes used as regression target in the regression task.
    seed:
        Random seed for reproducibility.

    Returns
    ----------
    mse:
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

    dataset = SpatialAnnTorchDataset(adata=adata,
                                     counts_key=counts_key,
                                     adj_key=adj_key)
    
    if node_label_method == "self":
        node_label_agg = SelfNodeLabelNoneAggregator()
    elif node_label_method == "one-hop-agg":
        neighbor_genes_idx = selected_genes_idx + len(adata.var_names)
        selected_genes_idx = np.concatenate(
            (selected_genes_idx, neighbor_genes_idx),
            axis=None)
        node_label_agg = OneHopGCNNormNodeLabelAggregator()

    gene_expr_labels = (node_label_agg(x=dataset.x,
                                       edge_index=dataset.edge_index,
                                       batch_size=len(dataset.x)).detach().cpu()
                                       .numpy())

    if regressor == "baseline":
        gene_expr_preds = np.repeat(gene_expr_labels.mean(0)[np.newaxis, :],
                                    gene_expr_labels.shape[0],
                                    axis=0)
    else:
        # Get selected gps and their index in all active gps
        active_gps = list(adata.uns[active_gp_names_key])
        if selected_gps is None:
            selected_gps = active_gps
        else:
            if isinstance(selected_gps, str):
                selected_gps = [selected_gps]
        selected_gps_idx = np.array([active_gps.index(gp) for gp in 
                                     selected_gps])

        # Get gene program / latent scores for selected gene programs
        gp_scores = adata.obsm[latent_key][:, selected_gps_idx]
        if regressor == "decoder":
            if model is None:
                raise ValueError("Please provide an Autotalker model instance "
                                 "or select a regressor model other than "
                                 "'decoder'. ")
            if model.gene_expr_recon_dist_ == "zinb":
                nb_means, zi_probs = model.get_gene_expr_dist_params()
                zi_mask = np.random.binomial(1, p=zi_probs)
                gene_expr_preds = nb_means
                gene_expr_preds[zi_mask] = 0
            elif model.gene_expr_recon_dist_ == "nb":
                gene_expr_preds = model.get_gene_expr_dist_params()
        else:
            # Train regressor and use it for scoring
            if regressor == "mlp":
                regr = MLPRegressor(
                    hidden_layer_sizes=int(gene_expr_labels.shape[1] / 2),
                    random_state=seed,
                    max_iter=500)
            elif regressor == "tree":
                regr = DecisionTreeRegressor(random_state=seed)
            elif regressor == "svm":
                regr = make_pipeline(StandardScaler(),
                                     MultiOutputRegressor(SVR(C=1.0,
                                                              epsilon=0.2)))
            regr.fit(X=gp_scores, y=gene_expr_labels)
            gene_expr_preds = regr.predict(X=gp_scores)

    mse = mean_squared_error(gene_expr_labels, gene_expr_preds)
    return mse