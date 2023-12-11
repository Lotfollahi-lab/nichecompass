from typer.testing import CliRunner
from pprint import pprint
from nichecompass.main import app
from nichecompass.utils import add_gps_from_gp_dict_to_adata
import json
import pickle
import numpy as np
import pandas as pd
import anndata as ad
import squidpy as sq
from scipy.sparse import csr_matrix
import os


def generate_mock_adata(sample_prefix="spatial"):
    counts = csr_matrix(np.random.poisson(100, size=(100, 2000)), dtype=np.float32)
    adata = ad.AnnData(counts)
    adata.obs_names = [f"Cell_{sample_prefix}_{i:d}" for i in range(adata.n_obs)]
    adata.var_names = [f"Gene_{i:d}" for i in range(adata.n_vars)]
    ct = np.random.choice(["B", "T", "Monocyte"], size=(adata.n_obs,))
    adata.obs["cell_type"] = pd.Categorical(ct)
    label = np.repeat([f"{sample_prefix}_sample_1", f"{sample_prefix}_sample_2", f"{sample_prefix}_sample_3", f"{sample_prefix}_sample_4"], round(adata.n_obs/4))
    adata.obs["label"] = pd.Categorical(label)
    adata.obsm["spatial"] = np.random.randint(0, 10000, size=(adata.n_obs, 2))
    return adata


def generate_mock_gene_programs(genes, n_receptors, n_ligands, min_size, max_size):
    included_genes = np.random.choice(genes, n_receptors + n_ligands, replace=False)
    receptors = included_genes[:n_receptors]
    ligands = included_genes[-n_ligands:]
    ligand_targets = {ligand: np.random.choice(receptors, np.random.randint(min_size, max_size), replace=False) for ligand in ligands}
    gp_dict = {f"{ligand}_gene_program": {
        "sources": [ligand],
        "sources_categories": ["ligand"],
        "targets": list(receptors),
        "targets_categories": ["receptor"] * len(receptors)
    } for ligand, receptors in ligand_targets.items()}
    return gp_dict


runner = CliRunner()


def test_train_reference(tmpdir):
    """Check that `nichecompass train-reference` command runs without an error"""

    run_config = {
        "artefact_directory": str(tmpdir.join("artefacts")),
        "gene_programs": {
            "sources": ["omnipath"],
            "species": "human",
            "filter_mode": "subset",
            "gene_orthologs_mapping_file_path": None,
            "export_file_path": str(tmpdir.join("gene_programs.pkl")),
        },
        "dataset": {
            "file_path": str(tmpdir.join("spatial_dataset.h5ad")),
            "library_key": "label",
            "spatial_key": "spatial",
            "export_file_path": str(tmpdir.join("spatial_dataset_built.h5ad")),
        },
        "gene_filters": {
            "n_highly_variable": None,
            "n_spatially_variable": None,
            "min_cell_gene_thresh_ratio": 0.1,
            "gene_program_relevant": True
        },
        "graph": {
            "n_neighbors": 3,
        },
        "model": {
            "cat_covariates_embeds_injection": ["gene_expr_decoder"],
            "cat_covariates_keys": None,
            "cat_covariates_no_edges": None,
            "cat_covariates_embeds_nums": None,
            "n_addon_gp": 10,
            "active_gp_thresh_ratio": 0.05,
            "n_fc_layers_encoder": 1,
            "n_layers_encoder": 1,
            "conv_layer_encoder": "gcnconv",
            "n_hidden_encoder": None,
            "node_label_method": "one-hop-norm",
        },
        "training": {
            "n_epochs": 400,
            "n_epochs_all_gps": 25,
            "n_epochs_no_cat_covariates_contrastive": 5,
            "lr": 0.001,
            "lambda_edge_recon": 5000000,
            "lambda_gene_expr_recon": 3000,
            "lambda_cat_covariates_contrastive": 0,
            "contrastive_logits_pos_ratio": 0,
            "contrastive_logits_neg_ratio": 0,
            "lambda_group_lasso": 0,
            "lambda_l1_masked": 0,
            "lambda_l1_addon": 0,
            "edge_batch_size": 256,
            "node_batch_size": None,
            "n_sampled_neighbors": 1,
            "artefact_directory": str(tmpdir.join("artefacts"))
        }
    }

    config_path = tmpdir.join("run-config.json")
    with open(config_path, 'w') as file:
        json.dump(run_config, file, indent=4)

    adata = generate_mock_adata()
    genes = adata.var_names
    gene_programs = generate_mock_gene_programs(genes, n_receptors=100, n_ligands=5, min_size=2, max_size=5)
    sq.gr.spatial_neighbors(adata, coord_type="generic", spatial_key="spatial", library_key="label", n_neighs=3)
    adjacency = adata.obsp["spatial_connectivities"]
    symmetrical_adjacency = adjacency.maximum(adjacency.T)
    adata.obsp["spatial_connectivities"] = symmetrical_adjacency
    add_gps_from_gp_dict_to_adata(gp_dict=gene_programs, adata=adata)
    adata.write(str(tmpdir.join("spatial_dataset_built.h5ad")))

    result_train = runner.invoke(app, ["train-reference", str(config_path)])
    assert result_train.exit_code == 0


def test_train_query(tmpdir):
    """Check that `nichecompass train-reference` command runs without an error"""

    # reference model

    run_config_reference = {
        "artefact_directory": str(tmpdir.join("artefacts")),
        "gene_programs": {
            "sources": ["omnipath"],
            "species": "human",
            "filter_mode": "subset",
            "gene_orthologs_mapping_file_path": None,
            "export_file_path": str(tmpdir.join("gene_programs.pkl")),
        },
        "dataset": {
            "file_path": str(tmpdir.join("spatial_dataset.h5ad")),
            "library_key": "label",
            "spatial_key": "spatial",
            "export_file_path": str(tmpdir.join("spatial_dataset_built.h5ad")),
        },
        "gene_filters": {
            "n_highly_variable": None,
            "n_spatially_variable": None,
            "min_cell_gene_thresh_ratio": 0.1,
            "gene_program_relevant": True
        },
        "graph": {
            "n_neighbors": 12,
        },
        "model": {
            "cat_covariates_embeds_injection": ["gene_expr_decoder"],
            "cat_covariates_keys": ["label"],
            "cat_covariates_no_edges": None,
            "cat_covariates_embeds_nums": [4],
            "n_addon_gp": 10,
            "active_gp_thresh_ratio": 0.05,
            "n_fc_layers_encoder": 1,
            "n_layers_encoder": 1,
            "conv_layer_encoder": "gcnconv",
            "n_hidden_encoder": None,
            "node_label_method": "one-hop-norm",
        },
        "training": {
            "n_epochs": 400,
            "n_epochs_all_gps": 25,
            "n_epochs_no_cat_covariates_contrastive": 5,
            "lr": 0.001,
            "lambda_edge_recon": 5000000,
            "lambda_gene_expr_recon": 3000,
            "lambda_cat_covariates_contrastive": 0,
            "contrastive_logits_pos_ratio": 0,
            "contrastive_logits_neg_ratio": 0,
            "lambda_group_lasso": 0,
            "lambda_l1_masked": 0,
            "lambda_l1_addon": 0,
            "edge_batch_size": 256,
            "node_batch_size": 256,
            "n_sampled_neighbors": 1,
            "artefact_directory": str(tmpdir.join("artefacts"))
        }
    }

    reference_config_path = tmpdir.join("run-config-reference.json")
    with open(reference_config_path, 'w') as file:
        json.dump(run_config_reference, file, indent=4)

    adata_reference = generate_mock_adata(sample_prefix="reference")
    genes = adata_reference.var_names
    gene_programs = generate_mock_gene_programs(genes, n_receptors=100, n_ligands=5, min_size=2, max_size=5)
    sq.gr.spatial_neighbors(adata_reference, coord_type="generic", spatial_key="spatial", library_key="label", n_neighs=3)
    adjacency = adata_reference.obsp["spatial_connectivities"]
    symmetrical_adjacency = adjacency.maximum(adjacency.T)
    adata_reference.obsp["spatial_connectivities"] = symmetrical_adjacency
    add_gps_from_gp_dict_to_adata(gp_dict=gene_programs, adata=adata_reference)
    adata_reference.write(str(tmpdir.join("spatial_dataset_built.h5ad")))

    result_train_reference = runner.invoke(app, ["train-reference", str(reference_config_path)])
    assert result_train_reference.exit_code == 0

    # query model

    reference_run_label = os.listdir(tmpdir.join("artefacts"))[0]

    run_config_query = {
        "artefact_directory": str(tmpdir.join("artefacts")),
        "gene_programs": {
            "sources": ["omnipath"],
            "species": "human",
            "filter_mode": "subset",
            "gene_orthologs_mapping_file_path": None,
            "export_file_path": str(tmpdir.join("gene_programs.pkl")),
        },
        "dataset": {
            "file_path": str(tmpdir.join("spatial_dataset.h5ad")),
            "library_key": "label",
            "spatial_key": "spatial",
            "export_file_path": str(tmpdir.join("spatial_dataset_built.h5ad")),
        },
        "gene_filters": {
            "n_highly_variable": None,
            "n_spatially_variable": None,
            "min_cell_gene_thresh_ratio": 0.1,
            "gene_program_relevant": True
        },
        "graph": {
            "n_neighbors": 3,
        },
        "model": {
            "cat_covariates_embeds_injection": ["gene_expr_decoder"],
            "cat_covariates_keys": ["label"],
            "cat_covariates_no_edges": None,
            "cat_covariates_embeds_nums": [4],
            "n_addon_gp": 10,
            "active_gp_thresh_ratio": 0.05,
            "n_fc_layers_encoder": 1,
            "n_layers_encoder": 1,
            "conv_layer_encoder": "gcnconv",
            "n_hidden_encoder": None,
            "node_label_method": "one-hop-norm",
        },
        "training": {
            "n_epochs": 400,
            "n_epochs_all_gps": 25,
            "n_epochs_no_cat_covariates_contrastive": 5,
            "lr": 0.001,
            "lambda_edge_recon": 5000000,
            "lambda_gene_expr_recon": 3000,
            "lambda_cat_covariates_contrastive": 0,
            "contrastive_logits_pos_ratio": 0,
            "contrastive_logits_neg_ratio": 0,
            "lambda_group_lasso": 0,
            "lambda_l1_masked": 0,
            "lambda_l1_addon": 0,
            "edge_batch_size": 256,
            "node_batch_size": 256,
            "n_sampled_neighbors": 1,
            "artefact_directory": str(tmpdir.join("artefacts"))
        },
        "train_query": {
            "reference_model": {
                "artefact_directory": str(tmpdir.join("artefacts", reference_run_label)),
                "file_path": "spatial_dataset_built.h5ad"
            },
            "dataset": {
                "file_path": str(tmpdir.join("spatial_dataset_query.h5ad")),
                "spatial_key": "spatial",
                "library_key": "label"
            },
            "graph": {
                "n_neighbors": 3
            },
            "training": {
                "n_epochs": 400,
                "n_epochs_all_gps": 25,
                "n_epochs_no_cat_covariates_contrastive": 5,
                "lr": 0.001,
                "lambda_edge_recon": 5000000,
                "lambda_gene_expr_recon": 3000,
                "lambda_cat_covariates_contrastive": 0,
                "contrastive_logits_pos_ratio": 0,
                "contrastive_logits_neg_ratio": 0,
                "lambda_group_lasso": 0,
                "lambda_l1_masked": 0,
                "lambda_l1_addon": 0,
                "edge_batch_size": 256,
                "node_batch_size": None,
                "n_sampled_neighbors": 1,
                "artefact_directory": str(tmpdir.join("artefacts"))
            }
        }
    }

    query_config_path = tmpdir.join("run-config-query.json")
    with open(query_config_path, 'w') as file:
        json.dump(run_config_query, file, indent=4)

    adata_query = generate_mock_adata(sample_prefix="query")
    sq.gr.spatial_neighbors(adata_query, coord_type="generic", spatial_key="spatial", library_key="label", n_neighs=3)
    adjacency = adata_query.obsp["spatial_connectivities"]
    symmetrical_adjacency = adjacency.maximum(adjacency.T)
    adata_query.obsp["spatial_connectivities"] = symmetrical_adjacency
    add_gps_from_gp_dict_to_adata(gp_dict=gene_programs, adata=adata_query)
    adata_query.write(str(tmpdir.join("spatial_dataset_query.h5ad")))

    result_train_query = runner.invoke(app, ["train-query", str(query_config_path)])
    assert result_train_query.exit_code == 0
