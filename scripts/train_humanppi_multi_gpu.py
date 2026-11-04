
import os

import scanpy as sc
import squidpy as sq

from nichecompass.models import NicheCompass
from nichecompass.train import is_main_process
from nichecompass.utils import (add_gps_from_gp_dict_to_adata,
                                extract_gp_dict_from_humanppi_interactions)

GP_DIR = os.environ["GP_DATA_DIR"]
SO_DIR = os.environ["SO_DATA_DIR"]
GA_DIR = os.environ["GA_DATA_DIR"]
OUT_DIR = os.environ["OUT_DIR"]
PRECISION = os.environ.get("HUMANPPI_PRECISION", "90")

# 1. Prior gene programs. Every process runs this, so the caches must already
#    exist: a multi-process job must not have each rank race the same
#    download. Run this script once on one process first, or populate the
#    folder with prepare_gp_caches.py from the reproducibility repository.
network_file = f"{GP_DIR}/humanppi_network_{PRECISION}.csv"
humanppi_gp_dict = extract_gp_dict_from_humanppi_interactions(
    species="mouse",
    precision=PRECISION,
    program_type="intercellular",
    topology_file_path=f"{GP_DIR}/humanppi_protein_topology.tsv",
    omnipath_annotation_file_path=f"{GP_DIR}/omnipath_intercell_annotation.tsv",
    complex_portal_file_path=f"{GP_DIR}/complex_portal_human.tsv",
    load_from_disk=os.path.exists(network_file),
    save_to_disk=not os.path.exists(network_file),
    ppi_network_file_path=network_file,
    gene_orthologs_mapping_file_path=f"{GA_DIR}/human_mouse_gene_orthologs.csv",
    plot_gp_gene_count_distributions=False)
# The programs are not combined, as in the human PPI tutorial: combining merges
# programs with the same source genes into one ´<GENE>_combined_GP´, whose name
# no longer carries the interaction class.

# 2. Data and spatial graph
adata = sc.read_h5ad(f"{SO_DIR}/starmap_plus_mouse_cns_batch1.h5ad")
sq.gr.spatial_neighbors(adata, coord_type="generic", spatial_key="spatial",
                        n_neighs=4)
adata.obsp["spatial_connectivities"] = adata.obsp[
    "spatial_connectivities"].maximum(
        adata.obsp["spatial_connectivities"].T)
add_gps_from_gp_dict_to_adata(gp_dict=humanppi_gp_dict, adata=adata,
                              min_genes_per_gp=2,
                              min_source_genes_per_gp=1,
                              min_target_genes_per_gp=1)

# 3. Train across every process the launcher started
model = NicheCompass(adata, counts_key="counts", conv_layer_encoder="gcnconv",
                     active_gp_thresh_ratio=0.01)
model.train(n_epochs=400,
            n_epochs_all_gps=25,
            lambda_edge_recon=500000.,
            lambda_gene_expr_recon=300.,
            edge_batch_size=1024,
            n_sampled_neighbors=4,
            multi_gpu=True,
            profile="phase")

# 4. Everything after training belongs to the main process alone. The other
#    ranks returned from train() without a latent representation, so
#    sc.pp.neighbors would raise on them, and under mpirun one non-zero exit
#    tears down the whole job -- including the rank that was still writing.
if is_main_process():
    sc.pp.neighbors(model.adata, use_rep="nichecompass_latent",
                    key_added="nichecompass_latent")
    sc.tl.umap(model.adata, neighbors_key="nichecompass_latent")
    model.save(dir_path=f"{OUT_DIR}/model", overwrite=True, save_adata=True,
               adata_file_name="adata.h5ad")
