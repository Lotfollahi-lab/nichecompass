import torch
from sklearn.metrics import roc_auc_score


def compute_dot_product_edge_reconstruction_auroc(
        adata,
        latent_rep_key,
        ):
    """
    
    """
    z = adata.obsm[latent_rep_key]
    dot_product = torch.mm(z, z.t())
    adj_recon_logits = 
    adj_recon_probs = torch.sigmoid(adj_recon_logits)
    torch.sigmoid()


