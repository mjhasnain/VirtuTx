# zero_shot_virtual_knockout.py
# PURPOSE: True zero-shot in-silico knockout of ANY gene (even completely unseen)
# Run: python zero_shot_virtual_knockout.py --gene MYC --cell_line K562 --num_cells 8000

import torch
import pandas as pd
import numpy as np
import pickle
import argparse
from tqdm import tqdm
from scipy.stats import ttest_ind
import gseapy as gp
import os
import mygene
from model import DiffusionModel, ConditionalVAE
from opt import parse_args


def generate_population(model, vae_model, latent_scaler, ldm_artifacts, vae_artifacts, args, device,
                        cell_line, gene_perturbation, n_samples=5000, use_lfc_hint=False):
    """
    Generates a population of virtual cells.

    Args:
        ... (all previous args) ...
        use_lfc_hint (bool): If True, use the real LFC signature and gene token.
                             If False, performs a true zero-shot prediction.
    """
    mode = "LFC-Hinted" if use_lfc_hint else "TRUE ZERO-SHOT"
    print(f"\n=== Generating {n_samples} virtual cells | {cell_line} | {gene_perturbation} ({mode}) ===")

    metadata_df = pd.DataFrame({
        'Cell_Line': [cell_line] * n_samples,
        'Genetic_perturbations': ['CRISPRi'] * n_samples,
        'Perturbed_Gene': [gene_perturbation] * n_samples
    })

    model.eval()
    vae_model.eval()

    # Encoders
    cl_enc = ldm_artifacts['ldm_encoders']['Cell_Line'].transform(metadata_df[['Cell_Line']])
    pm_enc = ldm_artifacts['ldm_encoders']['Genetic_perturbations'].transform(metadata_df[['Genetic_perturbations']])

    cl_tensor = torch.tensor(cl_enc, dtype=torch.float32).to(device)
    pm_tensor = torch.tensor(pm_enc, dtype=torch.float32).to(device)

    # --- DYNAMIC CONDITIONING BASED ON MODE ---
    embedding_dim = next(iter(ldm_artifacts['gene_embedding_dict'].values())).shape[0]
    gene2id = ldm_artifacts['gene2id']

    if use_lfc_hint:
        # LFC-Hinted Mode: Use the real pre-calculated signature and gene token
        signature_key = f"{cell_line}_{gene_perturbation}"
        signature_dict = ldm_artifacts['gene_embedding_dict']
        lfc_signature = signature_dict.get(signature_key, np.zeros(embedding_dim))
        
        signature_tensor = torch.tensor(lfc_signature, dtype=torch.float32).unsqueeze(0).repeat(n_samples, 1).to(device)
        
        gene_id = gene2id.get(gene_perturbation, gene2id['__unknown__']) # Use real token if known
        gene_token_ids = torch.full((n_samples,), gene_id, dtype=torch.long, device=device)

    else:
        # Zero-Shot Mode: Use zero signature and unknown gene token
        signature_tensor = torch.zeros(n_samples, embedding_dim, device=device)
        unknown_gene_id = gene2id['__unknown__']
        gene_token_ids = torch.full((n_samples,), unknown_gene_id, dtype=torch.long, device=device)
    # --- END DYNAMIC CONDITIONING ---


    generated_latents_scaled = []
    batch_size = args.batch_size

    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc=f"Diffusion sampling ({mode})"):
            bs = min(batch_size, n_samples - i)
            x_t = torch.randn(bs, vae_artifacts['latent_dim'], device=device)

            cl_b = cl_tensor[i:i+bs]
            pm_b = pm_tensor[i:i+bs]
            sig_b = signature_tensor[i:i+bs]
            gene_id_b = gene_token_ids[i:i+bs]

            for t in reversed(range(args.noise_steps)):
                t_tensor = torch.full((bs,), t, dtype=torch.long, device=device)
                pred_uncond = model(x_t, t_tensor, cl_b, pm_b, sig_b, gene_id_b, torch.zeros_like(cl_b[:, :1]))
                pred_cond = model(x_t, t_tensor, cl_b, pm_b, sig_b, gene_id_b, torch.ones_like(cl_b[:, :1]))
                pred_x0 = pred_uncond + args.guidance_scale * (pred_cond - pred_uncond)
                pred_x0 = torch.clamp(pred_x0, 0, 1)

                if t > 0:
                    a_bar = model.alpha_bar[t_tensor].view((-1, 1))
                    a_bar_prev = model.alpha_bar[t_tensor - 1].view(-1, 1)
                    pred_noise = (x_t - torch.sqrt(a_bar) * pred_x0) / torch.sqrt(1 - a_bar)
                    x_t = torch.sqrt(a_bar_prev) * pred_x0 + torch.sqrt(1 - a_bar_prev) * pred_noise
                else:
                    x_t = pred_x0

            generated_latents_scaled.append(x_t.cpu())

    generated_latents_scaled = torch.cat(generated_latents_scaled).numpy()
    generated_latents = latent_scaler.inverse_transform(generated_latents_scaled)
    generated_latents = torch.tensor(generated_latents, dtype=torch.float32).to(device)

    vae_conds = torch.tensor(vae_artifacts['condition_encoder'].transform(metadata_df[vae_artifacts['condition_cols']]), dtype=torch.float32).to(device)

    decoded_profiles = []
    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc=f"Decoding ({mode})"):
            z = generated_latents[i:i+batch_size]; c = vae_conds[i:i+batch_size]
            dec = vae_model.decode(z, c)
            decoded_profiles.append(dec.cpu().numpy())

    profiles_scaled = np.concatenate(decoded_profiles)
    profiles = vae_artifacts['scaler'].inverse_transform(profiles_scaled)
    return pd.DataFrame(profiles, columns=vae_artifacts['gene_cols'])


def convert_ensembl_to_symbols(ensembl_ids):
    mg = mygene.MyGeneInfo()
    results = mg.querymany(ensembl_ids, scopes='ensembl.gene', fields='symbol', species='human', verbose=False)
    symbols = []
    for r in results:
        symbols.append(r.get('symbol', r['query']))
    return symbols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gene', type=str, required=True, help="Gene symbol to knock out (e.g. MYC)")
    parser.add_argument('--cell_line', type=str, required=True, help="Cell line (e.g. K562)")
    parser.add_argument('--num_cells', type=int, default=8000)
    parser.add_argument('--output_dir', type=str, default='./zero_shot_virtual_results')
    args_cmd = parser.parse_args()

    args = parse_args([])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args_cmd.output_dir, exist_ok=True)

    # Load VAE
    with open(args.vae_artifacts_path, 'rb') as f:
        vae_artifacts = pickle.load(f)

    vae_model = ConditionalVAE(
        num_genes=len(vae_artifacts['gene_cols']),
        num_conditions=len(vae_artifacts['condition_encoder'].get_feature_names_out()),
        latent_dim=vae_artifacts['latent_dim']
    ).to(device)
    vae_model.load_state_dict(torch.load(args.vae_path, map_location=device))
    vae_model.eval()

    # Load LDM
    ldm_artifacts_path = os.path.join(args.output_dir, 'ldm_artifacts.pkl')
    with open(ldm_artifacts_path, 'rb') as f:
        ldm_artifacts = pickle.load(f)

    model = DiffusionModel(
        latent_dim=vae_artifacts['latent_dim'],
        hidden_dim=args.embedding_dim,
        noise_steps=args.noise_steps,
        device=device,
        num_transformer_blocks=args.num_transformer_blocks,
        num_heads=args.num_heads,
        cell_line_dim=len(ldm_artifacts['ldm_encoders']['Cell_Line'].get_feature_names_out()),
        pert_method_dim=len(ldm_artifacts['ldm_encoders']['Genetic_perturbations'].get_feature_names_out()),
        gene_embedding_dim=next(iter(ldm_artifacts['gene_embedding_dict'].values())).shape[0],
        gene_token_dim=128,
        num_known_genes=len(ldm_artifacts['gene2id'])
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pth'), map_location=device))
    model.eval()

    # Generate control and perturbed
    control_df = generate_population(model, vae_model, ldm_artifacts['latent_scaler'], ldm_artifacts,
                                     vae_artifacts, args, device, args_cmd.cell_line, "Control", args_cmd.num_cells)
    perturbed_df = generate_population(model, vae_model, ldm_artifacts['latent_scaler'], ldm_artifacts,
                                       vae_artifacts, args, device, args_cmd.cell_line, args_cmd.gene, args_cmd.num_cells)

    # DE analysis
    results = []
    for gene in vae_artifacts['gene_cols']:
        c = control_df[gene]
        p = perturbed_df[gene]
        stat, pv = ttest_ind(p, c, equal_var=False)
        l2fc = np.log2((p.mean() + 1e-6) / (c.mean() + 1e-6))
        results.append({'gene': gene, 'log2FC': l2fc, 'pval': pv})

    de_df = pd.DataFrame(results).sort_values('pval')
    de_path = os.path.join(args_cmd.output_dir, f"DE_{args_cmd.cell_line}_{args_cmd.gene}.csv")
    de_df.to_csv(de_path, index=False)
    print(f"DE results saved to {de_path}")

    # Pathway enrichment on top 100 up/down
    top_up = de_df[de_df['log2FC'] > 0].head(100)['gene'].tolist()
    top_down = de_df[de_df['log2FC'] < 0].head(100)['gene'].tolist()
    gene_list = top_up + top_down
    symbols = convert_ensembl_to_symbols(gene_list)

    enr = gp.enrichr(gene_list=symbols,
                     gene_sets=['KEGG_2021_Human', 'Reactome_2022', 'GO_Biological_Process_2021'],
                     organism='Human',
                     outdir=args_cmd.output_dir,
                     cutoff=0.05)

    print("Top predicted pathways:")
    if not enr.results.empty:
        print(enr.results.head(10)[['Term', 'Adjusted P-value']])

    print(f"\nAll results saved in {args_cmd.output_dir}")
    print("This was a TRUE zero-shot prediction — no LFC leakage!")


if __name__ == "__main__":
    main()