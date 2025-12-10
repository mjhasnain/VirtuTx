# =========================================================================================
# validate_zeroshot_only.py
#
# PURPOSE:
# A focused validation script that directly compares real test-set cells against
# the model's zero-shot predictions. This version removes the LFC-Hinted analysis
# to create clean, publication-ready figures for the main scientific claim.
#
# OUTPUTS:
# 1. Console output with the Real vs. Zero-Shot PCC score.
# 2. Five saved figures in the output directory, each showing a direct comparison:
#    - UMAP projection.
#    - Gene-gene correlation heatmaps.
#    - PCC scatter plot.
#    - Annotated distribution plots for key genes.
#    - Annotated covariance plots for key gene pairs.
# =========================================================================================

import torch
import pandas as pd
import numpy as np
import pickle
import argparse
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr, wasserstein_distance, entropy

# --- Dependency Checks ---
try:
    import umap
except ImportError:
    print("[ERROR] UMAP is not installed. Please run: pip install umap-learn")
    exit()

# --- Model and Helper Function Imports ---
from model import DiffusionModel, ConditionalVAE
from opt import parse_args

try:
    from zero_shot_virtual_knockout import generate_population
except ImportError:
    print("[ERROR] Could not import `generate_population`.")
    print("        Please ensure `zero_shot_virtual_knockout.py` (with the updated function) is in the same directory.")
    exit()

def main():
    parser = argparse.ArgumentParser(description="Deep validation of single-cell generative model distributions.")
    parser.add_argument('--gene', type=str, required=True, help='The perturbed gene to analyze.')
    parser.add_argument('--cell_line', type=str, required=True, help='The cell line context.')
    parser.add_argument('--output_dir', type=str, default='./zeroshot_validation_results', help='Directory to save the validation figures.')
    parser.add_argument('--num_cells_to_generate', type=int, default=None, help='Number of virtual cells to generate. Defaults to matching real cells.')
    
    args = parser.parse_args()
    model_args = parse_args([]) 
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("--- Loading all models and artifacts ---")
    with open(model_args.vae_artifacts_path, 'rb') as f: vae_artifacts = pickle.load(f)
    ldm_artifacts_path = os.path.join(model_args.output_dir, 'ldm_artifacts.pkl')
    with open(ldm_artifacts_path, 'rb') as f: ldm_artifacts = pickle.load(f)
    with open('perturbation_signatures.pkl', 'rb') as f: signature_dict = pickle.load(f)
    ldm_artifacts['gene_embedding_dict'] = signature_dict
    
    vae_model = ConditionalVAE(len(vae_artifacts['gene_cols']), len(vae_artifacts['condition_encoder'].get_feature_names_out()), vae_artifacts['latent_dim']).to(device)
    vae_model.load_state_dict(torch.load(model_args.vae_path, map_location=device))
    
    model = DiffusionModel(
        latent_dim=vae_artifacts['latent_dim'], hidden_dim=model_args.embedding_dim, noise_steps=model_args.noise_steps,
        device=device, num_transformer_blocks=model_args.num_transformer_blocks, num_heads=model_args.num_heads,
        cell_line_dim=len(ldm_artifacts['ldm_encoders']['Cell_Line'].get_feature_names_out()),
        pert_method_dim=len(ldm_artifacts['ldm_encoders']['Genetic_perturbations'].get_feature_names_out()),
        gene_embedding_dim=next(iter(ldm_artifacts['gene_embedding_dict'].values())).shape[0],
        gene_token_dim=128, num_known_genes=len(ldm_artifacts['gene2id'])
    )
    model.load_state_dict(torch.load(os.path.join(model_args.output_dir, 'best_model.pth'), map_location=device))
    print("✓ All models loaded.")

    print(f"\n--- Step 1: Loading real test cells for: {args.cell_line} | {args.gene} ---")
    test_data_full = pd.read_csv('test_data.tsv', sep='\t')
    real_profiles_df = test_data_full[(test_data_full['Cell_Line'] == args.cell_line) & (test_data_full['Perturbed_Gene'] == args.gene)]
    if real_profiles_df.empty:
        print(f"[ERROR] No cells found for the condition '{args.cell_line} | {args.gene}' in 'test_data.tsv'.")
        return
    
    gene_cols = vae_artifacts['gene_cols']
    real_profiles_df = real_profiles_df[gene_cols]
    print(f"Found {len(real_profiles_df)} real cells.")
    num_to_generate = args.num_cells_to_generate if args.num_cells_to_generate is not None else len(real_profiles_df)

    print("\n--- Step 2: Generating ZERO-SHOT virtual population ---")
    generated_zero_shot_df = generate_population(model, vae_model, ldm_artifacts['latent_scaler'], ldm_artifacts, vae_artifacts, model_args, device, args.cell_line, args.gene, num_to_generate, use_lfc_hint=False)

    real_profiles_df['Source'] = 'Real'; generated_zero_shot_df['Source'] = 'Generated (Zero-Shot)'
    combined_df = pd.concat([real_profiles_df, generated_zero_shot_df], ignore_index=True)

    # --- Step 3: GLOBAL ANALYSIS ---
    print("\n--- Step 3a: Calculating Mean Profile Correlations (PCC) ---")
    real_mean = real_profiles_df[gene_cols].mean(); zero_shot_mean = generated_zero_shot_df[gene_cols].mean()
    pcc_zero_shot, _ = pearsonr(real_mean, zero_shot_mean)
    print(f"  -> PCC (Real vs. Zero-Shot): {pcc_zero_shot:.4f}")

    print("\n--- Step 3b: Generating Gene-Gene Correlation Heatmaps ---")
    top_variable_genes = real_profiles_df[gene_cols].var().sort_values(ascending=False).head(50).index
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True) # Changed to 1, 2
    fig.suptitle(f'Gene-Gene Correlation (Top 50 Variable Genes) for {args.cell_line} | {args.gene}', fontsize=16)
    sns.heatmap(real_profiles_df[top_variable_genes].corr(), ax=axes[0], cmap='viridis', vmin=-1, vmax=1); axes[0].set_title('Real Cells')
    sns.heatmap(generated_zero_shot_df[top_variable_genes].corr(), ax=axes[1], cmap='viridis', vmin=-1, vmax=1); axes[1].set_title('Generated (Zero-Shot)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); fig_path_corr = os.path.join(args.output_dir, f'correlation_heatmap_{args.gene}.png'); plt.savefig(fig_path_corr, dpi=300); plt.close()
    print(f"✓ Correlation heatmap saved.")

    print("\n--- Step 3c: Generating UMAP Projections ---")
    n_neighbors_umap = min(30, len(real_profiles_df) - 1)
    reducer = umap.UMAP(n_neighbors=n_neighbors_umap, min_dist=0.1, random_state=42, n_components=2)
    reducer.fit(real_profiles_df[gene_cols])
    embedding_df = pd.DataFrame(reducer.transform(combined_df[gene_cols]), columns=['UMAP1', 'UMAP2']); embedding_df['Source'] = combined_df['Source']
    plt.figure(figsize=(10, 8)); sns.scatterplot(data=embedding_df, x='UMAP1', y='UMAP2', hue='Source', style='Source', s=10, alpha=0.6, palette={'Real': 'blue', 'Generated (Zero-Shot)': 'red'})
    title_str = (f"UMAP Projection for {args.cell_line} | {args.gene}\n" f"PCC (Real vs. Zero-Shot) = {pcc_zero_shot:.3f}")
    plt.title(title_str); plt.legend(title='Cell Source', markerscale=2)
    fig_path_umap = os.path.join(args.output_dir, f'umap_projection_{args.gene}.png'); plt.savefig(fig_path_umap, dpi=300); plt.close()
    print(f"✓ UMAP plot saved.")

    print("\n--- Step 3d: Generating PCC Scatter Plot ---")
    plt.figure(figsize=(8, 8)); pcc_df = pd.DataFrame({'Real Mean Expression': real_mean, 'Zero-Shot Mean Expression': zero_shot_mean})
    sns.scatterplot(data=pcc_df, x='Real Mean Expression', y='Zero-Shot Mean Expression', color='red', alpha=0.5, s=10, label=f'Zero-Shot (PCC={pcc_zero_shot:.3f})')
    lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]; plt.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='Ideal (y=x)')
    plt.title(f'Mean Gene Expression Correlation for {args.cell_line} | {args.gene}'); plt.xlabel('Mean Expression in Real Cells'); plt.ylabel('Mean Expression in Generated Cells')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
    fig_path_pcc = os.path.join(args.output_dir, f'pcc_scatter_{args.gene}.png'); plt.savefig(fig_path_pcc, dpi=300); plt.close()
    print(f"✓ PCC scatter plot saved.")

    # --- Step 4: GRANULAR (GENE-LEVEL) ANALYSIS with ANNOTATIONS ---
    key_genes = real_profiles_df[gene_cols].var().sort_values(ascending=False).head(5).index.tolist()
    
    print("\n--- Step 4a: Generating ANNOTATED Individual Gene Distribution Plots ---")
    fig, axes = plt.subplots(1, len(key_genes), figsize=(5 * len(key_genes), 5), sharey=True)
    if len(key_genes) == 1: axes = [axes]
    fig.suptitle(f'Distribution Comparison for {args.cell_line} | Perturbation: {args.gene}', fontsize=16)

    for i, gene in enumerate(key_genes):
        sns.kdeplot(data=combined_df, x=gene, hue='Source', fill=True, ax=axes[i], common_norm=False, palette={'Real': 'blue', 'Generated (Zero-Shot)': 'red'})
        axes[i].set_title(f'Expression of {gene.split(".")[0]}')
        
        real_data = combined_df[combined_df['Source'] == 'Real'][gene]; gen_data = combined_df[combined_df['Source'] == 'Generated (Zero-Shot)'][gene]
        var_delta = abs(real_data.var() - gen_data.var()); w_dist = wasserstein_distance(real_data, gen_data)
        bins = np.histogram(np.hstack((real_data, gen_data)), bins=50)[1]; real_hist = np.histogram(real_data, bins=bins, density=True)[0]; gen_hist = np.histogram(gen_data, bins=bins, density=True)[0]
        real_hist += 1e-9; gen_hist += 1e-9; kl_div = entropy(pk=gen_hist, qk=real_hist)
        
        text_str = f'VarΔ = {var_delta:.3f}\nKL = {kl_div:.3f}\nW = {w_dist:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        axes[i].text(0.05, 0.95, text_str, transform=axes[i].transAxes, fontsize=10, verticalalignment='top', bbox=props)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); fig_path_dist = os.path.join(args.output_dir, f'annotated_distribution_{args.gene}.png'); plt.savefig(fig_path_dist, dpi=300); plt.close()
    print(f"✓ Annotated distribution plot saved.")

    print("\n--- Step 4b: Generating ANNOTATED Gene-Pair Covariance Plots ---")
    primary_gene = key_genes[0]; secondary_genes = key_genes[1:4]
    fig, axes = plt.subplots(1, len(secondary_genes), figsize=(7 * len(secondary_genes), 6))
    if len(secondary_genes) == 1: axes = [axes]
    fig.suptitle(f'Covariance Comparison for {args.cell_line} | Perturbation: {args.gene}', fontsize=16)
    
    for i, secondary_gene in enumerate(secondary_genes):
        sns.scatterplot(data=combined_df, x=primary_gene, y=secondary_gene, hue='Source', style='Source', alpha=0.3, s=10, ax=axes[i], palette={'Real': 'blue', 'Generated (Zero-Shot)': 'red'})
        axes[i].set_title(f'{primary_gene.split(".")[0]} vs. {secondary_gene.split(".")[0]}')
        
        real_p, real_s = combined_df[combined_df['Source'] == 'Real'][[primary_gene, secondary_gene]].T.values
        gen_p, gen_s = combined_df[combined_df['Source'] == 'Generated (Zero-Shot)'][[primary_gene, secondary_gene]].T.values
        corr_real, _ = pearsonr(real_p, real_s); corr_gen, _ = pearsonr(gen_p, gen_s); corr_delta = abs(corr_real - corr_gen)
        
        text_str = f'CorrΔ = {corr_delta:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        axes[i].text(0.05, 0.95, text_str, transform=axes[i].transAxes, fontsize=10, verticalalignment='top', bbox=props)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); fig_path_cov = os.path.join(args.output_dir, f'annotated_covariance_{args.gene}.png'); plt.savefig(fig_path_cov, dpi=300); plt.close()
    print(f"✓ Annotated covariance plot saved.")
    
    print("\n\n--- Deep Validation Complete ---")

if __name__ == "__main__":
    main()