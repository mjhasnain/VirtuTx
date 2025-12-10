# =========================================================================================
# run_lfc_screen_final.py
#
# PURPOSE:
# A full-service screening and visualization script that:
# 1. Generates a "disease state" signature on the fly.
# 2. Screens it against a pre-computed library of drug/target signatures.
# 3. Ranks candidates for genetic rescue or synthetic lethality.
# 4. Automatically generates two publication-quality figures:
#    a) An annotated bar chart grouping hits by biological function.
#    b) A novel "Concordance Heatmap" to visualize the mechanistic landscape
#       of the top-ranked candidates.
# =========================================================================================

import pandas as pd
import numpy as np
import os
import argparse
import torch
import pickle
from tqdm import tqdm
from scipy.spatial.distance import cosine
import warnings

# Visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Suppress the expected RuntimeWarning from log2(0) and FutureWarning from Seaborn
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Model & Generation Imports ---
from model import DiffusionModel, ConditionalVAE
from opt import parse_args
try:
    from zero_shot_virtual_knockout import generate_population
except ImportError:
    print("[ERROR] Could not import `generate_population` from 'zero_shot_virtual_knockout.py'.")
    exit()

# --- Helper function to categorize drug targets for plotting ---
def categorize_gene(gene_name):
    """Categorizes a gene into a functional group for plotting."""
    gene_name = gene_name.upper()
    if any(kinase in gene_name for kinase in ['ABL', 'SRC', 'BCR', 'KIT', 'FLT3', 'LCK', 'FYN', 'MAPK', 'CDK', 'STK', 'PDGFR', 'FGFR', 'TEK', 'DDR1']):
        return "Kinase Signaling"
    if any(dna_syn in gene_name for dna_syn in ['RRM', 'POL', 'TOP', 'TYMS', 'IMPDH', 'PPAT', 'HPRT1']):
        return "DNA Synthesis & Repair"
    if any(apoptosis in gene_name for apoptosis in ['BCL2', 'BAX', 'MDM', 'TP53']):
        return "Apoptosis & Cell Cycle"
    if any(retinoid in gene_name for retinoid in ['RAR', 'RXR']):
        return "Retinoid Signaling"
    return "Other"

def main():
    parser = argparse.ArgumentParser(description="Run an LFC-based screen and generate figures.")
    parser.add_argument('--disease_gene', type=str, required=True, help='The gene whose knockout represents the disease/cancer state.')
    parser.add_argument('--cell_line', type=str, required=True, help='The cell line context (e.g., "K562").')
    parser.add_argument('--screen_type', type=str, required=True, choices=['rescue', 'lethality'], help='The goal of the screen.')
    parser.add_argument('--control_profiles_csv', type=str, default='control_profiles.csv', help='Path to the pre-generated control profiles CSV.')
    parser.add_argument('--signatures_dir', type=str, default='./moa_signatures', help='Directory of pre-computed drug/target LFC signatures.')
    parser.add_argument('--num_cells', type=int, default=2000, help='Number of virtual cells to generate for the disease state.')
    parser.add_argument('--output_dir', type=str, default='./screening_results', help='Directory to save results and figures.')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    model_args = parse_args([])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Step 1: Load Models and Data ---
    print("--- Loading models and pre-generated Control data ---")
    with open(model_args.vae_artifacts_path, 'rb') as f: vae_artifacts = pickle.load(f)
    ldm_artifacts_path = os.path.join(model_args.output_dir, 'ldm_artifacts.pkl')
    with open(ldm_artifacts_path, 'rb') as f: ldm_artifacts = pickle.load(f)
    with open('perturbation_signatures.pkl', 'rb') as f: signature_dict = pickle.load(f)
    ldm_artifacts['gene_embedding_dict'] = signature_dict
    vae_model = ConditionalVAE(len(vae_artifacts['gene_cols']), len(vae_artifacts['condition_encoder'].get_feature_names_out()), vae_artifacts['latent_dim']).to(device)
    vae_model.load_state_dict(torch.load(model_args.vae_path, map_location=device))
    model = DiffusionModel(latent_dim=vae_artifacts['latent_dim'], hidden_dim=model_args.embedding_dim, noise_steps=model_args.noise_steps, device=device, num_transformer_blocks=model_args.num_transformer_blocks, num_heads=model_args.num_heads, cell_line_dim=len(ldm_artifacts['ldm_encoders']['Cell_Line'].get_feature_names_out()), pert_method_dim=len(ldm_artifacts['ldm_encoders']['Genetic_perturbations'].get_feature_names_out()), gene_embedding_dim=next(iter(ldm_artifacts['gene_embedding_dict'].values())).shape[0], gene_token_dim=128, num_known_genes=len(ldm_artifacts['gene2id']))
    model.load_state_dict(torch.load(os.path.join(model_args.output_dir, 'best_model.pth'), map_location=device))
    control_profiles = pd.read_csv(args.control_profiles_csv)
    print("✓ Models and Control data loaded successfully.")

    # --- Step 2: Generate Disease Signature ---
    print(f"\n--- Generating disease signature for '{args.disease_gene}' knockout ---")
    disease_profiles = generate_population(model, vae_model, ldm_artifacts['latent_scaler'], ldm_artifacts, vae_artifacts, model_args, device, args.cell_line, args.disease_gene, args.num_cells, use_lfc_hint=False)
    disease_lfc_values = np.log2((disease_profiles.mean() + 1e-9) / (control_profiles.mean() + 1e-9))
    sanitized_lfc = np.nan_to_num(disease_lfc_values, nan=0.0, posinf=15.0, neginf=-15.0)
    disease_vector = pd.DataFrame({'gene': disease_lfc_values.index, 'log2FC': sanitized_lfc}).set_index('gene')
    print("✓ Disease signature generated and sanitized.")

    # --- Step 3: Load Drug Signatures ---
    print(f"\n--- Loading {len(os.listdir(args.signatures_dir))} pre-computed drug signatures ---")
    drug_signatures = {}
    for filename in tqdm(os.listdir(args.signatures_dir), desc="Loading Signatures"):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(args.signatures_dir, filename))
            df['log2FC'] = np.nan_to_num(df['log2FC'], nan=0.0, posinf=15.0, neginf=-15.0)
            drug_signatures[filename] = df.set_index('gene')

    # --- Step 4: Screening ---
    print(f"\n--- Screening {len(drug_signatures)} signatures for {args.screen_type} ---")
    results = []
    for drug_filename, drug_df in tqdm(drug_signatures.items(), desc="Screening"):
        combined = disease_vector.join(drug_df, how='inner', lsuffix='_disease', rsuffix='_drug')
        if len(combined) < 100: continue
        v_disease, v_drug = combined['log2FC_disease'].values, combined['log2FC_drug'].values
        similarity = 0 if np.all(v_disease == 0) or np.all(v_drug == 0) else 1 - cosine(v_disease, v_drug)
        results.append({'candidate_signature_file': drug_filename, 'cosine_similarity': similarity})
    
    results_df = pd.DataFrame(results)

    # --- Step 5: Logic, Ranking, and Saving CSV ---
    results_df['target_gene'] = results_df['candidate_signature_file'].apply(lambda x: x.split('_')[2])
    results_df['drug_name'] = results_df['candidate_signature_file'].apply(lambda x: x.split('_')[3].replace('.csv', ''))

    if args.screen_type == 'rescue':
        results_df['score'] = -1 * results_df['cosine_similarity']
        results_df.sort_values(by='score', ascending=False, inplace=True)
        print("\n--- TOP 20 GENETIC RESCUE CANDIDATES ---")
    else: # lethality
        results_df['score'] = results_df['cosine_similarity']
        results_df.sort_values(by='score', ascending=False, inplace=True)
        print("\n--- TOP 20 SYNTHETIC LETHALITY CANDIDATES ---")
    
    print(results_df.head(20).to_string(index=False))
    output_path = os.path.join(args.output_dir, f"{args.screen_type}_screen_for_{args.disease_gene}.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Full ranked results saved to '{output_path}'")
    
    # ==============================================================================
    # --- Step 6: AUTOMATED FIGURE GENERATION ---
    # ==============================================================================
    print("\n--- Generating publication-quality figures ---")
    
    top_n_hits = 15
    plot_df = results_df.head(top_n_hits).copy()

    # --- FIGURE 1: Annotated Bar Chart ---
    plot_df['category'] = plot_df['target_gene'].apply(categorize_gene)
    plot_df['label'] = plot_df['target_gene'] + ' (' + plot_df['drug_name'].str.capitalize() + ')'
    
    category_colors = {
        "Kinase Signaling": "#3498db", "DNA Synthesis & Repair": "#e74c3c",
        "Apoptosis & Cell Cycle": "#2ecc71", "Retinoid Signaling": "#f1c40f", "Other": "#95a5a6"
    }

    plt.figure(figsize=(12, 8))
    
    # --- THE FIX IS HERE ---
    # Instead of passing a pre-made color list, we tell seaborn which column to
    # use for `hue` and provide our dictionary of colors to the `palette` argument.
    sns.barplot(
        data=plot_df, 
        y='label', 
        x='score', 
        hue='category',         # <-- Color by this column
        palette=category_colors,# <-- Use this color mapping
        orient='h',
        legend=False            # <-- Disable the default legend, we will create a custom one
    )
    # --- END OF FIX ---
    
    plt.xlabel(f"Predicted {args.screen_type.capitalize()} Score", fontsize=14)
    plt.ylabel("Candidate Drug Target", fontsize=14)
    plt.title(f"In Silico Screen for {args.screen_type.capitalize()} of '{args.disease_gene}' KO in {args.cell_line}", fontsize=16, pad=20)
    plt.tick_params(axis='y', labelsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    # This manual legend creation will now work perfectly
    legend_patches = [Patch(color=color, label=cat) for cat, color in category_colors.items() if cat in plot_df['category'].unique()]
    plt.legend(handles=legend_patches, title='Functional Category', fontsize=12, title_fontsize=13)
    
    plt.tight_layout()
    fig1_path = os.path.join(args.output_dir, f"{args.screen_type}_barchart.png")
    plt.savefig(fig1_path, dpi=300)
    print(f"✓ Annotated bar chart saved to '{fig1_path}'")
    plt.close()

    # --- FIGURE 2: Novel Concordance Heatmap ---
    top_hits_signatures = []
    top_hits_labels = []
    for idx, row in plot_df.iterrows():
        sig_file = row['candidate_signature_file']
        if sig_file in drug_signatures:
            top_hits_signatures.append(drug_signatures[sig_file])
            top_hits_labels.append(row['label'])

    # Get the "disease fingerprint" genes
    disease_df_sorted = disease_vector.reindex(disease_vector.log2FC.abs().sort_values(ascending=False).index)
    fingerprint_genes = disease_df_sorted.head(100).index
    
    # Create a matrix of LFC values for the fingerprint genes across top hits
    concordance_matrix = pd.DataFrame(index=fingerprint_genes)
    for label, sig_df in zip(top_hits_labels, top_hits_signatures):
        concordance_matrix[label] = sig_df.reindex(fingerprint_genes)['log2FC']
    
    concordance_matrix.fillna(0, inplace=True)

    # Add the disease signature itself for comparison and sort columns by original score
    concordance_matrix[f'{args.disease_gene} KO (Disease)'] = disease_vector.reindex(fingerprint_genes)['log2FC']
    final_column_order = [f'{args.disease_gene} KO (Disease)'] + top_hits_labels
    concordance_matrix = concordance_matrix[final_column_order]
    
    # Cluster the HITS (not the genes) to see mechanistic groups
    clustergrid = sns.clustermap(
        concordance_matrix.T,  # Transpose to cluster the candidates
        cmap='vlag',
        center=0,
        xticklabels=False,
        dendrogram_ratio=(0.15, 0.05), # More space for row labels
        cbar_pos=(1.02, 0.6, 0.03, 0.2), # Position cbar neatly
        figsize=(10, 10),
        method='average',
        metric='cosine',
        row_cluster=True, # Cluster the candidates
        col_cluster=False # Do not cluster the genes, keep them sorted by effect
    )
    
    clustergrid.ax_heatmap.set_xlabel(f"Top 100 Genes in '{args.disease_gene}' KO Signature", fontsize=12)
    clustergrid.ax_heatmap.set_ylabel("")
    clustergrid.fig.suptitle(f"Transcriptional Concordance of Top Hits for {args.screen_type.capitalize()} of '{args.disease_gene}' KO", fontsize=16, y=1.02)
    plt.setp(clustergrid.ax_heatmap.get_yticklabels(), rotation=0, fontsize=10)
    
    fig2_path = os.path.join(args.output_dir, f"{args.screen_type}_concordance_heatmap.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    print(f"✓ Concordance heatmap saved to '{fig2_path}'")
    plt.close()

if __name__ == "__main__":
    main()