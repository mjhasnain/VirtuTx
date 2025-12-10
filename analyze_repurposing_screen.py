# =========================================================================================
# analyze_repurposing_screen.py (V7 - ALL BUGS FIXED)
# =========================================================================================

import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob

# You need to import these for the script to run
import torch
import pickle
from model import DiffusionModel, ConditionalVAE
from opt import parse_args

try:
    from zero_shot_virtual_knockout import generate_population
except ImportError:
    print("[ERROR] Could not import `generate_population`.")
    exit()

def get_de_signature(control_profiles, perturbed_profiles):
    """
    Helper function to calculate a differential expression signature (log2FC).
    This version now floors expression values at zero to prevent log(negative) errors.
    """
    # --- THIS IS THE CRITICAL FIX FOR THE NaN PROBLEM ---
    control_mean = control_profiles.mean(axis=0).clip(lower=0)
    perturbed_mean = perturbed_profiles.mean(axis=0).clip(lower=0)
    # --- END OF FIX ---
    
    l2fc = np.log2((perturbed_mean + 1e-9) / (control_mean + 1e-9))
    return pd.DataFrame({'log2FC': l2fc})

def main():
    parser = argparse.ArgumentParser(description="Analyze a completed drug repurposing screen.")
    parser.add_argument('--disease_gene', type=str, required=True, help='The gene defining the disease state.')
    parser.add_argument('--cell_line', type=str, required=True, help='The cell line context.')
    parser.add_argument('--signatures_dir', type=str, default='./moa_signatures', help='Directory containing pre-generated drug signatures.')
    parser.add_argument('--drug_list_csv', type=str, required=True, help='Path to the original drug experiment list to get proper names.')
    parser.add_argument('--output_dir', type=str, default='./repurposing_results', help='Directory to save the final analysis.')
    
    args = parser.parse_args()
    model_args = parse_args([])
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("--- Loading models to generate disease signature ---")
    # ... (Model loading is the same)
    with open(model_args.vae_artifacts_path, 'rb') as f: vae_artifacts = pickle.load(f)
    ldm_artifacts_path = os.path.join(model_args.output_dir, 'ldm_artifacts.pkl')
    with open(ldm_artifacts_path, 'rb') as f: ldm_artifacts = pickle.load(f)
    with open('perturbation_signatures.pkl', 'rb') as f: signature_dict = pickle.load(f)
    ldm_artifacts['gene_embedding_dict'] = signature_dict
    vae_model = ConditionalVAE(len(vae_artifacts['gene_cols']), len(vae_artifacts['condition_encoder'].get_feature_names_out()), vae_artifacts['latent_dim']).to(device)
    vae_model.load_state_dict(torch.load(model_args.vae_path, map_location=device))
    model = DiffusionModel(latent_dim=vae_artifacts['latent_dim'], hidden_dim=model_args.embedding_dim, noise_steps=model_args.noise_steps, device=device, num_transformer_blocks=model_args.num_transformer_blocks, num_heads=model_args.num_heads, cell_line_dim=len(ldm_artifacts['ldm_encoders']['Cell_Line'].get_feature_names_out()), pert_method_dim=len(ldm_artifacts['ldm_encoders']['Genetic_perturbations'].get_feature_names_out()), gene_embedding_dim=next(iter(ldm_artifacts['gene_embedding_dict'].values())).shape[0], gene_token_dim=128, num_known_genes=len(ldm_artifacts['gene2id']))
    model.load_state_dict(torch.load(os.path.join(model_args.output_dir, 'best_model.pth'), map_location=device))
    print("✓ Models loaded.")

    try:
        drug_lookup_df = pd.read_csv(args.drug_list_csv)
    except FileNotFoundError:
        print(f"[ERROR] The drug list file '{args.drug_list_csv}' was not found.")
        return

    print("\n--- Step 1: Generating the Disease Signature ---")
    control_profiles = generate_population(model, vae_model, ldm_artifacts['latent_scaler'], ldm_artifacts, vae_artifacts, model_args, device, args.cell_line, "Control", 2000)
    disease_profiles = generate_population(model, vae_model, ldm_artifacts['latent_scaler'], ldm_artifacts, vae_artifacts, model_args, device, args.cell_line, args.disease_gene, 2000)
    disease_signature = get_de_signature(control_profiles, disease_profiles)
    disease_log2fc = disease_signature['log2FC']
    print(f"✓ Disease signature for '{args.disease_gene}' created.")
    
    drug_signature_files = glob.glob(os.path.join(args.signatures_dir, "*.csv"))
    print(f"\n--- Step 2: Screening {len(drug_signature_files)} pre-generated drug signatures ---")

    results = []
    for file_path in tqdm(drug_signature_files, desc="Calculating Connectivity"):
        try:
            filename = os.path.basename(file_path)
            target_gene_from_file = filename.split('_')[2]

            drug_df = pd.read_csv(file_path, index_col='gene')
            # The DE signatures from run_moa_generation might not have the clip() fix, so we clean them here too.
            drug_log2fc = drug_df['log2FC'].replace([np.inf, -np.inf], np.nan).dropna()

            aligned_disease, aligned_drug = disease_log2fc.align(drug_log2fc, join='inner')
            
            if len(aligned_drug) < 1000: continue

            connectivity_score, _ = pearsonr(aligned_disease, aligned_drug)
            
            matching_drug_row = drug_lookup_df[drug_lookup_df['target_gene'] == target_gene_from_file]
            proper_drug_name = matching_drug_row['drug_name'].iloc[0] if not matching_drug_row.empty else "Unknown Drug"

            results.append({'drug_name': proper_drug_name, 'target_gene': target_gene_from_file, 'connectivity_score': connectivity_score})
        except Exception as e:
            print(f"Could not process file {file_path}. Error: {e}")

    results_df = pd.DataFrame(results).sort_values(by='connectivity_score', ascending=True)
    results_df['plot_label'] = results_df['drug_name'] + " (" + results_df['target_gene'] + ")"
    
    print("\n--- Drug Repurposing Screen Results ---")
    print("Top 15 predicted therapeutic candidates:")
    print(results_df.head(15).to_string(index=False))
    results_df.to_csv(os.path.join(args.output_dir, f'repurposing_results_{args.disease_gene}.csv'), index=False)

    print("\nGenerating final plot...")
    top_hits = results_df.head(20); bottom_hits = results_df.tail(10)
    plot_df = pd.concat([top_hits, bottom_hits]).sort_values('connectivity_score', ascending=False)
    plot_df.set_index('plot_label', inplace=True)

    plt.figure(figsize=(8, 12))
    sns.heatmap(plot_df[['connectivity_score']], cmap="coolwarm_r", linewidths=.5, annot=True, fmt=".2f", vmin=-1, vmax=1)
    plt.title(f"Drug Repurposing Screen for '{args.disease_gene}' KO", fontsize=16)
    plt.xlabel("Connectivity Score", fontsize=12)
    plt.ylabel("Drug (Target)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'repurposing_heatmap_{args.disease_gene}.png'), dpi=300)
    
    print(f"\n✓ Repurposing analysis complete. Results and heatmap saved to '{args.output_dir}'")

if __name__ == "__main__":
    main()