# =========================================================================================
# run_moa_generation.py (V4 - Efficient Version)
#
# PURPOSE: (STAGE 1 of 2)
# To perform the memory-intensive generation phase of the MoA screen in a highly
# efficient manner. It loads a pre-generated Control population and then, in a loop,
# generates only the perturbed population for each drug, calculates the DE signature,
# and saves it.
# =========================================================================================

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import argparse
import torch
import pickle
from scipy.stats import ttest_ind

from model import DiffusionModel, ConditionalVAE
from opt import parse_args

try:
    from zero_shot_virtual_knockout import generate_population
except ImportError:
    print("[ERROR] Could not import `generate_population`.")
    exit()

def main():
    parser = argparse.ArgumentParser(description="Run the efficient generation stage of a MoA screen.")
    parser.add_argument('--drug_list_csv', type=str, required=True, help='Path to the CSV file of drug-target pairs.')
    parser.add_argument('--cell_line', type=str, required=True, help='The cell line context.')
    parser.add_argument('--control_profiles_csv', type=str, default='control_profiles.csv', help='Path to the pre-generated control profiles CSV.')
    parser.add_argument('--sample', type=int, default=None, help='Optional: Run on a random sample of N drugs.')
    parser.add_argument('--num_cells', type=int, default=2000, help='Number of virtual cells to generate.')
    parser.add_argument('--output_dir', type=str, default='./moa_signatures', help='Directory to save all generated DE signature files.')
    
    args = parser.parse_args()
    model_args = parse_args([])
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("--- Loading models and pre-generated Control data ---")
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

    try:
        control_profiles = pd.read_csv(args.control_profiles_csv)
        print("✓ Models and Control data loaded successfully.")
    except FileNotFoundError:
        print(f"[ERROR] The control profiles file '{args.control_profiles_csv}' was not found.")
        print("Please run `generate_control.py` first to create this file.")
        return

    drug_list_df = pd.read_csv(args.drug_list_csv)
    if args.sample:
        sample_size = min(args.sample, len(drug_list_df))
        drug_list_df = drug_list_df.sample(n=sample_size, random_state=42)

    print(f"\n--- Starting screen for {len(drug_list_df)} drug-target pairs ---")

    for index, row in tqdm(drug_list_df.iterrows(), total=len(drug_list_df), desc="Generating MoA Signatures"):
        drug_name = row['drug_name']; target_gene = row['target_gene']
        
        safe_drug_name = "".join(c if c.isalnum() else "_" for c in drug_name)
        signature_filename = f"DE_signature_{target_gene}_{safe_drug_name}.csv"
        final_output_path = os.path.join(args.output_dir, signature_filename)

        if os.path.exists(final_output_path):
            continue

        # Generate ONLY the perturbed population
        perturbed_profiles = generate_population(model, vae_model, ldm_artifacts['latent_scaler'], ldm_artifacts, vae_artifacts, model_args, device, args.cell_line, target_gene, args.num_cells, use_lfc_hint=False)
        
        # Perform DE analysis directly
        de_results = []
        for gene_col in vae_artifacts['gene_cols']:
            l2fc = np.log2((perturbed_profiles[gene_col].mean() + 1e-9) / (control_profiles[gene_col].mean() + 1e-9))
            de_results.append({'gene': gene_col, 'log2FC': l2fc})
        de_df = pd.DataFrame(de_results)
        
        de_df.to_csv(final_output_path, index=False)

    print("\n--- MoA Signature Generation Complete ---")

if __name__ == "__main__":
    main()