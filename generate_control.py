# =========================================================================================
# generate_control.py
#
# PURPOSE:
# A utility script to generate the common "Control" cell population ONCE. This
# pre-computed data will be reused by the main screening script to save a massive
# amount of time (cutting compute time nearly in half).
# =========================================================================================

import pandas as pd
import argparse
import os
import torch
import pickle

from model import DiffusionModel, ConditionalVAE
from opt import parse_args

try:
    from zero_shot_virtual_knockout import generate_population
except ImportError:
    print("[ERROR] Could not import `generate_population` from 'zero_shot_virtual_knockout.py'.")
    exit()

def main():
    parser = argparse.ArgumentParser(description="Generate a common Control cell population for screening.")
    parser.add_argument('--cell_line', type=str, required=True, help='The cell line context (e.g., "K562").')
    parser.add_argument('--num_cells', type=int, default=5000, help='Number of virtual cells to generate.')
    parser.add_argument('--output_file', type=str, default='./control_profiles.csv', help='Path to save the generated control profiles.')
    
    args = parser.parse_args()
    model_args = parse_args([]) 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("--- Loading all models and artifacts ---")
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
    print("✓ All models loaded.")

    print(f"\n--- Generating {args.num_cells} Control cells for {args.cell_line} ---")
    control_profiles = generate_population(model, vae_model, ldm_artifacts['latent_scaler'], ldm_artifacts, vae_artifacts, model_args, device, args.cell_line, "Control", args.num_cells, use_lfc_hint=False)
    
    control_profiles.to_csv(args.output_file, index=False)
    print(f"\n✓ Control profiles successfully generated and saved to '{args.output_file}'")

if __name__ == "__main__":
    main()