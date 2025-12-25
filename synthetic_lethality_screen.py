# synthetic_lethality_screen.py
# PURPOSE: Gene-centric Synthetic Lethality Screening with VirtuTx
# Uses target_gene column only — drug_name is ignored (can be anything).
# Handles duplicate genes, flexible headers, and your exact file format.

import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import os
import torch
import pickle
from zero_shot_virtual_knockout import generate_population
from model import DiffusionModel, ConditionalVAE
from opt import parse_args

# Can Expanded marker sets on Requirement
APOPTOSIS_GENES = [
    'ENSG00000121879', 'ENSG00000165806', 'ENSG00000106105',  # CASP3, CASP7, CASP6
    'ENSG00000137752', 'ENSG00000064012', 'ENSG00000120889',  # CASP9, CASP8, CASP10
    'ENSG00000143384', 'ENSG00000153064', 'ENSG00000171503',  # BAX, BAK1, BID
    'ENSG00000185344', 'ENSG00000100100', 'ENSG00000141682',  # BAD, BBC3, PMAIP1
    'ENSG00000171067', 'ENSG00000171791', 'ENSG00000171557',  # BMF, BCL2, BCL2L1
    'ENSG00000166741', 'ENSG00000087586', 'ENSG00000140465',  # MCL1, APAF1, CYCS
    'ENSG00000169410', 'ENSG00000165949', 'ENSG00000137757',  # DIABLO, XIAP, FAS
    'ENSG00000026136', 'ENSG00000136244', 'ENSG00000120896',  # FASLG, TNFRSF10B, TNFRSF10A
]

DIFFERENTIATION_GENES = [
    'ENSG00000112306',  # GATA1
    'ENSG00000105610',  # KLF1
    'ENSG00000159216',  # RUNX1
    'ENSG00000157554',  # TAL1/SCL
    'ENSG00000157456',  # HBB
    'ENSG00000105383',  # HBA1/HBA2
    'ENSG00000188536',  # HBG1/HBG2
    'ENSG00000148795',  # EPOR
]

def compute_marker_score(profile_df, marker_genes):
    available = [g for g in marker_genes if g in profile_df.columns]
    if not available:
        return 0.0
    return profile_df[available].mean().mean()

def compute_additive_deviation(disease_lfc, target_lfc, double_lfc):
    additive = disease_lfc + target_lfc
    return np.abs(double_lfc - additive).mean()

def get_lfc(control_df, perturbed_df):
    control_mean = control_df.mean(axis=0).clip(lower=0)
    perturbed_mean = perturbed_df.mean(axis=0).clip(lower=0)
    return np.log2((perturbed_mean + 1e-9) / (control_mean + 1e-9))

def main():
    parser = argparse.ArgumentParser(description="Gene-Centric Synthetic Lethality Screening")
    parser.add_argument('--disease_gene', type=str, required=True, help='Disease gene (e.g., TET2)')
    parser.add_argument('--cell_line', type=str, required=True, help='Cell line (e.g., K562)')
    parser.add_argument('--target_list_csv', type=str, required=True, 
                        help='CSV with columns: drug_name, target_gene (drug_name is ignored)')
    parser.add_argument('--num_cells', type=int, default=2000, help='Cells per population')
    parser.add_argument('--output_dir', type=str, default='./sl_results', help='Output directory')
    parser.add_argument('--score_type', type=str, default='combined',
                        choices=['apoptosis', 'differentiation', 'additive', 'combined'])
    
    args = parser.parse_args()
    model_args = parse_args([])
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    print("--- Loading models and artifacts ---")
    with open(model_args.vae_artifacts_path, 'rb') as f: vae_artifacts = pickle.load(f)
    ldm_artifacts_path = os.path.join(model_args.output_dir, 'ldm_artifacts.pkl')
    with open(ldm_artifacts_path, 'rb') as f: ldm_artifacts = pickle.load(f)
    with open('perturbation_signatures.pkl', 'rb') as f: signature_dict = pickle.load(f)
    ldm_artifacts['gene_embedding_dict'] = signature_dict

    vae_model = ConditionalVAE(len(vae_artifacts['gene_cols']), 
                              len(vae_artifacts['condition_encoder'].get_feature_names_out()), 
                              vae_artifacts['latent_dim']).to(device)
    vae_model.load_state_dict(torch.load(model_args.vae_path, map_location=device))

    model = DiffusionModel(
        latent_dim=vae_artifacts['latent_dim'],
        hidden_dim=model_args.embedding_dim,
        noise_steps=model_args.noise_steps,
        device=device,
        num_transformer_blocks=model_args.num_transformer_blocks,
        num_heads=model_args.num_heads,
        cell_line_dim=len(ldm_artifacts['ldm_encoders']['Cell_Line'].get_feature_names_out()),
        pert_method_dim=len(ldm_artifacts['ldm_encoders']['Genetic_perturbations'].get_feature_names_out()),
        gene_embedding_dim=next(iter(ldm_artifacts['gene_embedding_dict'].values())).shape[0],
        gene_token_dim=128,
        num_known_genes=len(ldm_artifacts['gene2id'])
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(model_args.output_dir, 'best_model.pth'), map_location=device))

    # Generate control
    print("\nGenerating control population...")
    control_df = generate_population(model, vae_model, ldm_artifacts['latent_scaler'], ldm_artifacts,
                                     vae_artifacts, model_args, device, args.cell_line, "Control", args.num_cells)

    # Generate disease KO
    print(f"\nGenerating disease KO: {args.disease_gene}")
    disease_df = generate_population(model, vae_model, ldm_artifacts['latent_scaler'], ldm_artifacts,
                                     vae_artifacts, model_args, device, args.cell_line, args.disease_gene, args.num_cells)
    disease_lfc = get_lfc(control_df, disease_df)

    # Load target list — only use target_gene
    print(f"\nLoading target genes from {args.target_list_csv}...")
    target_df = pd.read_csv(args.target_list_csv)
    
    # Flexible column names
    if 'target_gene' not in target_df.columns:
        raise ValueError("CSV must have a 'target_gene' column")
    # Optional: drug_name column is ignored
    
    # Remove duplicates — keep unique genes
    unique_targets = target_df['target_gene'].drop_duplicates().tolist()
    print(f"Found {len(unique_targets)} unique target genes (duplicates removed)")

    # Baselines
    baseline_apop = max(compute_marker_score(control_df, APOPTOSIS_GENES),
                        compute_marker_score(disease_df, APOPTOSIS_GENES))
    baseline_diff = max(compute_marker_score(control_df, DIFFERENTIATION_GENES),
                        compute_marker_score(disease_df, DIFFERENTIATION_GENES))

    results = []

    print(f"\nScreening {len(unique_targets)} unique target genes...")
    for target_gene in tqdm(unique_targets, desc="SL Screening"):
        # Single target KO
        target_df_pop = generate_population(model, vae_model, ldm_artifacts['latent_scaler'], ldm_artifacts,
                                            vae_artifacts, model_args, device, args.cell_line, target_gene, args.num_cells)
        target_lfc = get_lfc(control_df, target_df_pop)

        # Double KO
        double_genes = [args.disease_gene, target_gene]
        double_df = generate_population(model, vae_model, ldm_artifacts['latent_scaler'], ldm_artifacts,
                                        vae_artifacts, model_args, device, args.cell_line, double_genes, args.num_cells)
        double_lfc = get_lfc(control_df, double_df)

        # Scores
        double_apop = compute_marker_score(double_df, APOPTOSIS_GENES)
        target_apop = compute_marker_score(target_df_pop, APOPTOSIS_GENES)
        apop_score = double_apop - max(baseline_apop, target_apop)

        double_diff = compute_marker_score(double_df, DIFFERENTIATION_GENES)
        target_diff = compute_marker_score(target_df_pop, DIFFERENTIATION_GENES)
        diff_score = double_diff - max(baseline_diff, target_diff)

        additive_score = compute_additive_deviation(disease_lfc, target_lfc, double_lfc)

        if args.score_type == 'apoptosis':
            final_score = apop_score
        elif args.score_type == 'differentiation':
            final_score = diff_score
        elif args.score_type == 'additive':
            final_score = additive_score
        else:  # combined
            final_score = apop_score + diff_score + 0.5 * additive_score

        results.append({
            'target_gene': target_gene,
            'apoptosis_component': apop_score,
            'differentiation_component': diff_score,
            'additive_deviation': additive_score,
            'sl_score': final_score
        })

    results_df = pd.DataFrame(results).sort_values('sl_score', ascending=False)
    output_path = os.path.join(args.output_dir, f'sl_results_{args.disease_gene}_{args.score_type}.csv')
    results_df.to_csv(output_path, index=False)

    print(f"\n✓ Screening complete!")
    print(f"   Top target gene: {results_df.iloc[0]['target_gene']} | Score: {results_df.iloc[0]['sl_score']:.4f}")
    print(f"   Results saved to: {output_path}")

if __name__ == "__main__":
    main()
