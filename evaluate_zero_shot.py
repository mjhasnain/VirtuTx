# zero_shot_evaluate.py
# FULL COMPLETE — TRUE ZERO-SHOT WITH GENE TOKEN + UNKNOWN TOKEN

import os
import torch
import pandas as pd
import numpy as np
import pickle
import argparse
from tqdm import tqdm
from model import DiffusionModel, ConditionalVAE
from opt import parse_args
from metrics import compute_all_metrics


def generate_zero_shot_profiles(model, vae_model, latent_scaler, ldm_artifacts, metadata_df, vae_artifacts, args, device):
    model.eval()
    vae_model.eval()
    num_samples = len(metadata_df)
    latent_dim = vae_artifacts['latent_dim']
    embedding_dim = next(iter(ldm_artifacts['gene_embedding_dict'].values())).shape[0]

    # Load gene2id and unknown token
    gene2id = ldm_artifacts['gene2id']
    unknown_gene_id = gene2id['__unknown__']

    # Encode cell line and method
    cl_enc = ldm_artifacts['ldm_encoders']['Cell_Line'].transform(metadata_df[['Cell_Line']])
    pm_enc = ldm_artifacts['ldm_encoders']['Genetic_perturbations'].transform(metadata_df[['Genetic_perturbations']])

    cl_tensor = torch.tensor(cl_enc, dtype=torch.float32).to(device)
    pm_tensor = torch.tensor(pm_enc, dtype=torch.float32).to(device)

    # ZERO LFC signature + UNKNOWN gene token
    zero_sig = torch.zeros(num_samples, embedding_dim, device=device)
    gene_token_ids = torch.full((num_samples,), unknown_gene_id, dtype=torch.long, device=device)

    generated_latents_scaled = []
    bs = args.batch_size

    with torch.no_grad():
        for i in tqdm(range(0, num_samples, bs), desc="Generating zero-shot latents"):
            cur_bs = min(bs, num_samples - i)
            x_t = torch.randn(cur_bs, latent_dim, device=device)

            cl_b = cl_tensor[i:i+cur_bs]
            pm_b = pm_tensor[i:i+cur_bs]
            sig_b = zero_sig[i:i+cur_bs]
            gid_b = gene_token_ids[i:i+cur_bs]

            for t in reversed(range(args.noise_steps)):
                t_t = torch.full((cur_bs,), t, dtype=torch.long, device=device)

                p_un = model(x_t, t_t, cl_b, pm_b, sig_b, gid_b, torch.zeros_like(cl_b[:, :1]))
                p_co = model(x_t, t_t, cl_b, pm_b, sig_b, gid_b, torch.ones_like(cl_b[:, :1]))
                pred = p_un + args.guidance_scale * (p_co - p_un)
                pred = torch.clamp(pred, 0, 1)

                if t > 0:
                    a = model.alpha_bar[t_t].view(-1, 1)
                    a_prev = model.alpha_bar[t_t - 1].view(-1, 1)
                    noise = (x_t - a.sqrt() * pred) / (1 - a).sqrt()
                    x_t = a_prev.sqrt() * pred + (1 - a_prev).sqrt() * noise
                else:
                    x_t = pred

            generated_latents_scaled.append(x_t.cpu())

    latents_scaled = torch.cat(generated_latents_scaled).numpy()
    latents = latent_scaler.inverse_transform(latents_scaled)
    latents = torch.tensor(latents, dtype=torch.float32).to(device)

    # Decode
    vae_conds = torch.tensor(vae_artifacts['condition_encoder'].transform(metadata_df[vae_artifacts['condition_cols']]), dtype=torch.float32).to(device)

    decoded = []
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, bs), desc="Decoding"):
            dec = vae_model.decode(latents[i:i+bs], vae_conds[i:i+bs])
            decoded.append(dec.cpu().numpy())

    return np.concatenate(decoded)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    cmd_args = parser.parse_args()

    args = parse_args([])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.vae_artifacts_path, 'rb') as f:
        vae_artifacts = pickle.load(f)
    with open(os.path.join(args.output_dir, 'ldm_artifacts.pkl'), 'rb') as f:
        ldm_artifacts = pickle.load(f)

    vae_model = ConditionalVAE(len(vae_artifacts['gene_cols']), len(vae_artifacts['condition_encoder'].get_feature_names_out()), vae_artifacts['latent_dim']).to(device)
    vae_model.load_state_dict(torch.load(args.vae_path, map_location=device))
    vae_model.eval()

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

    path = 'test_data.tsv' #if cmd_args.split == 'test' else 'val_data.tsv'
    df = pd.read_csv(path, sep='\t')

    print(f"\nTRUE ZERO-SHOT EVALUATION ON {cmd_args.split.upper()} ({len(df)} cells, {df['Perturbed_Gene'].nunique()} genes)")

    gen = generate_zero_shot_profiles(model, vae_model, ldm_artifacts['latent_scaler'], ldm_artifacts, df, vae_artifacts, args, device)

    real = vae_artifacts['scaler'].transform(df[vae_artifacts['gene_cols']].values)
    metrics = compute_all_metrics(real, gen, df, vae_artifacts['gene_cols'])

    print("\nZERO-SHOT METRICS:")
    for k, v in metrics.items():
        print(f"{k:20}: {v:.6f}")

    os.makedirs(os.path.join(args.output_dir, f"zero_shot_{cmd_args.split}"), exist_ok=True)
    pd.DataFrame([metrics]).to_csv(os.path.join(args.output_dir, f"zero_shot_{cmd_args.split}/metrics.csv"), index=False)

if __name__ == "__main__":
    main()