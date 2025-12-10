# train.py


import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from model import DiffusionModel, ConditionalVAE
from opt import parse_args
from metrics import compute_all_metrics


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


class PerturbationSignatureDataset(Dataset):
    def __init__(self, df, vae_latents, ldm_encoders, signature_dict, gene2id):
        self.latents = vae_latents
        self.cell_line_enc = ldm_encoders['Cell_Line'].transform(df[['Cell_Line']])
        self.pert_method_enc = ldm_encoders['Genetic_perturbations'].transform(df[['Genetic_perturbations']])
        self.signature_keys = (df['Cell_Line'] + "_" + df['Perturbed_Gene']).values
        self.signature_dict = signature_dict
        self.embedding_dim = next(iter(signature_dict.values())).shape[0]
        self.gene2id = gene2id

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        signature_key = self.signature_keys[idx]
        signature = self.signature_dict.get(signature_key, np.zeros(self.embedding_dim))
        gene_name = signature_key.split('_')[1]
        gene_id = self.gene2id.get(gene_name, self.gene2id['__unknown__'])

        return (
            torch.tensor(self.cell_line_enc[idx], dtype=torch.float32),
            torch.tensor(self.pert_method_enc[idx], dtype=torch.float32),
            torch.tensor(signature, dtype=torch.float32),
            torch.tensor(self.latents[idx], dtype=torch.float32),
            torch.tensor(gene_id, dtype=torch.long)
        )


def get_latent_array(df, vae_model, vae_artifacts, batch_size, device):
    print(f" - Pre-calculating latents for {len(df)} samples...")
    encoder = vae_artifacts['condition_encoder']
    conds = encoder.transform(df[vae_artifacts['condition_cols']])
    genes = vae_artifacts['scaler'].transform(df[vae_artifacts['gene_cols']].values)
    latents = []
    vae_model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(genes), batch_size), desc="VAE encoding"):
            g = torch.tensor(genes[i:i+batch_size], dtype=torch.float32).to(device)
            c = torch.tensor(conds[i:i+batch_size], dtype=torch.float32).to(device)
            mu, _ = vae_model.encode(g, c)
            latents.append(mu.cpu().numpy())
    return np.concatenate(latents)


def evaluate_and_log(model, vae_model, latent_scaler, ldm_artifacts, val_data_raw, vae_artifacts, args, device, epoch):
    print(f"\n--- Validation at Epoch {epoch} ---")
    model.eval()
    vae_model.eval()

    gene2id = ldm_artifacts['gene2id']
    unknown_id = gene2id['__unknown__']
    embedding_dim = next(iter(ldm_artifacts['gene_embedding_dict'].values())).shape[0]

    # Prepare metadata
    df = val_data_raw
    cl_enc = ldm_artifacts['ldm_encoders']['Cell_Line'].transform(df[['Cell_Line']])
    pm_enc = ldm_artifacts['ldm_encoders']['Genetic_perturbations'].transform(df[['Genetic_perturbations']])

    cl_tensor = torch.tensor(cl_enc, dtype=torch.float32).to(device)
    pm_tensor = torch.tensor(pm_enc, dtype=torch.float32).to(device)

    # Real LFC signatures + gene tokens (guided mode)
    signature_keys = (df['Cell_Line'] + "_" + df['Perturbed_Gene']).values
    sig_list = []
    gene_id_list = []
    for key in signature_keys:
        sig = ldm_artifacts['gene_embedding_dict'].get(key, np.zeros(embedding_dim))
        sig_list.append(sig)
        gname = key.split('_')[1]
        gid = gene2id.get(gname, unknown_id)
        gene_id_list.append(gid)

    sig_tensor = torch.tensor(np.stack(sig_list), dtype=torch.float32).to(device)
    gene_id_tensor = torch.tensor(gene_id_list, dtype=torch.long).to(device)

    generated_latents_scaled = []
    bs = args.batch_size
    with torch.no_grad():
        for i in tqdm(range(0, len(df), bs), desc="Generating validation latents"):
            cur_bs = min(bs, len(df) - i)
            x_t = torch.randn(cur_bs, vae_artifacts['latent_dim'], device=device)

            cl_b = cl_tensor[i:i+cur_bs]
            pm_b = pm_tensor[i:i+cur_bs]
            sig_b = sig_tensor[i:i+cur_bs]
            gid_b = gene_id_tensor[i:i+cur_bs]

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
    vae_conds = torch.tensor(vae_artifacts['condition_encoder'].transform(df[vae_artifacts['condition_cols']]), dtype=torch.float32).to(device)
    decoded = []
    with torch.no_grad():
        for i in tqdm(range(0, len(df), bs), desc="Decoding"):
            dec = vae_model.decode(latents[i:i+bs], vae_conds[i:i+bs])
            decoded.append(dec.cpu().numpy())
    generated_profiles = np.concatenate(decoded)

    real_profiles = vae_artifacts['scaler'].transform(val_data_raw[vae_artifacts['gene_cols']].values)
    metrics = compute_all_metrics(real_profiles, generated_profiles, val_data_raw, vae_artifacts['gene_cols'])

    print("Validation Metrics:")
    for k, v in metrics.items():
        print(f"  {k:20}: {v:.6f}")

    return metrics


def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load VAE
    with open(args.vae_artifacts_path, 'rb') as f:
        vae_artifacts = pickle.load(f)
    with open('perturbation_signatures.pkl', 'rb') as f:
        signature_dict = pickle.load(f)

    vae_model = ConditionalVAE(
        len(vae_artifacts['gene_cols']),
        len(vae_artifacts['condition_encoder'].get_feature_names_out()),
        vae_artifacts['latent_dim']
    ).to(device)
    vae_model.load_state_dict(torch.load(args.vae_path, map_location=device))
    vae_model.eval()

    train_df = pd.read_csv('train_data.tsv', sep='\t')
    val_df = pd.read_csv('val_data.tsv', sep='\t')

    # Gene token mapping
    train_genes = sorted(train_df['Perturbed_Gene'].unique())
    gene2id = {g: i for i, g in enumerate(train_genes)}
    gene2id['__unknown__'] = len(gene2id)
    num_known_genes = len(gene2id)
    print(f"Gene tokens: {num_known_genes}")

    # Encoders
    combined = pd.concat([train_df, val_df])
    ldm_encoders = {
        'Cell_Line': OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(combined[['Cell_Line']]),
        'Genetic_perturbations': OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(combined[['Genetic_perturbations']])
    }

    # Latents
    train_latents = get_latent_array(train_df, vae_model, vae_artifacts, args.batch_size, device)
    latent_scaler = MinMaxScaler()
    train_latents_scaled = latent_scaler.fit_transform(train_latents)

    # Dataset
    train_dataset = PerturbationSignatureDataset(train_df, train_latents_scaled, ldm_encoders, signature_dict, gene2id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Dimensions
    cell_line_dim = len(ldm_encoders['Cell_Line'].get_feature_names_out())
    pert_dim = len(ldm_encoders['Genetic_perturbations'].get_feature_names_out())
    gene_embedding_dim = next(iter(signature_dict.values())).shape[0]

    # Model
    model = DiffusionModel(
        latent_dim=vae_artifacts['latent_dim'],
        hidden_dim=args.embedding_dim,
        noise_steps=args.noise_steps,
        device=device,
        num_transformer_blocks=args.num_transformer_blocks,
        num_heads=args.num_heads,
        cell_line_dim=cell_line_dim,
        pert_method_dim=pert_dim,
        gene_embedding_dim=gene_embedding_dim,
        gene_token_dim=128,
        num_known_genes=num_known_genes
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.num_warmup_epochs * len(train_loader),
        num_training_steps=args.num_epochs * len(train_loader)
    )

    best_score = -float('inf')
    log_path = os.path.join(args.output_dir, 'training_log.csv')
    with open(log_path, 'w') as f:
        f.write("epoch,train_loss,perturbation_corr\n")

    # Save artifacts
    ldm_artifacts = {
        'latent_scaler': latent_scaler,
        'ldm_encoders': ldm_encoders,
        'gene_embedding_dict': signature_dict,
        'gene2id': gene2id
    }
    with open(os.path.join(args.output_dir, 'ldm_artifacts.pkl'), 'wb') as f:
        pickle.dump(ldm_artifacts, f)

    print("Starting training...")
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        total_loss = 0.0
        for cl, pm, sig, z, gid in tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}"):
            cl, pm, sig, z, gid = cl.to(device), pm.to(device), sig.to(device), z.to(device), gid.to(device)

            optimizer.zero_grad()
            t = torch.randint(0, args.noise_steps, (z.size(0),), device=device)
            z_noisy, _ = model.add_noise(z, t)
            mask = (torch.rand(z.size(0), 1, device=device) > args.uncond_prob).float()

            pred = model(z_noisy, t, cl, pm, sig, gid, mask)
            loss = loss_fn(pred, z)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} | Loss: {avg_loss:.6f}")

        # Validation every 50 epochs
        if epoch % 50 == 0 or epoch == args.num_epochs:
            metrics = evaluate_and_log(model, vae_model, latent_scaler, ldm_artifacts, val_df, vae_artifacts, args, device, epoch)
            if metrics and (score := metrics.get('perturbation_corr', -1)) > best_score:
                best_score = score
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
                print(f"Saved new best model (pert_corr = {best_score:.4f})")

            with open(log_path, 'a') as f:
                f.write(f"{epoch},{avg_loss:.6f},{score if metrics else ''}\n")

    print("Training finished!")
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))


if __name__ == "__main__":
    train()