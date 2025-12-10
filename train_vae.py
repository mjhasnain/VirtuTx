# train_vae.py (Final Version)
# PURPOSE: To train a Conditional VAE that learns a compressed latent space from the 
#          cleaned and split gene expression data. This must be run after preparing the
#          data and before training the main Latent Diffusion Model.

import os
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import argparse
import warnings

# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def parse_vae_args():
    parser = argparse.ArgumentParser(description="Conditional VAE for Latent Space Learning")
    parser.add_argument('--output_dir', type=str, default='./final_model_run', help='Directory to save VAE model and artifacts')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=300, help='Maximum VAE training epochs')
    parser.add_argument('--latent_dim', type=int, default=128, help='Dimension of the VAE latent space')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for VAE optimizer')
    return parser.parse_args()

# --- VAE Model Architecture ---
class ConditionalVAE(nn.Module):
    def __init__(self, num_genes, num_conditions, latent_dim=128, hidden_dim=512):
        super().__init__()
        self.latent_dim = latent_dim
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(num_genes + num_conditions, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_conditions, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_genes),
            nn.Sigmoid()
        )

    def encode(self, x, c):
        combined = torch.cat([x, c], dim=1); h = self.encoder(combined); return self.fc_mu(h), self.fc_logvar(h)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar); eps = torch.randn_like(std); return mu + eps * std
    def decode(self, z, c):
        combined = torch.cat([z, c], dim=1); return self.decoder(combined)
    def forward(self, x, c):
        mu, logvar = self.encode(x, c); z = self.reparameterize(mu, logvar); return self.decode(z, c), mu, logvar

def vae_loss_function(recon_x, x, mu, logvar, beta=0.001):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + beta * KLD

def train():
    args = parse_vae_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print("--- Loading and Scaling Data for VAE ---")
    train_df = pd.read_csv('train_data.tsv', sep='\t')
    val_df = pd.read_csv('val_data.tsv', sep='\t')
    
    gene_cols = [c for c in train_df.columns if c.startswith('ENSG')]
    condition_cols = ['Cell_Line', 'Perturbed_Gene', 'Genetic_perturbations']
    
    scaler = MinMaxScaler()
    train_df[gene_cols] = scaler.fit_transform(train_df[gene_cols])
    val_df[gene_cols] = scaler.transform(val_df[gene_cols])
    
    print(f"  - Gene data scaled to [0, 1]. Number of genes: {len(gene_cols)}")

    # Fit the OneHotEncoder on the combined train and validation data to see all possible categories
    combined_df_for_encoder = pd.concat([train_df, val_df], ignore_index=True)
    condition_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(combined_df_for_encoder[condition_cols])
    
    print("  - Saving VAE-specific artifacts (scaler, one-hot encoder)...")
    vae_artifacts = {
        'scaler': scaler, 
        'condition_encoder': condition_encoder, 
        'gene_cols': gene_cols, 
        'latent_dim': args.latent_dim,
        'condition_cols': condition_cols
    }
    with open(os.path.join(args.output_dir, 'vae_artifacts.pkl'), 'wb') as f:
        pickle.dump(vae_artifacts, f)

    def get_loader(df, shuffle=True):
        conditions = torch.tensor(condition_encoder.transform(df[condition_cols]), dtype=torch.float32)
        genes = torch.tensor(df[gene_cols].values, dtype=torch.float32)
        print(f"  - Dataloader created with Genes shape: {genes.shape}, Conditions shape: {conditions.shape}")
        return DataLoader(TensorDataset(genes, conditions), batch_size=args.batch_size, shuffle=shuffle, num_workers=4)

    train_loader = get_loader(train_df, shuffle=True)
    val_loader = get_loader(val_df, shuffle=False)
    
    num_conditions_total = len(condition_encoder.get_feature_names_out())
    
    print(f"\n--- Model Initialization ---")
    print(f"  - Number of Genes (Input): {len(gene_cols)}")
    print(f"  - Number of Conditions (One-Hot Encoded): {num_conditions_total}")
    print(f"  - Latent Dimension: {args.latent_dim}")
    
    model = ConditionalVAE(len(gene_cols), num_conditions_total, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model_path = os.path.join(args.output_dir, 'best_vae_model.pth')
    
    best_val_loss = float('inf'); patience = 20; epochs_no_improve = 0
    print(f"\n--- Starting VAE Training on {device} ---")
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0
        for genes, conditions in train_loader:
            genes, conditions = genes.to(device), conditions.to(device)
            optimizer.zero_grad()
            recon_genes, mu, logvar = model(genes, conditions)
            loss = vae_loss_function(recon_genes, genes, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for genes, conditions in val_loader:
                genes, conditions = genes.to(device), conditions.to(device)
                recon_genes, mu, logvar = model(genes, conditions)
                val_loss += vae_loss_function(recon_genes, genes, mu, logvar).item()
        
        avg_train_loss = train_loss / len(train_df)
        avg_val_loss = val_loss / len(val_df)
        print(f"Epoch {epoch+1}/{args.num_epochs} | Train Loss: {avg_train_loss:.6f} | VAE Val Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
            print(f"  -> Validation loss improved. Saving model to {model_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs with no improvement.")
                break
    
    print("\n--- VAE Training Complete ---")
    print(f"Best VAE model saved to {model_path}")

if __name__ == "__main__":
    train()