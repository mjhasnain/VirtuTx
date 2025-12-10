# model.py


import torch
import torch.nn as nn
import math


class ConditionalVAE(nn.Module):
    def __init__(self, num_genes, num_conditions, latent_dim=128, hidden_dim=512):
        super().__init__()
        self.latent_dim = latent_dim
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
        h = self.encoder(torch.cat([x, c], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        return self.decoder(torch.cat([z, c], dim=1))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar


class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=1)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.15):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)   # ← FIXED: "tapped" removed
        )
        self.norm3 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context):
        attn, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn))
        cross, _ = self.cross_attn(x, context, context)
        x = self.norm2(x + self.dropout(cross))
        ffn = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn))
        return x


class DiffusionModel(nn.Module):
    def __init__(
        self,
        latent_dim,
        hidden_dim,
        noise_steps,
        device,
        num_transformer_blocks,
        num_heads,
        cell_line_dim,
        pert_method_dim,
        gene_embedding_dim,
        gene_token_dim=128,
        num_known_genes=0
    ):
        super().__init__()
        self.device = device
        self.noise_steps = noise_steps

        self.beta = torch.linspace(0.0001, 0.02, noise_steps, device=device).float()
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.time_emb = TimestepEmbedding(hidden_dim)
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)

        self.cell_line_proj = nn.Linear(cell_line_dim, hidden_dim)
        self.pert_method_proj = nn.Linear(pert_method_dim, hidden_dim)
        self.pert_gene_proj = nn.Linear(gene_embedding_dim, hidden_dim)

        self.gene_token_emb = nn.Embedding(num_known_genes + 1, gene_token_dim)
        self.gene_token_proj = nn.Linear(gene_token_dim, hidden_dim)

        self.transformer_blocks = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, num_heads) for _ in range(num_transformer_blocks)
        ])

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.Sigmoid()
        )
        self.to(device)

    def forward(self, x, t, cell_line_meta, pert_method_meta, pert_gene_embedding, gene_token_id, uncond_mask):
        B = x.shape[0]
        x = self.input_proj(x).unsqueeze(1)           # (B, 1, H)
        t_emb = self.time_proj(self.time_emb(t))      # (B, H)

        cl_emb = self.cell_line_proj(cell_line_meta)
        pm_emb = self.pert_method_proj(pert_method_meta)
        sig_emb = self.pert_gene_proj(pert_gene_embedding)
        gene_tok_raw = self.gene_token_emb(gene_token_id)
        gene_tok_emb = self.gene_token_proj(gene_tok_raw)

        context_emb = cl_emb + pm_emb + sig_emb + gene_tok_emb

        # Safe uncond_mask handling
        uncond_mask = uncond_mask.view(B, 1) * torch.ones_like(context_emb[:, :1])
        context_emb = context_emb * uncond_mask

        context = context_emb.unsqueeze(1)
        x = x + t_emb.unsqueeze(1)

        for block in self.transformer_blocks:
            x = block(x, context)

        return self.output_head(x.squeeze(1))

    def add_noise(self, x0, t):
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t]).view(-1, 1)
        sqrt_one_minus = torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1)
        noise = torch.randn_like(x0)
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise, noise