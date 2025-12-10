# opt.py
import argparse

# --- FIX: Allow the function to accept an optional list of arguments ---
def parse_args(arg_list=None):
    parser = argparse.ArgumentParser(description="Latent Diffusion Model with Perturbation Signature Conditioning")
    
    parser.add_argument('--output_dir', type=str, default='./final_model_run', help='Directory for all model runs and artifacts')
    parser.add_argument('--vae_path', type=str, default='./final_model_run/best_vae_model.pth', help='Path to the pretrained VAE model')
    parser.add_argument('--vae_artifacts_path', type=str, default='./final_model_run/vae_artifacts.pkl', help='Path to VAE artifacts')
    
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=1500, help='Maximum number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Maximum learning rate after warmup.')
    parser.add_argument('--num_warmup_epochs', type=int, default=50, help='Number of epochs for learning rate warmup.')
    
    parser.add_argument('--noise_steps', type=int, default=1000, help='Number of noise steps in diffusion process')
    parser.add_argument('--embedding_dim', type=int, default=256, help='Dimension of hidden embeddings in Transformer')
    parser.add_argument('--num_transformer_blocks', type=int, default=8, help='Number of transformer blocks')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')

    parser.add_argument('--uncond_prob', type=float, default=0.1, help='Probability of dropping conditions for unconditional training')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Scale for amplifying conditional generation during evaluation')

    # --- FIX: Pass the optional list to the internal parse_args call ---
    # If arg_list is None, it defaults to using the command line.
    # If we provide a list (like []), it parses that list instead.
    return parser.parse_args(arg_list)

if __name__ == "__main__":
    args = parse_args()
    print("--- Model Arguments ---")
    for key, value in vars(args).items():
        print("{:<25}: {}".format(key, value))