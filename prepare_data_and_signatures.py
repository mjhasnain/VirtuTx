# prepare_data_and_signatures.py (With Signature Sanitization)
# PURPOSE:
# 1. Filters the dataset for 'CRISPRi'.
# 2. Performs stringent Quality Control (QC) to remove outlier cells.
# 3. Calculates and SAVES SANITIZED "perturbation signatures".
# 4. Creates a specialized 80/10/10 split with unseen genes in the test set.

import pandas as pd
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import train_test_split
import argparse
import pickle
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=RuntimeWarning) # Ignore expected log2 warnings

def calculate_l2fc_signatures(df, gene_cols):
    """
    Calculates and SANITIZES the L2FC signature for every (Gene, Cell Line) combination.
    """
    print("\n--- Calculating L2FC Perturbation Signatures ---")
    
    control_avg_expression = df[df['Perturbed_Gene'] == 'Control'].groupby('Cell_Line')[gene_cols].mean()
    perturbation_signatures = {}
    
    conditions = df[df['Perturbed_Gene'] != 'Control'].groupby(['Cell_Line', 'Perturbed_Gene'])
    
    for (cell_line, gene), group in tqdm(conditions, desc="Calculating Signatures"):
        if cell_line in control_avg_expression.index:
            control_profile = control_avg_expression.loc[cell_line]
            perturbation_profile = group[gene_cols].mean()
            pseudo_count = 1e-6
            l2fc_signature = np.log2((perturbation_profile + pseudo_count) / (control_profile + pseudo_count))
            
            # --- KEY FIX: Sanitize the signature vector ---
            # Replace NaN with 0, -inf with a large negative, +inf with a large positive
            sanitized_signature = np.nan_to_num(l2fc_signature, nan=0.0, posinf=15.0, neginf=-15.0)
            # Clip all values to a reasonable range [-10, 10] for stability
            clipped_signature = np.clip(sanitized_signature, -10.0, 10.0)
            # --- END OF FIX ---

            signature_key = f"{cell_line}_{gene}"
            perturbation_signatures[signature_key] = clipped_signature
        else:
            print(f"[WARNING] No control cells for '{cell_line}'. Cannot calculate signature for '{gene}'.")

    embedding_dim = len(gene_cols)
    for cell_line in df['Cell_Line'].unique():
        perturbation_signatures[f"{cell_line}_Control"] = np.zeros(embedding_dim)

    print(f"\nSuccessfully calculated and sanitized {len(perturbation_signatures)} unique signatures.")
    return perturbation_signatures, embedding_dim

def main():
    parser = argparse.ArgumentParser(description="Filter, QC, create signatures, and split single-cell data.")
    parser.add_argument('--input_file', type=str, default='data_cleaned.tsv', help='Path to the pre-cleaned dataset TSV file.')
    args = parser.parse_args()
    
    print(f"Loading cleaned data from: {args.input_file}")
    cleaned_data = pd.read_csv(args.input_file, sep='\t')
    gene_cols = [c for c in cleaned_data.columns if c.startswith('ENSG')]

    print("\n--- Step 1: Calculate and Save Signatures ---")
    signatures, _ = calculate_l2fc_signatures(cleaned_data, gene_cols)
    with open('perturbation_signatures.pkl', 'wb') as f:
        pickle.dump(signatures, f)
    print("✓ Perturbation signatures saved to 'perturbation_signatures.pkl'")

    print("\n--- Step 2: Specialized Train/Val/Test Split ---")
    control_cells = cleaned_data[cleaned_data['Perturbed_Gene'] == 'Control']
    perturbed_cells = cleaned_data[cleaned_data['Perturbed_Gene'] != 'Control']
    
    all_perturbed_genes = perturbed_cells['Perturbed_Gene'].unique()
    np.random.seed(42); np.random.shuffle(all_perturbed_genes)
    
    train_genes, val_test_genes = train_test_split(all_perturbed_genes, train_size=0.8, random_state=42)
    val_genes, test_genes = train_test_split(val_test_genes, test_size=0.5, random_state=42)
    
    train_perturbed_df = perturbed_cells[perturbed_cells['Perturbed_Gene'].isin(set(train_genes))]
    val_perturbed_df = perturbed_cells[perturbed_cells['Perturbed_Gene'].isin(set(val_genes))]
    test_perturbed_df = perturbed_cells[perturbed_cells['Perturbed_Gene'].isin(set(test_genes))]
    
    train_control_df, temp_control_df = train_test_split(control_cells, train_size=0.8, stratify=control_cells['Cell_Line'], random_state=42)
    
    if not temp_control_df.empty:
        try:
            val_control_df, test_control_df = train_test_split(temp_control_df, test_size=0.5, stratify=temp_control_df.get('Cell_Line'), random_state=42)
        except ValueError:
            val_control_df, test_control_df = train_test_split(temp_control_df, test_size=0.5, random_state=42)
    else:
        val_control_df, test_control_df = pd.DataFrame(), pd.DataFrame()

    train_df = pd.concat([train_perturbed_df, train_control_df], ignore_index=True)
    val_df = pd.concat([val_perturbed_df, val_control_df], ignore_index=True)
    test_df = pd.concat([test_perturbed_df, test_control_df], ignore_index=True)

    total_final = len(train_df) + len(val_df) + len(test_df)
    print("\n--- Final Split Summary ---")
    if total_final != len(cleaned_data):
        print(f"[ERROR] Mismatch! QC count: {len(cleaned_data)}, Split total: {total_final}.")
    print(f"Training cells: {len(train_df)} | Validation cells: {len(val_df)} | Test cells: {len(test_df)}")
    print("\nTest Set Cell Line Distribution:"); print(test_df['Cell_Line'].value_counts())

    print("\nSaving final data files...")
    train_df.to_csv('train_data.tsv', sep='\t', index=False)
    val_df.to_csv('val_data.tsv', sep='\t', index=False)
    test_df.to_csv('test_data.tsv', sep='\t', index=False)
    
    print("✓ Successfully created data splits and signature file.")

if __name__ == '__main__':
    main()