# =========================================================================================
# analyze_multi_pathway.py
#
# PURPOSE: (STAGE 2 - Multi-Pathway Version)
# To perform a broad, unbiased analysis of pre-generated DE signature files.
# This script screens each drug signature against a curated list of key cancer
# pathways across multiple databases and visualizes the results as a comprehensive heatmap.
# =========================================================================================

import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import os
import gseapy as gp
import glob
import mygene

def convert_ensembl_to_symbols(ensembl_ids):
    mg = mygene.MyGeneInfo()
    results = mg.querymany(ensembl_ids, scopes='ensembl.gene', fields='symbol', species='human', verbose=False)
    return [r.get('symbol', r['query']) for r in results]

def main():
    parser = argparse.ArgumentParser(description="Analyze a completed MoA screen against multiple key pathways.")
    parser.add_argument('--signatures_dir', type=str, default='./moa_signatures', help='Directory containing all generated DE signature files.')
    parser.add_argument('--output_dir', type=str, default='./multi_pathway_analysis', help='Directory to save the final analysis.')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Step 1: Define Key Cancer Pathways ---
    # This list is curated from literature for relevance to K562 (CML) and general cancer biology.
    # The names must be EXACT matches to terms in the databases.
    key_pathways = [
        'Cell cycle', # KEGG
        'Apoptosis', # KEGG
        'p53 signaling pathway', # KEGG
        'Chronic myeloid leukemia', # KEGG
        'Jak-STAT signaling pathway', # KEGG
        'MAPK signaling pathway', # KEGG
        'PI3K-Akt signaling pathway', # KEGG
        'MYC Targets V1', # GO/Hallmark (if using MSigDB gmt)
        'HALLMARK_MYC_TARGETS_V1', # MSigDB Hallmark
        'HALLMARK_APOPTOSIS', # MSigDB Hallmark
        'positive regulation of apoptotic process', # GO Biological Process
        'erythrocyte differentiation' # GO Biological Process
    ]
    key_pathways_lower = [p.lower() for p in key_pathways]

    # Define the broad databases to search
    databases = ['KEGG_2021_Human', 'GO_Biological_Process_2021', 'Reactome_2022']

    signature_files = glob.glob(os.path.join(args.signatures_dir, "*.csv"))
    if not signature_files:
        print(f"[ERROR] No signature files found in '{args.signatures_dir}'.")
        return
        
    print(f"Found {len(signature_files)} signatures to analyze against {len(key_pathways)} key pathways.")

    all_results = []
    for file_path in tqdm(signature_files, desc="Analyzing Signatures"):
        filename = os.path.basename(file_path)
        try:
            parts = filename.replace('DE_signature_', '').replace('.csv', '').split('_')
            target_gene = parts[0]
            drug_name = "_".join(parts[1:])
        except IndexError:
            continue
        
        de_df = pd.read_csv(file_path).sort_values('log2FC', ascending=False)
        top_genes_symbols = convert_ensembl_to_symbols(de_df.head(200)['gene'].tolist())

        try:
            # Run enrichment across all specified databases at once
            enr = gp.enrichr(gene_list=top_genes_symbols, gene_sets=databases, organism='Human', outdir=None, cutoff=1.0)
            
            drug_scores = {'drug_target': f"{drug_name.replace('_', ' ').title()} ({target_gene})"}
            
            # Filter the full results for our key pathways
            enr_results = enr.results
            enr_results['Term_lower'] = enr_results['Term'].str.lower()
            found_pathways = enr_results[enr_results['Term_lower'].isin(key_pathways_lower)]

            # For each key pathway, find the score if it exists
            for pathway in key_pathways:
                pathway_row = found_pathways[found_pathways['Term_lower'] == pathway.lower()]
                
                efficacy_score = 0.0
                if not pathway_row.empty:
                    adj_pval = pathway_row.iloc[0]['Adjusted P-value']
                    efficacy_score = -np.log10(adj_pval) if adj_pval > 0 else 50.0
                
                drug_scores[pathway] = efficacy_score

            all_results.append(drug_scores)
            
        except Exception as e:
            print(f"Warning: Enrichment failed for {target_gene}. Error: {e}")
            continue

    if not all_results:
        print("\n[ERROR] Analysis failed to produce results.")
        return

    # --- Step 2: Create the Heatmap Data ---
    results_df = pd.DataFrame(all_results)
    results_df.set_index('drug_target', inplace=True)
    
    # Clean up the results: drop pathways that had no hits across all drugs
    results_df = results_df.loc[:, (results_df != 0).any(axis=0)]
    
    # Sort the dataframe for better visualization
    # Sort columns by their total score (most active pathways first)
    results_df = results_df[results_df.sum().sort_values(ascending=False).index]
    # Sort rows (drugs) by their highest score
    results_df = results_df.loc[results_df.max(axis=1).sort_values(ascending=False).index]

    print("\n--- Multi-Pathway Screen Results ---")
    print(results_df.head(15))
    results_df.to_csv(os.path.join(args.output_dir, 'multi_pathway_screen_results.csv'))
    
    # --- Step 3: Generate the Heatmap ---
    print("\nGenerating final heatmap...")
    
    fig_height = max(10, 0.4 * len(results_df))
    plt.figure(figsize=(12, fig_height))
    
    sns.heatmap(results_df, cmap="viridis", linewidths=.5, annot=True, fmt=".2f")
    
    plt.title("Multi-Pathway Efficacy Screen for Drug Targets in K562", fontsize=16, weight='bold')
    plt.xlabel("Key Biological Pathways", fontsize=12)
    plt.ylabel("Drug (Target)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'multi_pathway_heatmap.png'), dpi=300)
    
    print(f"\n✓ Multi-pathway analysis complete. Heatmap saved to '{args.output_dir}'")

if __name__ == "__main__":
    main()