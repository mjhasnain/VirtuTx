# check_leakage.py (Simplified)
# PURPOSE: To verify that the perturbed genes in the test set were not present
# in the training set, ensuring a valid "zero-shot" evaluation.

import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Check for gene perturbation leakage between train and test sets.")
    parser.add_argument('--train_file', type=str, default='train_data.tsv', 
                        help='Path to the training set TSV file.')
    parser.add_argument('--test_file', type=str, default='val_data.tsv', 
                        help='Path to the test set TSV file.')
    args = parser.parse_args()

    print(f"--- Loading Data for Comparison ---")
    try:
        train_df = pd.read_csv(args.train_file, sep='\t')
        test_df = pd.read_csv(args.test_file, sep='\t')
        print("✓ Train and test data loaded successfully.")
    except FileNotFoundError as e:
        print(f"[ERROR] Could not find a data file: {e}")
        return

    print("\n--- Performing Leakage Check ---")

    # Get the set of unique genes the model was trained on.
    trained_genes = set(train_df['Perturbed_Gene'].unique())
    
    # Get the set of unique genes present in the test set.
    test_genes = set(test_df['Perturbed_Gene'].unique())
    
    # Find the intersection (genes present in both sets).
    leaked_genes = trained_genes.intersection(test_genes)
    
    # In a correct "unseen gene" split, the only gene that should be in both sets is 'Control'.
    expected_overlap = {'Control'}

    # Check if the actual overlap is exactly what we expect.
    if leaked_genes == expected_overlap:
        print("\n========================================")
        print("  [PASS] No data leakage detected.")
        print("         The only shared perturbation between train and test is 'Control', which is correct.")
        print("         Your test set is valid for evaluating generalization to unseen genes.")
        print("========================================")
    else:
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("  [FAIL] Data leakage detected!")
        
        leaked_perturbed_genes = leaked_genes - expected_overlap
        if leaked_perturbed_genes:
            print("         The following perturbed genes from the test set were also found in the training set:")
            print(f"         {leaked_perturbed_genes}")
            print("\n         This means your evaluation is NOT a true test of generalization.")
            print("         You should re-run the `prepare_data_and_signatures.py` script.")
        else:
            # This case handles if 'Control' is missing from one of the sets, which is also an issue.
            print("         The overlap between train and test genes is not as expected.")
            print(f"         Expected overlap: {expected_overlap}")
            print(f"         Actual overlap: {leaked_genes}")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    print("\nVerification complete.")

if __name__ == '__main__':
    main()