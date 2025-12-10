# VirtuTx
================================================================================
 VirtuTx: A Generative AI Platform for Zero-Shot Functional Genomics
 README.txt
================================================================================

This file provides the complete, step-by-step workflow to run the VirtuTx project.

--------------------------------------------------------------------------------
 PART I: SETUP AND DATA PREPARATION
--------------------------------------------------------------------------------

### STEP 0: ENVIRONMENT SETUP

# PURPOSE: To create the Conda environment with all required packages.
# INSTRUCTIONS:
#   1. Place all code (.py), environment.yml, and initial datasets in one folder.
#   2. Run the following commands in your terminal.

conda env create -f environment.yml
conda activate gpu_env


### STEP 1: DATA CURATION AND SIGNATURE GENERATION

# PURPOSE: To process raw data, perform QC, calculate LFC signatures, and
#          create the 80/10/10 gene-held-out data split.
# SCRIPT: prepare_data_and_signatures.py

python prepare_data_and_signatures.py


### STEP 2: VERIFY DATA SPLIT INTEGRITY

# PURPOSE: To confirm that no test set genes have leaked into the training set.
# SCRIPT: check_leakage.py

python check_leakage.py


--------------------------------------------------------------------------------
 PART II: CORE MODEL TRAINING
--------------------------------------------------------------------------------

### STEP 3: TRAIN THE CONDITIONAL VAE (STAGE 1)

# PURPOSE: To train the VAE, which compresses the 2000-gene data into a
#          128-dimensional latent space.
# SCRIPT: train_vae.py

python train_vae.py --output_dir ./final_model_run


### STEP 4: TRAIN THE LATENT DIFFUSION MODEL (STAGE 2)

# PURPOSE: To train the main Transformer-based generative model.
# SCRIPT: train.py

python train.py --output_dir ./final_model_run


--------------------------------------------------------------------------------
 PART III: MODEL VALIDATION
--------------------------------------------------------------------------------

### STEP 5: QUANTITATIVE ZERO-SHOT EVALUATION

# PURPOSE: To assess the model's performance across the entire test set
#          using population-level metrics.
# SCRIPT: evaluate_zero_shot.py

python evaluate_zero_shot.py


### STEP 6: DEEP QUALITATIVE ZERO-SHOT VALIDATION

# PURPOSE: To generate a detailed, multi-panel figure comparing real vs.
#          generated cells for a single unseen gene.
# SCRIPT: validate_zeroshot.py
# EXAMPLE COMMAND:

python validate_zeroshot.py --gene UPF3B  --cell_line K562 --output_dir ./validation_GATA1


--------------------------------------------------------------------------------
 PART IV: DOWNSTREAM APPLICATIONS & THERAPEUTIC SCREENING
--------------------------------------------------------------------------------

### STEP 7: PATHWAY DISCOVERY VIA VIRTUAL KNOCKOUT (SINGLE GENE MoA)

# PURPOSE: To predict the functional consequences (MoA) of knocking out any
#          single gene of interest in a true zero-shot manner.
# SCRIPT: zero_shot_virtual_knockout.py
# EXAMPLE COMMAND:

python zero_shot_virtual_knockout.py --gene TP53 --cell_line K562 --num_cells 8000


### STEP 8: PRE-COMPUTATION FOR LARGE-SCALE SCREENING

# PURPOSE: To efficiently pre-compute the data needed for all subsequent
#          therapeutic screening analyses.

# 8a. Generate a reusable Control cell population
# SCRIPT: generate_control.py
python generate_control.py --cell_line K562 --num_cells 5000

# 8b. Generate a library of LFC signatures for all FDA-approved drug targets
# SCRIPT: run_moa_generation.py
python run_moa_generation.py --drug_list_csv drug_screen_experiments.csv --cell_line K562


### STEP 9: MULTI-PATHWAY EFFICACY SCREENING (DRUG MoA PROFILING)

# PURPOSE: To analyze the pre-computed drug signatures and determine the
#          efficacy of each drug against a curated panel of key cancer pathways.
#          This provides a broad MoA profile for every drug in the library.
# SCRIPT: analyze_multi_pathway.py
# EXAMPLE COMMAND:

python analyze_multi_pathway.py \
  --signatures_dir ./moa_signatures \
  --output_dir ./multi_pathway_analysis_results

# OUTPUT: A comprehensive heatmap showing the predicted efficacy of each drug
#         against pathways like 'Apoptosis', 'p53 signaling', etc.


### STEP 10: DRUG REPURPOSING SCREEN (CONNECTIVITY MAPPING)

# PURPOSE: An analysis script that performs a drug repurposing screen by finding
#          drug signatures that are anti-correlated with a disease signature.
# SCRIPT: analyze_repurposing_screen.py
# EXAMPLE COMMAND:

python analyze_repurposing_screen.py \
  --disease_gene "TET2" \
  --cell_line "K562" \
  --drug_list_csv "drug_screen_experiments.csv" \
  --output_dir "./TET2_repurposing_results"


### STEP 11: UNIFIED THERAPEUTIC SCREENING (MAIN APPLICATION)

# PURPOSE: The final, full-service screening script to find candidates for BOTH
#          synthetic lethality OR genetic rescue, with automated figure generation.
# SCRIPT: run_lfc_screen_final.py
# INSTRUCTIONS:
#   - This is the recommended script for all final connectivity-based screening.
#   - Ensure your GPU is specified, e.g., by setting CUDA_VISIBLE_DEVICES.
#   - Choose 'lethality' or 'rescue' for the --screen_type argument.

# EXAMPLE COMMAND FOR SYNTHETIC LETHALITY:
CUDA_VISIBLE_DEVICES=0 python run_lfc_screen_final.py \
  --disease_gene "TET2" \
  --cell_line "K562" \
  --screen_type "lethality" \
  --output_dir "./TET2_lethality_results"

# EXAMPLE COMMAND FOR GENETIC RESCUE:
CUDA_VISIBLE_DEVICES=0 python run_lfc_screen_final.py \
  --disease_gene "GATA1" \
  --cell_line "K562" \
  --screen_type "rescue" \
  --output_dir "./GATA1_rescue_results"

# OUTPUT: The script will create the specified output directory and save the
#         ranked results CSV file and two publication-quality figures inside it.

================================================================================
 END OF WORKFLOW
================================================================================
