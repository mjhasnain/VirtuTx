# metrics.py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from scipy.linalg import sqrtm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

def calculate_mmd(real_data, generated_data, gamma=1.0):
    from sklearn.metrics.pairwise import rbf_kernel
    XX = rbf_kernel(real_data, real_data, gamma).mean()
    YY = rbf_kernel(generated_data, generated_data, gamma).mean()
    XY = rbf_kernel(real_data, generated_data, gamma).mean()
    return XX + YY - 2 * XY

def calculate_frechet_distance(real_data, generated_data):
    try:
        pca = PCA(n_components=min(50, real_data.shape[1])).fit(real_data)
        real_pca = pca.transform(real_data); gen_pca = pca.transform(generated_data)
        mu_real, sigma_real = real_pca.mean(axis=0), np.cov(real_pca.T)
        mu_gen, sigma_gen = gen_pca.mean(axis=0), np.cov(gen_pca.T)
        diff = mu_real - mu_gen; cov_mean = sqrtm(sigma_real.dot(sigma_gen))
        if np.iscomplexobj(cov_mean): cov_mean = cov_mean.real
        return diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * cov_mean)
    except Exception: return np.nan

def marker_gene_correlation(real_data_df, generated_data_df, marker_genes_indices):
    real_means = real_data_df.iloc[:, marker_genes_indices].mean(axis=0)
    gen_means = generated_data_df.iloc[:, marker_genes_indices].mean(axis=0)
    corr, _ = pearsonr(real_means, gen_means); return corr

def classifier_accuracy_gap(real_data, generated_data, labels):
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    X_train, X_test, y_train, y_test = train_test_split(real_data, labels, test_size=0.3, random_state=42, stratify=labels)
    clf.fit(X_train, y_train); real_accuracy = clf.score(X_test, y_test)
    gen_accuracy = clf.score(generated_data, labels)
    return real_accuracy - gen_accuracy

def calculate_perturbation_correlation(real_data_df, generated_data_df, metadata, perturbation):
    real_control = real_data_df[metadata['Perturbed_Gene'] == 'Control'].values
    real_perturbed = real_data_df[metadata['Perturbed_Gene'] == perturbation].values
    generated_perturbed = generated_data_df[metadata['Perturbed_Gene'] == perturbation].values
    
    if len(real_control) == 0 or len(real_perturbed) == 0 or len(generated_perturbed) == 0: return np.nan
    
    pseudo_count = 1e-6
    real_lfc = np.log2((real_perturbed.mean(axis=0) + pseudo_count) / (real_control.mean(axis=0) + pseudo_count))
    gen_lfc = np.log2((generated_perturbed.mean(axis=0) + pseudo_count) / (real_control.mean(axis=0) + pseudo_count))
    
    valid_indices = np.isfinite(real_lfc) & np.isfinite(gen_lfc)
    if np.sum(valid_indices) < 2: return np.nan

    corr, _ = pearsonr(real_lfc[valid_indices], gen_lfc[valid_indices]); return corr

def compute_all_metrics(real_profiles, generated_profiles, metadata, gene_cols, num_perturbations_to_test=20):
    results = {}
    results['mmd'] = calculate_mmd(real_profiles, generated_profiles)
    results['frechet'] = calculate_frechet_distance(real_profiles, generated_profiles)
    real_df = pd.DataFrame(real_profiles, columns=gene_cols); gen_df = pd.DataFrame(generated_profiles, columns=gene_cols)
    marker_gene_indices = list(range(min(10, len(gene_cols))))
    results['marker_corr'] = marker_gene_correlation(real_df, gen_df, marker_gene_indices)
    results['accuracy_gap'] = classifier_accuracy_gap(real_profiles, generated_profiles, metadata['Cell_Line'].values)
    
    perturbations = [p for p in metadata['Perturbed_Gene'].unique() if p != 'Control']
    pert_corrs = []
    for pert in perturbations[:num_perturbations_to_test]:
        corr = calculate_perturbation_correlation(real_df, gen_df, metadata, pert)
        if not np.isnan(corr): pert_corrs.append(corr)
    results['perturbation_corr'] = np.mean(pert_corrs) if pert_corrs else 0.0
    return results