
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_lambda_behavior(stats_list, save_dir='analysis/plots'):
    """
    Plots Lambda values vs Episode Difficulty metrics.
    
    Args:
        stats_list: List of dicts, each containing:
            'lambda_var', 'lambda_cov', 'intra_var', 'inter_sep', 'domain_shift', 'support_div', 'query_div', 'acc'
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    df = pd.DataFrame(stats_list)
    
    # 1. Lambda vs Intra-class Variance (Difficulty)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='intra_var', y='lambda_var', hue='acc', palette='viridis', alpha=0.7)
    plt.title('Adaptivity of $\lambda_{var}$ to Intra-Class Variance')
    plt.xlabel('Intra-Class Variance (Support)')
    plt.ylabel('Predicted $\lambda_{var}$')
    plt.savefig(os.path.join(save_dir, 'lambda_var_vs_intra_var.png'))
    plt.close()

    # 2. Lambda vs Domain Shift
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='domain_shift', y='lambda_var', hue='acc', palette='viridis', alpha=0.7)
    plt.title('Adaptivity of $\lambda_{var}$ to Domain Shift')
    plt.xlabel('Domain Shift (Cosine Dist)')
    plt.ylabel('Predicted $\lambda_{var}$')
    plt.savefig(os.path.join(save_dir, 'lambda_var_vs_domain_shift.png'))
    plt.close()

    # 3. Lambda Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='lambda_var', kde=True, label='$\lambda_{var}$')
    sns.histplot(data=df, x='lambda_cov', kde=True, label='$\lambda_{cov}$', color='orange')
    plt.title('Distribution of Predicted Lambdas')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'lambda_distribution.png'))
    plt.close()

def plot_static_vs_dynamic_gap(dynamic_stats, static_stats, save_dir='analysis/plots'):
    """
    Plots the performance gap between Dynamic and Static models across difficulty.
    
    Args:
        dynamic_stats: List of dicts from dynamic model
        static_stats: List of dicts from static model (must be same episodes!)
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Combine into dataframe
    df_dyn = pd.DataFrame(dynamic_stats)
    df_stat = pd.DataFrame(static_stats)
    
    # Ensure aligned (assuming lists are in order of episodes)
    df_dyn['model'] = 'Dynamic'
    df_stat['model'] = 'Static'
    
    # Calculate gap per episode
    gap = df_dyn['acc'] - df_stat['acc']
    
    # Define "Difficulty" as Intra-Class Variance (or Loss)
    # We use dynamic model's view of difficulty for x-axis
    difficulty = df_dyn['intra_var']
    
    # Bin difficulty
    bins = np.linspace(difficulty.min(), difficulty.max(), 6)
    labels = [f'{bins[i]:.2f}-{bins[i+1]:.2f}' for i in range(5)]
    df_dyn['diff_bin'] = pd.cut(difficulty, bins=bins, labels=labels, include_lowest=True)
    
    # Calculate Mean Acc per bin per model
    # We need to act carefully to align. 
    # Let's create a combined DF for seaborn
    df_combined = pd.DataFrame({
        'Difficulty': pd.concat([df_dyn['diff_bin'], df_dyn['diff_bin']]),
        'Accuracy': pd.concat([df_dyn['acc'], df_stat['acc']]),
        'Model': ['Dynamic']*len(df_dyn) + ['Static']*len(df_stat)
    })
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_combined, x='Difficulty', y='Accuracy', hue='Model')
    plt.title('Performance vs Episode Difficulty (Intra-Class Var)')
    plt.xlabel('Episode Difficulty (Intra-Class Variance)')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(save_dir, 'static_vs_dynamic_difficulty.png'))
    plt.close()
    
    # Plot Gap Curve
    # Calculate mean gap per bin
    df_gap = pd.DataFrame({
        'Difficulty': df_dyn['diff_bin'],
        'Gain': gap
    })
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_gap, x='Difficulty', y='Gain', marker='o')
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Dynamic Model Gain over Static vs Difficulty')
    plt.ylabel('Accuracy Gain (Dynamic - Static)')
    plt.savefig(os.path.join(save_dir, 'dynamic_gain_curve.png'))
    plt.close()
