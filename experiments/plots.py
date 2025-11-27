"""
Plotting script for POS-Expo experiments.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_synthetic_risk(df):
    """
    Generates risk comparison plots for synthetic scenarios.
    """
    scenarios = df[df["scenario"] != "Adult"]["scenario"].unique()
    
    if len(scenarios) == 0:
        return

    fig, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 5))
    if len(scenarios) == 1:
        axes = [axes]
        
    for ax, scenario in zip(axes, scenarios):
        subset = df[df["scenario"] == scenario]
        
        # Compute means and stds
        means = subset[["loss_erm", "loss_iw", "loss_pos"]].mean()
        stds = subset[["loss_erm", "loss_iw", "loss_pos"]].std()
        
        models = ["ERM", "IW*", "POS-Expo"]
        x_pos = range(len(models))
        
        ax.bar(x_pos, [means["loss_erm"], means["loss_iw"], means["loss_pos"]], 
               yerr=[stds["loss_erm"], stds["loss_iw"], stds["loss_pos"]],
               capsize=5, alpha=0.7, color=['gray', 'skyblue', 'orange'])
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models)
        ax.set_ylabel("Test Log-Loss")
        ax.set_title(f"Scenario: {scenario}")
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    plt.savefig("docs/figures/fig_synthetic_risk_curves.png")
    print("Saved docs/figures/fig_synthetic_risk_curves.png")

def plot_adult_comparison(df):
    """
    Generates comparison plot for Adult dataset.
    """
    subset = df[df["scenario"] == "Adult"]
    if subset.empty:
        return
        
    means = subset[["loss_erm", "loss_pos"]].mean()
    stds = subset[["loss_erm", "loss_pos"]].std()
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    models = ["ERM", "POS-Expo"]
    x_pos = range(len(models))
    
    ax.bar(x_pos, [means["loss_erm"], means["loss_pos"]], 
           yerr=[stds["loss_erm"], stds["loss_pos"]],
           capsize=5, alpha=0.7, color=['gray', 'orange'])
           
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models)
    ax.set_ylabel("Test Log-Loss")
    ax.set_title("Adult Dataset (Biased Sample)")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add note if difference is small
    diff = means["loss_erm"] - means["loss_pos"]
    ax.text(0.5, 0.95, f"Gap: {diff:.4f}", transform=ax.transAxes, ha='center')
    
    plt.tight_layout()
    plt.savefig("docs/figures/fig_adult_comparison.png")
    print("Saved docs/figures/fig_adult_comparison.png")

def plot_poset_layers_schematic():
    """
    Generates a schematic for poset layers (using random data for illustration).
    """
    import numpy as np  # Re-import to be safe if global failed, but should be fine
    np.random.seed(42)
    
    # Simulate 2D features
    phi = np.random.rand(100, 2)
    # Simple layer logic: sum of coords
    scores = phi.sum(axis=1)
    layers = np.digitize(scores, bins=np.linspace(0, 2, 5))
    
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(phi[:, 0], phi[:, 1], c=layers, cmap='viridis', s=50, edgecolor='k')
    
    ax.set_xlabel(r"$\phi_1$ (e.g. Rarity)")
    ax.set_ylabel(r"$\phi_2$ (e.g. Difficulty)")
    ax.set_title("Poset Layers (Schematic)")
    plt.colorbar(scatter, label="Layer Depth")
    
    plt.tight_layout()
    plt.savefig("docs/figures/fig_poset_layers.png")
    print("Saved docs/figures/fig_poset_layers.png")

def generate_pipeline_diagram():
    """
    Generates a simple pipeline block diagram.
    """
    import numpy as np
    # We can't easily draw flowcharts with matplotlib, so we'll make a text-based
    # image or a simple block plot. Let's do a simple block plot.
    
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')
    
    stages = ["Raw Data\n(X, y)", "Structural\nFeatures φ(z)", "Fit α\n(Tilt)", "Weights\n w_α(z)", "Weighted\nERM"]
    x_pos = np.linspace(0.1, 0.9, len(stages))
    y_pos = 0.5
    
    # Draw boxes
    for i, (x, stage) in enumerate(zip(x_pos, stages)):
        ax.text(x, y_pos, stage, ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black"), fontsize=10)
        
        if i < len(stages) - 1:
            # Draw arrow
            ax.arrow(x + 0.08, y_pos, (x_pos[i+1] - x) - 0.16, 0, 
                     head_width=0.05, head_length=0.02, fc='k', ec='k')
            
    ax.set_title("POS-Expo Pipeline", fontsize=14)
    plt.tight_layout()
    plt.savefig("docs/figures/fig_pos_expo_pipeline.png")
    print("Saved docs/figures/fig_pos_expo_pipeline.png")

def main():
    results_path = "experiments/results/benchmarks.csv"
    if not os.path.exists(results_path):
        print("No results found. Run benchmark.py first.")
        return
        
    df = pd.read_csv(results_path)
    
    plot_synthetic_risk(df)
    plot_adult_comparison(df)
    plot_poset_layers_schematic()
    generate_pipeline_diagram()

if __name__ == "__main__":
    main()


