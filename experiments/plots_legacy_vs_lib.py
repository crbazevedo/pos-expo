"""
Plotting script for POS-Expo Legacy vs Lib benchmarks.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "legacy_vs_lib.csv")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

def plot_legacy_comparison(df):
    """
    Generates comparison plots for each scenario.
    """
    scenarios = df["scenario"].unique()
    
    for scenario in scenarios:
        subset = df[df["scenario"] == scenario]
        
        # Prepare data for plotting
        models = subset["model"].unique()
        n_models = len(models)
        
        legacy_means = []
        lib_means = []
        
        for model in models:
            row = subset[subset["model"] == model].iloc[0]
            legacy_means.append(row["test_loss_legacy"])
            lib_means.append(row["test_loss_lib"])
            
        x = np.arange(n_models)
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(8, 5))
        rects1 = ax.bar(x - width/2, legacy_means, width, label='Legacy', color='gray', alpha=0.7)
        rects2 = ax.bar(x + width/2, lib_means, width, label='Library', color='orange', alpha=0.7)
        
        ax.set_ylabel('Test Log Loss')
        ax.set_title(f'Legacy vs Library Comparison - {scenario}')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add labels
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.4f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

        autolabel(rects1)
        autolabel(rects2)
        
        plt.tight_layout()
        filename = f"fig_{scenario.lower()}_comparison_legacy_vs_lib.png"
        # Map scenario names to filenames if needed
        if scenario == "1D_Piecewise":
            filename = "fig_synthetic_risk_curves_legacy_vs_lib.png" # User asked for this name
        elif scenario == "Adult":
            filename = "fig_adult_comparison_legacy_vs_lib.png"
            
        save_path = os.path.join(FIGURES_DIR, filename)
        plt.savefig(save_path)
        print(f"Saved {save_path}")
        plt.close()

def plot_poset_layers_schematic():
    """
    Generates a schematic for poset layers (using random data for illustration).
    """
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
    save_path = os.path.join(FIGURES_DIR, "fig_poset_layers.png")
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close()

def generate_pipeline_diagram():
    """
    Generates a simple pipeline block diagram.
    """
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
    save_path = os.path.join(FIGURES_DIR, "fig_pos_expo_pipeline.png")
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close()

def main():
    if not os.path.exists(RESULTS_FILE):
        print(f"Results file not found: {RESULTS_FILE}")
        return
        
    df = pd.read_csv(RESULTS_FILE)
    plot_legacy_comparison(df)
    plot_poset_layers_schematic()
    generate_pipeline_diagram()

if __name__ == "__main__":
    main()

