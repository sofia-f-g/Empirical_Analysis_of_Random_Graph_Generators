import matplotlib.pyplot as plt
import powerlaw
import numpy as np

def plot_rigorous_ccdf(indegree_seq, outdegree_seq, gamma, beta, save_path=None):
    """
    Generates a mathematically rigorous CCDF plot using the optimal fit parameters,
    overlaying theoretical models over the empirical data strictly from xmin onward.
    """
    
    in_data = np.array(indegree_seq)[np.array(indegree_seq) > 0]
    out_data = np.array(outdegree_seq)[np.array(outdegree_seq) > 0]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Instantiate fits specifically for visual mapping
    fit_in = powerlaw.Fit(in_data, discrete=True, estimate_discrete=True)
    fit_out = powerlaw.Fit(out_data, discrete=True, estimate_discrete=True)
    
    # 1. Plot the empirical data CCDF (Encompassing the entire dataset)
    fit_in.plot_ccdf(ax=ax, color='b', linewidth=2, label='Empirical Indegree')
    fit_out.plot_ccdf(ax=ax, color='r', linewidth=2, linestyle='--', label='Empirical Outdegree')
    
    # 2. Plot the optimal Power-Law fit starting strictly from xmin
    fit_in.power_law.plot_ccdf(ax=ax, color='b', linestyle=':', 
                               label=rf'PL Fit ($\alpha$={fit_in.alpha:.2f}, $x_{{min}}$={fit_in.xmin})')
    
    # 3. Plot a competing Lognormal fit for distinct visual contrast in the tail
    fit_in.lognormal.plot_ccdf(ax=ax, color='g', linestyle='-.', 
                               label='Lognormal Alternative')
                               
    # Theoretical expectation calculation for contextual comparison
    alpha_theory = 1 + (1 / gamma)
    
    ax.set_title(rf"Rigorous Degree Distribution Analysis ($\gamma={gamma}$, $\beta={beta}$)")
    ax.set_xlabel("Degree ($k$)")
    ax.set_ylabel("$P(D \geq k)$")
    
    # CORRECTED: Use empty lists [] instead of empty commas ,,
    ax.plot([], [], ' ', label=rf'Theoretical Indegree $\alpha$: {alpha_theory:.2f}')
    
    ax.legend(loc='lower left', frameon=False)
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        return None
    
    return fig