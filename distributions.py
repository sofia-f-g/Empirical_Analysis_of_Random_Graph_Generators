import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import generator
import metrics

def get_directed_degree_arrays(params, n, seed=0):
    """
    Generate one graph and return indegree, outdegree, and total degree arrays.
    """
    V, E = generator.generate_graph(params, n, seed=seed)
    deg = metrics.compute_degree_sequences(V, E)
    return deg["indegree"], deg["outdegree"], deg["total_degree"]

def empirical_ccdf(values):
    """
    Compute empirical CCDF P(X >= k) for an integer-valued array.
    Returns (k_values, ccdf_values).
    """
    values = np.asarray(values, dtype=int)
    if len(values) == 0:
        return np.array([]), np.array([])

    k_values = np.arange(values.min(), values.max() + 1)
    ccdf = np.array([np.mean(values >= k) for k in k_values], dtype=float)
    return k_values, ccdf

def plot_degree_distributions(params, n, seed=0, bins="auto"):
    """
    Make three plots for one graph realization:
    1. indegree histogram
    2. outdegree histogram
    3. indegree vs outdegree CCDF on log-log axes
    """
    indegree, outdegree, total_degree = get_directed_degree_arrays(params, n, seed=seed)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # 1. Indegree histogram
    axes[0].hist(indegree, bins=bins)
    axes[0].set_title("Indegree Histogram")
    axes[0].set_xlabel("indegree")
    axes[0].set_ylabel("count")

    # 2. Outdegree histogram
    axes[1].hist(outdegree, bins=bins)
    axes[1].set_title("Outdegree Histogram")
    axes[1].set_xlabel("outdegree")
    axes[1].set_ylabel("count")

    # 3. Indegree vs Outdegree CCDF
    k_in, ccdf_in = empirical_ccdf(indegree)
    k_out, ccdf_out = empirical_ccdf(outdegree)

    mask_in = (k_in > 0) & (ccdf_in > 0)
    mask_out = (k_out > 0) & (ccdf_out > 0)

    axes[2].plot(
        k_in[mask_in],
        ccdf_in[mask_in],
        marker="o",
        linestyle="none",
        label="indegree"
    )
    axes[2].plot(
        k_out[mask_out],
        ccdf_out[mask_out],
        marker="x",
        linestyle="none",
        label="outdegree"
    )

    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_title("Degree CCDF Comparison (log-log)")
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("P(K >= k)")
    axes[2].legend()

    beta = params["beta"]
    gamma = params["gamma"]
    fig.suptitle(f"Degree diagnostics: n={n}, beta={beta}, gamma={gamma}, seed={seed}")
    fig.tight_layout()

    return fig


def plot_degree_distributions_grid(param_list, n, seed=0, bins="auto", row_label="gamma"):
    """
    For a list of parameter settings, create one combined figure with one row per setting
    and three columns:
      1. indegree histogram
      2. outdegree histogram
      3. indegree vs outdegree CCDF (log-log)
    """
    nrows = len(param_list)
    fig, axes = plt.subplots(nrows, 3, figsize=(16, 4 * nrows), squeeze=False)

    for row_idx, params in enumerate(param_list):
        indegree, outdegree, total_degree = get_directed_degree_arrays(params, n, seed=seed)

        label_value = params[row_label]

        # 1. Indegree histogram
        axes[row_idx, 0].hist(indegree, bins=bins)
        axes[row_idx, 0].set_title(f"Indegree Histogram ({row_label}={label_value})")
        axes[row_idx, 0].set_xlabel("indegree")
        axes[row_idx, 0].set_ylabel("count")

        # 2. Outdegree histogram
        axes[row_idx, 1].hist(outdegree, bins=bins)
        axes[row_idx, 1].set_title(f"Outdegree Histogram ({row_label}={label_value})")
        axes[row_idx, 1].set_xlabel("outdegree")
        axes[row_idx, 1].set_ylabel("count")

        # 3. Indegree vs Outdegree CCDF
        k_in, ccdf_in = empirical_ccdf(indegree)
        k_out, ccdf_out = empirical_ccdf(outdegree)

        mask_in = (k_in > 0) & (ccdf_in > 0)
        mask_out = (k_out > 0) & (ccdf_out > 0)

        axes[row_idx, 2].plot(
            k_in[mask_in],
            ccdf_in[mask_in],
            marker="o",
            linestyle="none",
            label="indegree"
        )
        axes[row_idx, 2].plot(
            k_out[mask_out],
            ccdf_out[mask_out],
            marker="x",
            linestyle="none",
            label="outdegree"
        )

        axes[row_idx, 2].set_xscale("log")
        axes[row_idx, 2].set_yscale("log")
        axes[row_idx, 2].set_title(f"Degree CCDF Comparison ({row_label}={label_value})")
        axes[row_idx, 2].set_xlabel("k")
        axes[row_idx, 2].set_ylabel("P(K >= k)")
        axes[row_idx, 2].legend()

    # figure headline including fixed/varying beta/gamma
    beta_values = {params["beta"] for params in param_list}
    gamma_values = {params["gamma"] for params in param_list}

    if len(beta_values) == 1 and len(gamma_values) > 1:
        fixed_beta = next(iter(beta_values))
        fig.suptitle(f"Degree diagnostics grid: varying gamma, fixed beta={fixed_beta}, n={n}, seed={seed}")
    elif len(gamma_values) == 1 and len(beta_values) > 1:
        fixed_gamma = next(iter(gamma_values))
        fig.suptitle(f"Degree diagnostics grid: varying beta, fixed gamma={fixed_gamma}, n={n}, seed={seed}")
    elif len(beta_values) == 1 and len(gamma_values) == 1:
        fixed_beta = next(iter(beta_values))
        fixed_gamma = next(iter(gamma_values))
        fig.suptitle(f"Degree diagnostics grid: beta={fixed_beta}, gamma={fixed_gamma}, n={n}, seed={seed}")
    else:
        fig.suptitle(f"Degree diagnostics grid: varying beta and gamma, n={n}, seed={seed}")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
