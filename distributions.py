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

def theoretical_indegree_alpha(gamma):
    """
    Theoretical PMF exponent for the indegree distribution.

    The Gracar et al. theory gives

        P(D_in = k) ~ k^(-alpha),

    with

        alpha = 1 + 1/gamma.

    Valid for 0 < gamma < 1.
    """
    gamma = float(gamma)

    if not (0.0 < gamma < 1.0):
        return float("nan")

    return 1.0 + 1.0 / gamma


def theoretical_indegree_ccdf_slope(gamma):
    """
    Theoretical slope for an indegree CCDF log-log plot.

    If

        P(D_in = k) ~ k^(-alpha),

    then

        P(D_in >= k) ~ k^(-(alpha - 1)).

    Since alpha = 1 + 1/gamma, the CCDF slope is

        -(alpha - 1) = -1/gamma.
    """
    gamma = float(gamma)

    if not (0.0 < gamma < 1.0):
        return float("nan")

    return -1.0 / gamma


def add_theoretical_indegree_ccdf_line(
    ax,
    indegree,
    gamma,
    anchor_quantile=0.75,
    color="black",
    linestyle="--",
    linewidth=2,
):
    """
    Add a theoretical CCDF slope guide to an existing indegree CCDF plot.

    This does not fit the data. It only draws a reference line with the
    theoretical slope

        -1/gamma.

    The line is anchored at an empirical CCDF point so that it appears
    on the same scale as the observed data.
    """
    indegree = np.asarray(indegree, dtype=int)

    positive = indegree[indegree > 0]

    if len(positive) == 0:
        return None

    max_k = int(np.max(positive))

    if max_k <= 1:
        return None

    gamma = float(gamma)
    alpha_theory = theoretical_indegree_alpha(gamma)
    ccdf_slope = theoretical_indegree_ccdf_slope(gamma)

    if not np.isfinite(alpha_theory) or not np.isfinite(ccdf_slope):
        return None

    # Choose an anchor point in the upper part of the empirical distribution.
    # This controls only where the guide line is placed vertically.
    anchor_k = int(np.percentile(positive, 100 * anchor_quantile))
    anchor_k = max(anchor_k, 1)
    anchor_k = min(anchor_k, max_k - 1)

    anchor_y = np.mean(indegree >= anchor_k)

    if anchor_y <= 0:
        return None

    k_ref = np.arange(anchor_k, max_k + 1)

    if len(k_ref) < 2:
        return None

    # Reference line:
    #     y = anchor_y * (k / anchor_k)^slope
    y_ref = anchor_y * (k_ref / anchor_k) ** ccdf_slope

    ax.plot(
        k_ref,
        y_ref,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=fr"theory CCDF slope $-1/\gamma={ccdf_slope:.2f}$",
    )

    ax.text(
        0.04,
        0.04,
        fr"$\alpha_{{theory}}=1+1/\gamma={alpha_theory:.2f}$"
        + "\n"
        + fr"CCDF slope $=-1/\gamma={ccdf_slope:.2f}$",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    return {
        "alpha_theory": alpha_theory,
        "ccdf_slope_theory": ccdf_slope,
        "anchor_k": anchor_k,
        "anchor_y": anchor_y,
    }



def plot_degree_distributions(params, n, seed=0, bins="auto", show_theory=True):
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
    
    if show_theory:
        add_theoretical_indegree_ccdf_line(
            axes[2],
            indegree=indegree,
            gamma=params["gamma"],
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


def plot_degree_distributions_grid(param_list, n, seed=0, bins="auto", row_label="gamma", show_theory=True):
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

        if show_theory:
            add_theoretical_indegree_ccdf_line(
                axes[row_idx, 2],
                indegree=indegree,
                gamma=params["gamma"],
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

if __name__ == "__main__":
    n = 1000
    seed = 0

    output_dir = Path("distribution_plots/")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_params = {
        "beta": 1.0,
        "gamma": 0.5,
        "dim": 2,
        "space_cfg": {"bounds": [[-0.5, 0.5], [-0.5, 0.5]]},
        "age_cfg": {"min": 0.0, "max": float(n)},
        "profile_cfg": {},
    }

    # ---------------------------------
    # Figure 1: fixed beta, varying gamma
    # ---------------------------------
    fixed_beta = 1.0
    gamma_values = [0.2, 0.5, 0.8]

    params_fixed_beta = []
    for gamma in gamma_values:
        params = base_params.copy()
        params["beta"] = fixed_beta
        params["gamma"] = gamma
        params_fixed_beta.append(params)

    fig1 = plot_degree_distributions_grid(
        params_fixed_beta,
        n=n,
        seed=seed,
        row_label="gamma",
    )

    path1 = output_dir / f"version2_degree_diagnostics_fixed_beta{fixed_beta}_vary_gamma_n{n}_seed{seed}.png"
    fig1.savefig(path1, dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {path1}")
    plt.close(fig1)

    # ---------------------------------
    # Figure 2: fixed gamma, varying beta
    # ---------------------------------
    fixed_gamma = 0.5
    beta_values = [0.5, 1.0, 2.0]

    params_fixed_gamma = []
    for beta in beta_values:
        params = base_params.copy()
        params["beta"] = beta
        params["gamma"] = fixed_gamma
        params_fixed_gamma.append(params)

    fig2 = plot_degree_distributions_grid(
        params_fixed_gamma,
        n=n,
        seed=seed,
        row_label="beta",
    )

    path2 = output_dir / f"version2_degree_diagnostics_fixed_gamma{fixed_gamma}_vary_beta_n{n}_seed{seed}.png"
    fig2.savefig(path2, dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {path2}")
    plt.close(fig2)