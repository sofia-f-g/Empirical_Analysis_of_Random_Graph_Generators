from pathlib import Path

from matplotlib import pyplot as plt

from distributions import plot_degree_distributions_grid
import limit_experiment
import run
import numpy as np
from datetime import datetime

def build_range(start, stop, step):
    if step <= 0:
        raise ValueError("step must be positive")

    values, current = [], float(start)
    stop = float(stop)
    while current <= stop + 1e-12:
        values.append(round(current, 6))
        current += float(step)

    if values[-1] != round(stop, 6):
        values.append(round(stop, 6))
    return values

def build_params(n, beta, gamma):
    return {
        "beta": beta,
        "gamma": gamma,
        "dim": 2,
        "space_cfg": {"bounds": [[-0.5, 0.5], [-0.5, 0.5]]},
        "age_cfg": {"min": 0.0, "max": float(n)},
        "profile_cfg": {"type": "normalized_cutoff"},
    }

def run_degree_distribution_plots(beta_values, gamma_values, n, seed):
    if len(beta_values) > 1 and len(gamma_values) > 1:
        raise ValueError(
            "For degree distribution plots, only one parameter may vary. "
            "Use either one beta with one or more gamma values, "
            "or one gamma with one or more beta values."
        )

    output_dir = Path("distribution_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_stamp = datetime.now().strftime("%d%b_%H%M").lower()

    param_list = []

    for beta in beta_values:
        for gamma in gamma_values:
            param_list.append(build_params(n, beta, gamma))

    if len(beta_values) == 1 and len(gamma_values) == 1:
        row_label = "gamma"
        beta = beta_values[0]
        gamma = gamma_values[0]
        filename = f"{run_stamp}_beta{beta}_gamma{gamma}_n{n}.png"

    elif len(beta_values) == 1:
        row_label = "gamma"
        fixed_beta = beta_values[0]
        filename = f"{run_stamp}_fixed_beta{fixed_beta}_n{n}.png"

    else:
        row_label = "beta"
        fixed_gamma = gamma_values[0]
        filename = f"{run_stamp}_fixed_gamma{fixed_gamma}_n{n}.png"

    fig = plot_degree_distributions_grid(
        param_list=param_list,
        n=n,
        seed=seed,
        row_label=row_label,
        show_theory=True
    )

    path = output_dir / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved degree distribution plot to: {path}")

def run_simulation(beta_cfg, gamma_cfg, n, R, base_seed, metrics_to_plot, hardcoded=False):
    if hardcoded:
        beta_values = beta_cfg
        gamma_values = gamma_cfg

        beta_label = "hardcoded"
        gamma_label = "hardcoded"
    else:
        beta_values = build_range(beta_cfg["min"], beta_cfg["max"], beta_cfg["step"])
        gamma_values = build_range(gamma_cfg["min"], gamma_cfg["max"], gamma_cfg["step"])

        beta_label = beta_cfg["step"]
        gamma_label = gamma_cfg["step"]


    base_params = build_params(
        n=n,
        beta=beta_values[0],
        gamma=gamma_values[0],
    )

    ranges_dict = {
        "beta": beta_values,
        "gamma": gamma_values,
    }


    param_grid = run.make_param_grid(ranges_dict, base_params)
    print(f"Running {len(param_grid)} parameter combinations x {R} replicates = {len(param_grid) * R} simulations")


    result_table, run_label = run.parameter_sweep(param_grid, n, R, base_seed)
    summary_table = run.summarise_over_replicates(result_table, run_label)
    print(f"Done. Results saved under label: {run_label}")

    plot_path = run.plot_metric_panels(
        summary_table,
        metrics_to_plot,
        x_param="beta",
        y_param="gamma",
        run_label=run_label,
        plot_info={
            "n": n,
            "R": R,
            "beta_step": beta_label,
            "gamma_step": gamma_label,
        },
    )
    if plot_path:
        print(f"Saved visualization panels to {plot_path}")

if __name__ == "__main__":
    # -----------------------------
    # General simulation settings
    # -----------------------------  
    n, R, base_seed = 200, 50, 0
    # -----------------------------
    # Choose what to run
    # -----------------------------
    hardcoded = False
    run_heatmaps = False
    run_ccdf_plots = False
    run_clustering_limit = True
    
    # -----------------------------
    # Heat-map / parameter sweep settings
    # -----------------------------
    if not hardcoded:
        beta_cfg = {"min": 50.0, "max": 250.0, "step": 50.0}    # edit step to change beta resolution
        gamma_cfg = {"min": 0.05, "max": 0.95, "step": 0.1}   # edit step to change gamma resolution KAN INTE VARA 1 ENLIGT TEORIN!!
    else:
        beta_cfg = [0.1, 0.3, 0.5]
        gamma_cfg = [0.4, 0.45, 0.5, 0.55, 0.6]

    metrics_to_plot = [
        "n_edges_mean",
        "avg_shortest_path_length_mean",

        "avg_local_clustering_mean",
        "global_clustering_mean",

        "outdegree_mean_vs_theory_ratio",
        "expected_theoretical_outdegree_mean",
        "outdegree_mean_mean",
        "indegree_mean_mean",
        "outdegree_max_mean",
        "indegree_max_mean",
        
    ]

    if run_heatmaps:
        run_simulation(beta_cfg, gamma_cfg, n, R, base_seed, metrics_to_plot, hardcoded=hardcoded)


    # -----------------------------
    # Degree distribution / CCDF settings
    # -----------------------------
    if run_ccdf_plots:
        ccdf_n = 2000
        ccdf_seed = base_seed

        # Valid examples:
        # One beta, multiple gamma:
        # ccdf_beta_values = [3.0]
        # ccdf_gamma_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        # Or multiple beta, one gamma:
        ccdf_beta_values = [3.0, 5.0, 10.0, 20.0, 50.0]
        ccdf_gamma_values = [0.75]

        # Or one beta, one gamma:
        # ccdf_beta_values = [0.5]
        # ccdf_gamma_values = [0.5]

        run_degree_distribution_plots(
            beta_values=ccdf_beta_values,
            gamma_values=ccdf_gamma_values,
            n=ccdf_n,
            seed=ccdf_seed,
        )

    # -----------------------------
# Global clustering limit experiment
# -----------------------------
if run_clustering_limit:
    limit_experiment.run_experiment(
        c_ed=5.0,
        gamma_values=[0.2, 0.4, 0.6, 0.9],
        n_values=[100, 250, 500, 1000, 2000, 5000],
        R=5,
        base_seed=base_seed,
        make_plot=True,
    )
    
