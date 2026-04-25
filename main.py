from pathlib import Path

from matplotlib import pyplot as plt

from distributions import plot_degree_distributions_grid
import run
import numpy as np

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


# if __name__ == "__main__":
#     n, R, base_seed = 200, 3, 0

#     eps = np.nextafter(0.0, 1.0)

#     beta_cfg = {"min": 0.1, "max": 2.0, "step": 0.3}    # edit step to change beta resolution
#     gamma_cfg = {"min": 0.1, "max": 0.9, "step": 0.2}   # edit step to change gamma resolution KAN INTE VARA 1 ENLIGT TEORIN!!

#     base_params = {
#         "beta":       beta_cfg["min"],
#         "gamma":      gamma_cfg["min"],
#         "dim":        2,
#         "space_cfg":  {"bounds": [[-0.5, 0.5], [-0.5, 0.5]]},
#         "age_cfg":    {"min": 0.0, "max": float(n)},
#         "profile_cfg": {"type": "normalized_cutoff"},
#     }

#     ranges_dict = {
#         "beta":  build_range(beta_cfg["min"], beta_cfg["max"], beta_cfg["step"]),
#         "gamma": build_range(gamma_cfg["min"], gamma_cfg["max"], gamma_cfg["step"]),
#     }


#     param_grid = run.make_param_grid(ranges_dict, base_params)
#     print(f"Running {len(param_grid)} parameter combinations x {R} replicates = {len(param_grid) * R} simulations")


#     result_table, run_label = run.parameter_sweep(param_grid, n, R, base_seed)
#     summary_table = run.summarise_over_replicates(result_table, run_label)
#     print(f"Done. Results saved under label: {run_label}")

#     metrics_to_plot = [
#         "n_edges_mean",
#         "avg_shortest_path_length_mean",

#         "avg_local_clustering_mean",
#         "global_clustering_mean",

#         "outdegree_mean_vs_theory_ratio",
#         "expected_theoretical_outdegree_mean",
#         "outdegree_mean_mean",
#         "indegree_mean_mean",
#         "outdegree_max_mean",
#         "indegree_max_mean",

        
#     ]
#     plot_path = run.plot_metric_panels(summary_table, metrics_to_plot, x_param="beta", y_param="gamma", run_label=run_label)
#     if plot_path:
#         print(f"Saved visualization panels to {plot_path}")

if __name__ == "__main__":
    n = 2000
    seed = 0

    gamma_values = [0.2, 0.4, 0.6, 0.8]

    param_list = []
    for gamma in gamma_values:
        param_list.append({
            "beta": 0.5,
            "gamma": gamma,
            "dim": 2,
            "space_cfg": {"bounds": [[-0.5, 0.5], [-0.5, 0.5]]},
            "age_cfg": {"min": 0.0, "max": float(n)},
            "profile_cfg": {"type": "normalized_cutoff"},
        })

    fig = plot_degree_distributions_grid(
        param_list=param_list,
        n=n,
        seed=seed,
        bins="auto",
        row_label="gamma",
        show_theory=True,
    )

    output_dir = Path("distribution_plots")
    output_dir.mkdir(exist_ok=True)

    fig.savefig(
        output_dir / "degree_ccdf_grid_varying_gamma_with_theory.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)