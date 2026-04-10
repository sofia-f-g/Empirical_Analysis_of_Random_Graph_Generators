import run

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


if __name__ == "__main__":
    beta_cfg = {"min": 0.01, "max": 0.8, "step": 0.1}    # edit step to change beta resolution
    gamma_cfg = {"min": 0.0, "max": 1.0, "step": 0.1}   # edit step to change gamma resolution

    base_params = {
        "beta":       beta_cfg["min"],
        "gamma":      gamma_cfg["min"],
        "dim":        2,
        "space_cfg":  {"bounds": [[-0.5, 0.5], [-0.5, 0.5]]},
        "age_cfg":    {"min": 0.0, "max": 1.0},
        "profile_cfg": {"a": 0.5},
    }

    ranges_dict = {
        "beta":  build_range(beta_cfg["min"], beta_cfg["max"], beta_cfg["step"]),
        "gamma": build_range(gamma_cfg["min"], gamma_cfg["max"], gamma_cfg["step"]),
    }

    """ Changed to hard-coding for interesting regions 
    fine_beta = [0.05, 0.08, 0.11, 0.14, 0.17, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]
    fine_gamma = [0.00, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90, 1.00]

    ranges_dict = {
        "beta":  fine_beta,
        "gamma": fine_gamma,
    }
    """


    n, R, base_seed = 1000, 10, 0

    param_grid = run.make_param_grid(ranges_dict, base_params)
    print(f"Running {len(param_grid)} parameter combinations x {R} replicates = {len(param_grid) * R} simulations")


    result_table, run_label = run.parameter_sweep(param_grid, n, R, base_seed)
    summary_table = run.summarise_over_replicates(result_table, run_label)
    print(f"Done. Results saved under label: {run_label}")

    metrics_to_plot = [
        "n_edges_mean",
        "avg_local_clustering_mean",
        "avg_shortest_path_length_mean",
        "degree_mean_mean",
        "global_clustering_mean",
    ]
    plot_path = run.plot_metric_panels(summary_table, metrics_to_plot, x_param="beta", y_param="gamma", run_label=run_label)
    if plot_path:
        print(f"Saved visualization panels to {plot_path}")
