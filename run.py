import results_io  # local results_io.py — save/load raw and summary CSVs
import generator
import metrics
import statistics

import itertools
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np



### Running Simulations ###

def run_one_simulation(params, n, seed):
    generator.validate_params(params)

    (V, E) = generator.generate_graph(params, n, seed)
    metric_row = metrics.compute_metrics(V, E)

    # Attach which parameters generated the results
    metric_row.update({
        "seed": seed,
        "beta": params["beta"],
        "gamma": params["gamma"],
        "dim": params["dim"],
    })
    
    return metric_row

def run_replicates(params, n, R, base_seed):
    """ Repeats run_one_simulation R times with controlled seeding
        to estimate variability. Returns a list of metric dicts,
        one per replicate. """

    rows = []
    for i in range(R):
        row = run_one_simulation(params, n, seed=base_seed + i)
        rows.append(row)
    return rows


### Sweeps + Aggregation ###

def make_param_grid(ranges_dict, base_params):
    """ Creates a list of parameter dicts for sweeps by taking the
        cartesian product of all values in ranges_dict.
        base_params provides the fixed default values for all other parameters.

        Example:
            ranges_dict  = {'beta': [0.3, 0.5], 'gamma': [0.5, 0.8]}
            base_params  = {'beta': 0.5, 'gamma': 0.8, 'dim': 2, ...}
            → 4 parameter dicts, one per combination
    """

    keys = list(ranges_dict.keys())
    value_lists = [ranges_dict[k] for k in keys]

    param_grid = []
    for combo in itertools.product(*value_lists):
        params = base_params.copy()
        for key, value in zip(keys, combo):
            params[key] = value
        param_grid.append(params)

    return param_grid


def parameter_sweep(param_grid, n, R, base_seed, progress_updates=100):
    """ Runs R replicates for each parameter setting in param_grid and
        collects all results into a flat table suitable for plots.
        Saves raw results to CSV and returns (result_table, run_label). """

    run_label = "sweep_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    result_table = []
    total = len(param_grid)
    stride = max(1, total // progress_updates)

    for idx, params in enumerate(param_grid, start=1):
        rows = run_replicates(params, n, R, base_seed)
        result_table.extend(rows)

        if idx % stride == 0 or idx == total:
            combos_done = idx
            sims_done = idx * R
            print(f"[sweep {run_label}] {combos_done}/{total} combos finished ({sims_done} sims)")

    results_io.save_raw_results(result_table, run_label)

    return result_table, run_label


### Output and plotting
def summarise_over_replicates(result_table, run_label=None):
    """ Aggregates replicate rows into means and standard deviations,
        grouped by parameter setting (beta, gamma, dim).
        Returns a summary table with one row per parameter combination. """

    param_cols = {'beta', 'gamma', 'dim'}

    # Group rows by their parameter combination
    groups = {}
    for row in result_table:
        key = (row['beta'], row['gamma'], row['dim'])
        if key not in groups:
            groups[key] = []
        groups[key].append(row)

    # Identify metric columns (everything except param and seed columns)
    all_cols = [c for c in result_table[0].keys() if c not in param_cols and c != 'seed']

    summary_table = []
    for (beta, gamma, dim), rows in groups.items():
        summary_row = {'beta': beta, 'gamma': gamma, 'dim': dim}
        for col in all_cols:
            values = [r[col] for r in rows]
            stats = statistics.compute_stats(values)
            summary_row[col + '_mean']    = stats['mean']
            summary_row[col + '_median']  = stats['median']
            summary_row[col + '_std']     = stats['std']
            summary_row[col + '_cv']      = stats['cv']
            summary_row[col + '_ci_low']  = stats['ci_low']
            summary_row[col + '_ci_high'] = stats['ci_high']
        summary_table.append(summary_row)

    if run_label is not None:
        results_io.save_summary(summary_table, run_label)

    return summary_table

def plot_metric_vs_param(summary_table, metric_key, x_param="beta", y_param="gamma", ax=None, cmap="viridis", title=None, show_colorbar=False):
    """ Produces your key empirical plots (metric vs parameter) """
    x_vals, y_vals, grid = _extract_surface(summary_table, x_param, y_param, metric_key)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
        created_fig = True
    else:
        fig = ax.figure

    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]],
        cmap=cmap,
    )
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_title(title or _pretty_metric(metric_key))
    ax.set_xticks(_select_ticks(list(x_vals)))
    ax.set_yticks(_select_ticks(list(y_vals)))

    if show_colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return fig if created_fig else ax

def plot_metric_panels(summary_table, metric_keys, x_param="beta", y_param="gamma", run_label=None, output_dir="results/plots"):
    if not metric_keys:
        return None
    
    ncols = 2
    nrows = (len(metric_keys) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    axes = axes.ravel()

    for idx, metric_key in enumerate(metric_keys):
        plot_metric_vs_param(summary_table, metric_key, x_param=x_param, y_param=y_param, ax=axes[idx], show_colorbar=True)

    for idx in range(len(metric_keys), len(axes)):
        fig.delaxes(axes[idx])

    fig.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = (run_label or "plots") + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"{filename}.png"
    fig.savefig(filepath, dpi=200)
    plt.close(fig)
    return str(filepath)

### --- HELPER METHODS --- ###
def _extract_surface(summary_table, x_param, y_param, metric_key):
    """prepares the data grid for plotting one metric versus beta/gamma"""

    if not summary_table:
        raise ValueError("summary_table is empty")

    def _clean(val):
        return round(float(val), 6)

    x_vals = sorted(set(_clean(row[x_param]) for row in summary_table))
    y_vals = sorted(set(_clean(row[y_param]) for row in summary_table))

    grid = np.full((len(y_vals), len(x_vals)), np.nan, dtype=float)

    # Lookup dictionaries: maps each beta/gamma value to its column/row index
    x_idx = {v: i for i, v in enumerate(x_vals)}
    y_idx = {v: i for i, v in enumerate(y_vals)}

    for row in summary_table:
        if metric_key not in row:
            raise KeyError(f"Metric '{metric_key}' not found in summary rows")
        grid[y_idx[_clean(row[y_param])], x_idx[_clean(row[x_param])]] = float(row[metric_key])

    return np.asarray(x_vals, dtype=float), np.asarray(y_vals, dtype=float), grid


def _select_ticks(values, max_ticks=6):
    if len(values) <= max_ticks:
        return values
    idx = np.linspace(0, len(values) - 1, num=max_ticks, dtype=int)
    return [values[i] for i in idx]


def _pretty_metric(metric_key):
    label = metric_key[:-5] + " (mean)" if metric_key.endswith("_mean") else metric_key
    return label.replace("_", " ").title()
