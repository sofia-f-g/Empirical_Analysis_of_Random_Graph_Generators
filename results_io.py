import csv
import os


### Saving Results ###

def save_raw_results(result_table, run_label, results_dir="results/raw"):
    """Saves the per-replicate result table to CSV.

    Parameters
    ----------
    result_table : list[dict]
        One dict per replicate, as returned by parameter_sweep.
    run_label : str
        Identifier for this run (e.g. 'sweep_20260303_143022').
        Used as the filename stem.
    results_dir : str
        Directory in which to write the file. Created if absent.

    Returns
    -------
    str
        Path to the written CSV file.
    """
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, f"{run_label}.csv")

    if not result_table:
        return filepath

    fieldnames = list(result_table[0].keys())
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        writer.writerows(result_table)

    return filepath


def save_summary(summary_table, run_label, results_dir="results/summary"):
    """Saves the aggregated summary table to CSV.

    Parameters
    ----------
    summary_table : list[dict]
        One dict per parameter setting, as returned by summarise_over_replicates.
    run_label : str
        Identifier for this run — should match the one used for save_raw_results
        so that raw and summary files from the same run share the same label.
    results_dir : str
        Directory in which to write the file. Created if absent.

    Returns
    -------
    str
        Path to the written CSV file.
    """
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, f"{run_label}.csv")

    if not summary_table:
        return filepath

    fieldnames = list(summary_table[0].keys())
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        writer.writerows(summary_table)

    return filepath


### Loading Results ###

def load_raw_results(filepath):
    """Reads a raw results CSV back into a list of dicts.

    Parameters
    ----------
    filepath : str
        Path to a CSV file previously written by save_raw_results.

    Returns
    -------
    list[dict]
        Same structure as the result_table produced by parameter_sweep.
        All values are strings; cast as needed after loading.
    """
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_summary(filepath, delimiter=';'):
    """Reads a summary CSV back into a list of dicts.

    Parameters
    ----------
    filepath : str
        Path to a CSV file previously written by save_summary.

    Returns
    -------
    list[dict]
        Same structure as the summary_table produced by summarise_over_replicates.
        Can be passed directly to plot_metric_vs_param after loading.
        All values are strings; cast as needed after loading.
    """
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        return list(reader)
