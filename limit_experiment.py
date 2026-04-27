from pathlib import Path
from datetime import datetime
import csv

import numpy as np
import matplotlib.pyplot as plt

import generator
import metrics


def make_start_label(start_time):
    month_names = {
        1: "jan", 2: "feb", 3: "mar", 4: "apr",
        5: "may", 6: "jun", 7: "jul", 8: "aug",
        9: "sep", 10: "oct", 11: "nov", 12: "dec",
    }

    day = start_time.day
    month = month_names[start_time.month]
    start_clock = start_time.strftime("%H%M")

    return f"{day}{month}_{start_clock}"


def make_run_label(start_time, end_time):
    start_label = make_start_label(start_time)
    end_clock = end_time.strftime("%H%M")
    return f"{start_label}_{end_clock}"


def build_params(n, beta, gamma):
    return {
        "beta": beta,
        "gamma": gamma,
        "dim": 2,
        "space_cfg": {"bounds": [[-0.5, 0.5], [-0.5, 0.5]]},
        "age_cfg": {"min": 0.0, "max": float(n)},
        "profile_cfg": {"type": "normalized_cutoff"},
    }


def run_one(n, gamma, c_ed, seed):
    """
    Run one simulation for the global clustering limit experiment.

    We use beta(gamma) = c_ed * (1 - gamma), so that

        beta / (1 - gamma) = c_ed.

    This keeps the theoretical mean outdegree fixed across gamma values.
    """
    beta = c_ed * (1.0 - gamma)
    params = build_params(n, beta, gamma)

    V, E = generator.generate_graph(params, n, seed)
    cc = metrics.clustering_coefficient(V, E)

    return {
        "n": int(n),
        "gamma": float(gamma),
        "beta": float(beta),
        "c_ed": float(c_ed),
        "seed": int(seed),
        "global_clustering": float(cc["global"]),
        "closed_wedges": float(cc["closed_wedges"]),
        "wedges": float(cc["wedges"]),
    }


def append_row_csv(row, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    file_exists = filepath.exists()

    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), delimiter=";")

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)
        f.flush()

def load_csv(filepath):
    filepath = Path(filepath)

    if not filepath.exists():
        return []

    with open(filepath, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        return list(reader)
    

def save_csv(rows, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return str(filepath)

    fieldnames = list(rows[0].keys())

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    return str(filepath)


def summarize(rows):
    """
    Summarize replicates by (c_ed, n, gamma, beta).

    We only summarize the quantities needed for the limit experiment.
    """
    groups = {}

    for row in rows:
        key = (
            float(row["c_ed"]),
            int(row["n"]),
            float(row["gamma"]),
            float(row["beta"]),
        )
        groups.setdefault(key, []).append(row)

    summary_rows = []

    for (c_ed, n, gamma, beta), group in sorted(groups.items()):
        global_values = np.asarray(
            [float(row["global_clustering"]) for row in group],
            dtype=float,
        )
        closed_wedges = np.asarray(
            [float(row["closed_wedges"]) for row in group],
            dtype=float,
        )
        wedges = np.asarray(
            [float(row["wedges"]) for row in group],
            dtype=float,
        )

        R = len(group)

        if R > 1:
            global_std = float(np.std(global_values, ddof=1))
            global_se = global_std / np.sqrt(R)
        else:
            global_std = 0.0
            global_se = 0.0

        summary_rows.append({
            "c_ed": c_ed,
            "n": n,
            "gamma": gamma,
            "beta": beta,
            "R": R,

            "global_clustering_mean": float(np.mean(global_values)),
            "global_clustering_std": global_std,
            "global_clustering_ci_low": float(np.mean(global_values) - 1.96 * global_se),
            "global_clustering_ci_high": float(np.mean(global_values) + 1.96 * global_se),

            "closed_wedges_mean": float(np.mean(closed_wedges)),
            "wedges_mean": float(np.mean(wedges)),
        })

    return summary_rows

def filter_complete_n_runs(rows, gamma_values, R):
    """
    Keep only rows belonging to n-values where every gamma value
    has completed at least R replicates.

    This is useful when the experiment is interrupted. It prevents
    partially completed n-values from appearing in the summary plot.
    """
    expected_gammas = {float(gamma) for gamma in gamma_values}

    counts = {}

    for row in rows:
        n = int(row["n"])
        gamma = float(row["gamma"])

        key = (n, gamma)
        counts[key] = counts.get(key, 0) + 1

    all_n_values = sorted({int(row["n"]) for row in rows})

    complete_n_values = []

    for n in all_n_values:
        n_is_complete = True

        for gamma in expected_gammas:
            if counts.get((n, gamma), 0) < R:
                n_is_complete = False
                break

        if n_is_complete:
            complete_n_values.append(n)

    complete_n_values = set(complete_n_values)

    return [
        row for row in rows
        if int(row["n"]) in complete_n_values
    ]


def plot_global_clustering(summary_rows, run_label, output_dir):
    """
    Plot mean global clustering against n, with one line per gamma.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gammas = sorted(set(float(row["gamma"]) for row in summary_rows))

    fig, ax = plt.subplots(figsize=(7, 5))

    for gamma in gammas:
        rows_g = [
            row for row in summary_rows
            if float(row["gamma"]) == gamma
        ]
        rows_g = sorted(rows_g, key=lambda row: int(row["n"]))

        n_values = [int(row["n"]) for row in rows_g]
        means = [float(row["global_clustering_mean"]) for row in rows_g]
        lows = [float(row["global_clustering_ci_low"]) for row in rows_g]
        highs = [float(row["global_clustering_ci_high"]) for row in rows_g]

        ax.plot(n_values, means, marker="o", label=f"$\\gamma={gamma}$")
        ax.fill_between(n_values, lows, highs, alpha=0.15)

    ax.set_xlabel("$n$")
    ax.set_ylabel("Global clustering coefficient")
    ax.set_title("Global clustering size-scaling")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    filepath = output_dir / f"{run_label}_global_clustering.png"
    fig.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return str(filepath)


def run_experiment(
    c_ed,
    gamma_values,
    n_values,
    R,
    base_seed=0,
    output_dir="limit_test/clustering_limit",
    make_plot=True,
):
    """
    Run the global clustering limit experiment.

    Results are saved continuously after each replicate. If the run is interrupted
    with Ctrl+C, the completed rows are still summarized and plotted.
    """
    start_time = datetime.now()
    start_label = make_start_label(start_time)

    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    summary_dir = output_dir / "summary"
    plot_dir = output_dir / "plots"

    temp_raw_path = raw_dir / f"{start_label}_RUNNING.csv"

    total = len(gamma_values) * len(n_values) * R
    done = 0

    print("Starting global clustering limit experiment")
    print(f"c_ed = {c_ed}")
    print(f"gamma_values = {gamma_values}")
    print(f"n_values = {n_values}")
    print(f"R = {R}")
    print(f"Total simulations = {total}")
    print(f"Saving raw results continuously to: {temp_raw_path}")

    interrupted = False

    try:
        for n in n_values:
            for gamma in gamma_values:
                beta = c_ed * (1.0 - gamma)
                print(f"\nRunning n={n}, gamma={gamma}, beta={beta:.6g}")

                for r in range(R):
                    seed = base_seed + done

                    row = run_one(
                        n=n,
                        gamma=gamma,
                        c_ed=c_ed,
                        seed=seed,
                    )

                    append_row_csv(row, temp_raw_path)

                    done += 1
                    print(f"  replicate {r + 1}/{R} done ({done}/{total})")

    except KeyboardInterrupt:
        interrupted = True
        print("\nExperiment interrupted by user.")
        print("Keeping completed raw results and creating partial summary...")

    end_time = datetime.now()
    run_label = make_run_label(start_time, end_time)

    final_raw_path = raw_dir / f"{run_label}.csv"
    final_summary_path = summary_dir / f"{run_label}.csv"

    # Rename the running raw file to the final run label.
    if temp_raw_path.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        temp_raw_path.rename(final_raw_path)

    rows = load_csv(final_raw_path)


    complete_rows = filter_complete_n_runs(
        rows=rows,
        gamma_values=gamma_values,
        R=R,
    )


    summary_rows = summarize(complete_rows)

    summary_path = save_csv(
        summary_rows,
        final_summary_path,
    )

    print(f"\nSaved raw results to: {final_raw_path}")
    print(f"Saved summary results to: {summary_path}")

    plot_path = None
    if make_plot:
        plot_path = plot_global_clustering(
            summary_rows=summary_rows,
            run_label=run_label,
            output_dir=plot_dir,
        )
        print(f"Saved plot to: {plot_path}")

    if interrupted:
        print("\nPartial results saved successfully.")
    else:
        print("\nExperiment completed successfully.")

    return {
        "run_label": run_label,
        "raw_path": str(final_raw_path),
        "summary_path": str(summary_path),
        "plot_path": plot_path,
        "interrupted": interrupted,
    }


if __name__ == "__main__":
    # Tiny test first.
    run_experiment(
        c_ed=5.0,
        gamma_values=[0.45, 0.55],
        n_values=[100, 200],
        R=2,
        base_seed=0,
        make_plot=True,
    )