import numpy as np


### Descriptive Statistics ###

def compute_stats(values):
    """ Computes all summary statistics for a list of replicate values.
        Returns a dict with mean, median, std, cv, ci_low, ci_high. """
    
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            'mean': float('nan'),
            'median': float('nan'),
            'std': float('nan'),
            'cv': float('nan'),
            'ci_low': float('nan'),
            'ci_high': float('nan'),
        }

    mean   = float(np.mean(values))
    median = float(np.median(values))
    std    = float(np.std(values))
    cv     = float(std / mean) if mean != 0 else float('nan')
    ci_low, ci_high = confidence_interval(values)

    return {
        'mean':    mean,
        'median':  median,
        'std':     std,
        'cv':      cv,
        'ci_low':  ci_low,
        'ci_high': ci_high,
    }


### Uncertainty ###

def confidence_interval(values, level=0.95):
    """ Computes a confidence interval for a metric across replicates.
        Returns (ci_low, ci_high). """

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    n      = len(values)
    if n == 0:
        return float('nan'), float('nan')
    if n == 1:
        x = float(values[0])
        return x, x
    
    mean   = np.mean(values)
    std    = np.std(values)

    margin = 1.96 * (std / np.sqrt(n))

    return float(mean - margin), float(mean + margin)

def aggregate_powerlaw_metrics(replicate_data_list):
    """
    Aggregates statistical power-law metrics across R simulated graph replicates.
    """
    
    def finite_values(key):
        values = []
        for row in replicate_data_list:
            try:
                value = float(row[key])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
        return values

    alphas = finite_values("indegree_alpha")
    xmins = finite_values("indegree_xmin")
    ks_dists = finite_values("indegree_KS")

    p_log = finite_values("indegree_LRT_log_p")
    r_log = finite_values("indegree_LRT_log_R")

    paired_log_results = list(zip(r_log, p_log))

    pl_favored_count = sum(
        1 for r, p in paired_log_results
        if r > 0 and p < 0.05
    )

    log_favored_count = sum(
        1 for r, p in paired_log_results
        if r < 0 and p < 0.05
    )

    inconclusive_count = sum(
        1 for r, p in paired_log_results
        if p >= 0.05
    )

    total_valid_runs = len(paired_log_results)

    return {
        "alpha_mean": float(np.mean(alphas)) if alphas else float("nan"),
        "alpha_std": float(np.std(alphas)) if alphas else float("nan"),
        "xmin_median": float(np.median(xmins)) if xmins else float("nan"),
        "KS_mean": float(np.mean(ks_dists)) if ks_dists else float("nan"),

        "fraction_PL_superior_to_lognormal": (
            pl_favored_count / total_valid_runs
            if total_valid_runs > 0 else float("nan")
        ),

        "fraction_lognormal_superior": (
            log_favored_count / total_valid_runs
            if total_valid_runs > 0 else float("nan")
        ),

        "fraction_inconclusive": (
            inconclusive_count / total_valid_runs
            if total_valid_runs > 0 else float("nan")
        ),
    }

