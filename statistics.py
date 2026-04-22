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