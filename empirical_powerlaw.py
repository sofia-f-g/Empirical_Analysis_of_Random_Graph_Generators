"""
This module interfaces with the 'powerlaw' package to implement rigorous 
statistical methodologies. It performs Maximum Likelihood Estimation, 
KS goodness-of-fit thresholding, and Likelihood Ratio Tests against 
competing heavy-tailed distributions for given network degree sequences.
"""

import numpy as np
import warnings
import powerlaw

def compute_degree_statistics(degree_sequence):
    """
    Computes power-law statistical metrics for a degree sequence.
    
    Parameters:
    degree_sequence (list or np.ndarray): The raw array of vertex degrees.
    
    Returns:
    dict: A dictionary containing the fitted alpha, xmin, KS distance, 
          and Likelihood Ratio Test results against alternative models.
    """
    
    # Power-law probability distributions are strictly undefined for zero.
    data = np.array(degree_sequence)
    data = data[data > 0]  # Filter out degree = 0
    
    # If the network is exceedingly sparse or the tail sample size is insufficient fitting cannot proceed 
    if len(data) < 10:
        return _empty_statistics_dictionary()
    
    # The package utilizes numerical approximations that occasionally throw safe true_divide warnings. 
    # These are suppressed to maintain clean stdout during grid searches involving thousands of replicates.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Instantiate the Fit object.
        # discrete=True specifies that network degrees are integers.
        # estimate_discrete=True uses the high-speed analytical approximation for discrete MLE
        fit = powerlaw.Fit(data, discrete=True, estimate_discrete=True)
        
        # Extract foundational parameters computed via KS distance minimization
        alpha = fit.power_law.alpha
        xmin = fit.power_law.xmin
        ks_distance = fit.power_law.D
        
        # Execute Likelihood Ratio Tests (LRT) against competing theoretical models.
        # R is the normalized log-likelihood ratio; p is the statistical significance.
        # A positive R indicates the empirical data favors the power law.
        
        # 1. Exponential (The absolute minimum baseline for heavy tails)
        R_exp, p_exp = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
        
        # 2. Lognormal (A highly prevalent alternative for heavy-tailed phenomena)
        R_log, p_log = fit.distribution_compare('power_law', 'lognormal', normalized_ratio=True)
        
        # 3. Truncated Power Law (A nested model identifying finite-size cut-offs)
        R_trunc, p_trunc = fit.distribution_compare('power_law', 'truncated_power_law', normalized_ratio=True)
        
        return {
            'alpha_empirical': alpha,
            'xmin_optimal': xmin,
            'KS_distance': ks_distance,
            'LRT_exp_R': R_exp,
            'LRT_exp_p': p_exp,
            'LRT_log_R': R_log,
            'LRT_log_p': p_log,
            'LRT_trunc_R': R_trunc,
            'LRT_trunc_p': p_trunc
        }

def _empty_statistics_dictionary():
    return {
        'alpha_empirical': np.nan, 'xmin_optimal': np.nan, 'KS_distance': np.nan,
        'LRT_exp_R': np.nan, 'LRT_exp_p': np.nan, 'LRT_log_R': np.nan,
        'LRT_log_p': np.nan, 'LRT_trunc_R': np.nan, 'LRT_trunc_p': np.nan
    }


