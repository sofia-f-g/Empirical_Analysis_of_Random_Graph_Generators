import numpy as np
from collections import deque
from scipy.optimize import minimize_scalar, minimize
from scipy.special import zeta, logsumexp
from scipy.stats import norm


### Basic Graph Helpers ###

def build_undirected_adjacency(V, E):
    """ Converts edge list into adjacency representaion to make
        metric computation efficient """
    
    vertex_ids = V.get('id')
    if vertex_ids is None:
        raise ValueError("V must contain an id entry with vertex identifiers")

    id_array = np.asarray(vertex_ids)
    n_vertices = len(id_array)

    # Map each original vertex id to a 0..n-1 slot in the adjacency structure
    # adjacency[idx] corresponds to vertex id V["id"][idx]
    index_of = {int(v_id): idx for idx, v_id in enumerate(id_array)}

    adjacency = [[] for _ in range(n_vertices)]

    if E is None:
        return adjacency

    edges = np.asarray(E, dtype=int)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("E must be of shape (m, 2) containing (src, dst) pairs")

    for v_i, v_j in edges:
        if v_i not in index_of or v_j not in index_of:
            raise ValueError("Edge references unknown vertex id")

        # Double edges represent undirected 
        adjacency[index_of[v_i]].append(int(v_j))
        adjacency[index_of[v_j]].append(int(v_i))

    return adjacency


def compute_degree_sequences(V, E):
    """
    Computes indegree, outdegree, and total degree arrays for all vertices
    from the directed edge list.

    Edges are assumed to be stored as (src, dst), where src is the younger
    vertex and dst is the older vertex.
    """
    
    vertex_ids = np.asarray(V['id'])
    n = len(vertex_ids)

    indegree = np.zeros(n, dtype=int)
    outdegree = np.zeros(n, dtype=int)

    if E is not None and len(E) > 0:
        edges = np.asarray(E, dtype=int)
        index_of = {int(v_id): idx for idx, v_id in enumerate(vertex_ids)}

        for src_id, dst_id in edges:

            src = index_of[src_id]
            dst = index_of[dst_id]

            outdegree[src] += 1
            indegree[dst] += 1
    
    total_degree = indegree + outdegree

    return {
        'id': np.asarray(V['id']),
        'indegree': indegree,
        'outdegree': outdegree,
        'total_degree': total_degree,
    }
    


### Degree Distribution ###

def degree_histogram(deg):
    """ Produces degree frequency counts used for plots
        and comparisions across parameter settings.
        Returns (k_values, counts) arrays. """

    degrees = np.asarray(deg, dtype=int)
    if len(degrees) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    k_min = degrees.min()
    k_max = degrees.max()
    k_values = np.arange(k_min, k_max + 1)
    counts = np.bincount(degrees - k_min)

    return k_values, counts

def estimate_powerlaw_exponent(deg, k_min=None, min_tail=20, min_k=2, max_k_percentile=90):
    """ Estimates the power-law tail exponent alpha using Maximum Likelihood
        Estimation (Hill estimator): alpha = 1 + n * [sum(ln(k_i/k_min))]^-1

        k_min is the lower cutoff for the power-law tail. If None, it is chosen
        automatically by minimizing the Kolmogorov-Smirnov distance between the
        empirical tail CDF and the fitted power-law CDF.

    Parameters
    ----------
    deg : array-like
        Degree sequence.
    k_min : int or None
        Fixed lower cutoff for the tail. If None, choose automatically by
        minimizing KS distance over candidate k_min values.
    min_tail : int
        Minimum number of tail observations required for a valid fit.
    min_k_min : int
        Smallest k_min to consider when searching automatically.
    max_k_min_percentile : float
        Only consider candidate k_min values up to this percentile of the
        observed degree distribution, to avoid extremely short tails.

    """

    degrees = np.asarray(deg, dtype=float)
    degrees = degrees[np.isfinite(degrees)]
    degrees = degrees[degrees >= 1]  # exclude isolates (degree 0)

    def invalid_result(reason, k_min_value=None, n_tail=0):
        return {
            "alpha": float("nan"),
            "k_min": None if k_min_value is None else int(k_min_value),
            "n_tail": int(n_tail),
            "ks": float("nan"),
            "valid_fit": False,
            "reason": reason,
        }

    if len(degrees) < min_tail:
        return invalid_result("insufficient_degree_data")
    
    def fit_for_k(k):
        """
        Fit alpha for a fixed k_min = k.
        Return a result dict if valid, else None.
        """

        tail = degrees[degrees >= k]
        n = len(tail)

        if n < min_tail:
            return None

        log_sum = np.sum(np.log(tail / k))
        if not np.isfinite(log_sum) or log_sum <= 0:
            return None

        alpha = 1.0 + n / log_sum
        if not np.isfinite(alpha):
            return None

        sorted_tail = np.sort(tail)
        empirical_cdf = np.arange(1, n + 1, dtype=float) / n

        # Continuous Pareto-style fitted CDF on the tail
        theoretical_cdf = 1.0 - (k / sorted_tail) ** (alpha - 1.0)
        ks = np.max(np.abs(empirical_cdf - theoretical_cdf))

        return {
            "alpha": float(alpha),
            "k_min": int(k),
            "n_tail": int(n),
            "ks": float(ks),
            "valid_fit": True,
            "reason": "ok",
        }
    
    # Case 1: fixed user-supplied k_min
    if k_min is not None:
        fit = fit_for_k(k_min)
        if fit is None:
            tail = degrees[degrees >= k_min]
            return invalid_result(
                "tail_too_small_or_degenerate",
                k_min_value=k_min,
                n_tail=len(tail),
            )
        return fit
    
    # Case 2: automatic search over candidate k_min values

    # Search over candidate k_min values (up to the 95th percentile to keep
    # enough tail points for a reliable fit)
    candidates = np.unique(degrees.astype(int))
    candidates = candidates[candidates >= min_k]

    upper = np.percentile(degrees, max_k_percentile)
    candidates = candidates[candidates <= upper]

    if len(candidates) == 0:
        return invalid_result("no_kmin_candidates_after_percentile_cap")

    best_fit = None

    for k in candidates:
        fit = fit_for_k(k)
        if fit is None:
            continue

        if best_fit is None or fit["ks"] < best_fit["ks"]:
            best_fit = fit

    if best_fit is None:
        return invalid_result("no_valid_tail_candidate")

    return best_fit

### Clustering ###

def clustering_coefficient(V, E):

    """
    Compute average local clustering and global clustering
    on the undirected projection of the graph.

    Average local clustering is computed only over vertices
    with undirected degree at least 2, matching the definition
    used in the reference paper.
    """

    adj = build_undirected_adjacency(V, E)
    adj_sets = [set(neighbors) for neighbors in adj]
    index_of = {int(v_id): idx for idx, v_id in enumerate(np.asarray(V['id']))}

    n = len(adj)
    local_coeffs = []
    triangle_sum = 0.0
    triplet_sum = 0.0
    n_degree_ge_2 = 0

    for v in range(n):
        deg = len(adj[v])
        if deg < 2:
            continue
        
        n_degree_ge_2 += 1
        possible = deg * (deg - 1) / 2
        edges_between = 0
        neighbors = adj[v]
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if neighbors[j] in adj_sets[index_of[neighbors[i]]]:
                    edges_between += 1

        local_coeffs.append(edges_between / possible)
        triangle_sum += edges_between
        triplet_sum += possible

    avg_local = float(np.mean(local_coeffs)) if n_degree_ge_2 > 0 else 0.0  # # We must have at least one node with degree at least 2
    global_cc = float(triangle_sum / triplet_sum) if triplet_sum > 0 else 0.0

    return {'avg_local': avg_local, 'global': global_cc, 'n_deg_ge_2': int(n_degree_ge_2),'closed_wedges': float(triangle_sum), 'wedges': float(triplet_sum),}


### Paths / Distances ###

def average_shortest_path_length(V, E):
    """ Average shortest-path length over all connected 
        vertex pairs in an undirected graph """
    
    adj = build_undirected_adjacency(V, E)
    id_array = np.asarray(V['id'])
    index_of = {int(v_id): idx for idx, v_id in enumerate(id_array)}
    n = len(adj)
    total, count = 0, 0

    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        queue = [src]
        head = 0
        while head < len(queue):
            u = queue[head]; head += 1
            for w_id in adj[u]:
                w = index_of[w_id]
                if dist[w] == -1:
                    dist[w] = dist[u] + 1
                    queue.append(w)
        for d in dist:
            if d > 0:
                total += d; count += 1
    return total / count if count > 0 else 0.0


### Wrapper ###

def compute_metrics(V, E):

    """ Return a dictionary of all required metrics,
        degrees, clustering, path length, edges etc. """

    n_vertices = len(V['id'])
    n_edges = 0 if E is None or len(E) == 0 else len(E)

    deg_seq = compute_degree_sequences(V, E)
    indegree = deg_seq['indegree']
    outdegree = deg_seq['outdegree']
    total_degree = deg_seq['total_degree']

    cc = clustering_coefficient(V, E)
    aspl = average_shortest_path_length(V, E)
    #pl = estimate_powerlaw_exponent(degrees)

    return {
        'n_vertices': n_vertices,
        'n_edges': n_edges,

        'indegree_mean': float(np.mean(indegree)),
        'indegree_max': int(np.max(indegree)),
        'outdegree_mean': float(np.mean(outdegree)),
        'outdegree_max': int(np.max(outdegree)),
        'total_degree_mean': float(np.mean(total_degree)),
        'total_degree_max': int(np.max(total_degree)),

        'avg_local_clustering': cc['avg_local'],
        'vertices_degree_ge_2': cc['n_deg_ge_2'],
        'global_clustering': cc['global'],
        'avg_shortest_path_length': aspl,

        'closed_wedges': cc['closed_wedges'],
        'wedges': cc['wedges'],

        #'powerlaw_alpha': pl['alpha'],
        #'powerlaw_k_min': pl['k_min'],
        #'powerlaw_n_tail': pl['n_tail'],
        #'powerlaw_ks': pl['ks'],
        #'powerlaw_valid': pl['valid_fit'],
        #'powerlaw_reason': pl['reason'],
    }

# Discard this!
def _clean_integer_degree_data(values):
    """
    Clean degree data for tail fitting.

    Degree data are nonnegative integers. Zero values are kept in the data
    because they form part of the empirical body below x_min, but they are
    never used in the power-law tail itself because x_min >= 1.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    return x.astype(int)


def _invalid_powerlaw_result(reason, n=0, xmin=None, n_tail=0):
    return {
        "method": "discrete_mle_ks_semiparametric_bootstrap",
        "fit_success": False,
        "reportable": False,
        "reason": reason,

        "alpha": float("nan"),
        "xmin": None if xmin is None else int(xmin),
        "n_tail": int(n_tail),
        "tail_fraction": float(n_tail / n) if n > 0 else float("nan"),
        "ks": float("nan"),
        "loglik_powerlaw": float("nan"),

        "gof_p": float("nan"),
        "gof_pass": False,
        "n_bootstrap": 0,
        "n_bootstrap_success": 0,

        "lognormal_loglik": float("nan"),
        "lognormal_mu": float("nan"),
        "lognormal_sigma": float("nan"),
        "llr_powerlaw_vs_lognormal": float("nan"),
        "aic_delta_lognormal_minus_powerlaw": float("nan"),

        "truncated_powerlaw_loglik": float("nan"),
        "truncated_powerlaw_alpha": float("nan"),
        "truncated_powerlaw_lambda": float("nan"),
        "llr_powerlaw_vs_truncated_powerlaw": float("nan"),
        "aic_delta_truncated_powerlaw_minus_powerlaw": float("nan"),

        "best_aic_model": None,
    }


def _fit_discrete_powerlaw_given_xmin(
    x,
    xmin,
    min_tail=50,
    min_unique_tail=2,
    alpha_bounds=(1.000001, 100.0),
):
    """
    Fit a discrete power law for a fixed xmin.

    Model:
        P(K = k | K >= xmin) = k^(-alpha) / zeta(alpha, xmin)

    Returns None if the tail is too small or degenerate.
    """
    xmin = int(xmin)
    tail = x[x >= xmin]
    n_tail = len(tail)

    if n_tail < min_tail:
        return None

    if len(np.unique(tail)) < min_unique_tail:
        return None

    sum_log = float(np.sum(np.log(tail)))

    def neg_log_likelihood(alpha):
        normalizer = zeta(alpha, xmin)
        if not np.isfinite(normalizer) or normalizer <= 0:
            return np.inf

        # Negative log-likelihood:
        # -ell(alpha) = alpha * sum(log k_i) + n_tail * log(zeta(alpha, xmin))
        return alpha * sum_log + n_tail * np.log(normalizer)

    result = minimize_scalar(
        neg_log_likelihood,
        bounds=alpha_bounds,
        method="bounded",
        options={"xatol": 1e-7},
    )

    if not result.success or not np.isfinite(result.fun):
        return None

    alpha = float(result.x)
    loglik = -float(result.fun)

    unique_k, counts = np.unique(tail, return_counts=True)
    empirical_cdf = np.cumsum(counts) / n_tail

    zeta_xmin = zeta(alpha, xmin)
    model_cdf = 1.0 - zeta(alpha, unique_k + 1) / zeta_xmin

    ks = float(np.max(np.abs(empirical_cdf - model_cdf)))

    near_upper_bound = alpha > alpha_bounds[1] - 1e-4

    return {
        "method": "discrete_mle_ks_semiparametric_bootstrap",
        "fit_success": True,
        "reportable": False,
        "reason": "alpha_at_upper_bound" if near_upper_bound else "ok",

        "alpha": alpha,
        "xmin": xmin,
        "n_tail": int(n_tail),
        "tail_fraction": float(n_tail / len(x)),
        "ks": ks,
        "loglik_powerlaw": loglik,
    }


def _fit_discrete_powerlaw_by_ks(
    x,
    xmin=None,
    min_tail=50,
    min_unique_tail=2,
    min_xmin=2,
    max_xmin_percentile=90,
    alpha_bounds=(1.000001, 100.0),
):
    """
    Fit a discrete power law and choose xmin by minimizing the KS distance.

    If xmin is supplied, only that cutoff is fitted.
    If xmin is None, all admissible observed integer cutoffs are tested.
    """
    n = len(x)

    if n < min_tail:
        return _invalid_powerlaw_result("insufficient_data", n=n)

    if xmin is not None:
        fit = _fit_discrete_powerlaw_given_xmin(
            x,
            xmin=xmin,
            min_tail=min_tail,
            min_unique_tail=min_unique_tail,
            alpha_bounds=alpha_bounds,
        )
        if fit is None:
            n_tail = int(np.sum(x >= int(xmin)))
            return _invalid_powerlaw_result(
                "tail_too_small_or_degenerate",
                n=n,
                xmin=xmin,
                n_tail=n_tail,
            )

        return fit

    positive_candidates = x[x >= min_xmin]

    if len(positive_candidates) == 0:
        return _invalid_powerlaw_result("no_positive_candidates", n=n)

    candidates = np.unique(positive_candidates)

    if max_xmin_percentile is not None:
        upper = np.percentile(positive_candidates, max_xmin_percentile)
        candidates = candidates[candidates <= upper]

    candidates = [
        int(k)
        for k in candidates
        if np.sum(x >= k) >= min_tail
    ]

    if len(candidates) == 0:
        return _invalid_powerlaw_result("no_valid_xmin_candidates", n=n)

    fits = []
    for candidate in candidates:
        fit = _fit_discrete_powerlaw_given_xmin(
            x,
            xmin=candidate,
            min_tail=min_tail,
            min_unique_tail=min_unique_tail,
            alpha_bounds=alpha_bounds,
        )
        if fit is not None:
            fits.append(fit)

    if len(fits) == 0:
        return _invalid_powerlaw_result("no_successful_fit", n=n)

    return min(fits, key=lambda row: row["ks"])


def _sample_discrete_powerlaw(alpha, xmin, size, rng, tail_tol=1e-12, max_support_size=200000):
    """
    Sample from the discrete power law

        P(K = k | K >= xmin) = k^(-alpha) / zeta(alpha, xmin).

    For moderate acceptance probability, exact rejection sampling from
    NumPy's Zipf distribution is used. Otherwise, inverse transform sampling
    on a truncated support is used; this is accurate when the omitted tail
    mass is below tail_tol.
    """
    size = int(size)
    xmin = int(xmin)

    if size <= 0:
        return np.empty(0, dtype=int)

    if alpha <= 1:
        raise ValueError("alpha must be > 1 for a normalizable discrete power law.")

    zeta_tail = zeta(alpha, xmin)
    zeta_full = zeta(alpha, 1.0)
    accept_prob = zeta_tail / zeta_full

    # Exact rejection sampling when this is not too inefficient.
    if np.isfinite(accept_prob) and accept_prob > 0.02:
        out = np.empty(size, dtype=int)
        filled = 0

        while filled < size:
            remaining = size - filled
            batch_size = max(1000, int(1.25 * remaining / accept_prob) + 10)
            draws = rng.zipf(alpha, size=batch_size)
            accepted = draws[draws >= xmin]

            take = min(len(accepted), remaining)
            if take > 0:
                out[filled:filled + take] = accepted[:take]
                filled += take

        return out

    # Inverse-transform fallback. This is useful when alpha is large and xmin
    # is not small, because rejection sampling can then be very inefficient.
    upper = max(xmin + 1024, 2 * xmin)
    max_upper = xmin + max_support_size - 1

    while True:
        upper = min(upper, max_upper)
        support = np.arange(xmin, upper + 1, dtype=float)
        weights = support ** (-alpha)
        covered_mass = float(np.sum(weights) / zeta_tail)

        if covered_mass >= 1.0 - tail_tol or upper >= max_upper:
            break

        span = upper - xmin + 1
        upper = xmin + min(2 * span, max_support_size) - 1

    cdf = np.cumsum(weights)
    cdf = cdf / cdf[-1]
    cdf[-1] = 1.0

    u = rng.random(size)
    idx = np.searchsorted(cdf, u, side="left")

    return support[idx].astype(int)


def _semiparametric_powerlaw_bootstrap_pvalue(
    x,
    observed_fit,
    n_bootstrap,
    rng,
    min_tail=50,
    min_unique_tail=2,
    min_xmin=2,
    max_xmin_percentile=90,
    alpha_bounds=(1.000001, 100.0),
):
    """
    Clauset-Shalizi-Newman style semi-parametric bootstrap.

    The empirical body below xmin is resampled from the observed data.
    The tail at or above xmin is generated from the fitted discrete power law.
    Each synthetic data set is refitted from scratch, including xmin selection.
    """
    n = len(x)
    xmin = int(observed_fit["xmin"])
    alpha = float(observed_fit["alpha"])
    observed_ks = float(observed_fit["ks"])
    tail_fraction = float(observed_fit["n_tail"] / n)

    body = x[x < xmin]

    n_success = 0
    n_larger_ks = 0

    for _ in range(int(n_bootstrap)):
        synthetic = np.empty(n, dtype=int)
        is_tail = rng.random(n) < tail_fraction

        n_tail_synth = int(np.sum(is_tail))
        n_body_synth = n - n_tail_synth

        if n_tail_synth > 0:
            synthetic[is_tail] = _sample_discrete_powerlaw(
                alpha=alpha,
                xmin=xmin,
                size=n_tail_synth,
                rng=rng,
            )

        if n_body_synth > 0:
            if len(body) > 0:
                synthetic[~is_tail] = rng.choice(body, size=n_body_synth, replace=True)
            else:
                synthetic[~is_tail] = _sample_discrete_powerlaw(
                    alpha=alpha,
                    xmin=xmin,
                    size=n_body_synth,
                    rng=rng,
                )

        synthetic_fit = _fit_discrete_powerlaw_by_ks(
            synthetic,
            xmin=None,
            min_tail=min_tail,
            min_unique_tail=min_unique_tail,
            min_xmin=min_xmin,
            max_xmin_percentile=max_xmin_percentile,
            alpha_bounds=alpha_bounds,
        )

        if not synthetic_fit["fit_success"]:
            continue

        n_success += 1

        if synthetic_fit["ks"] >= observed_ks:
            n_larger_ks += 1

    if n_success == 0:
        return float("nan"), 0

    # Add-one smoothing avoids reporting exactly zero from a finite bootstrap.
    p_value = (n_larger_ks + 1.0) / (n_success + 1.0)

    return float(p_value), int(n_success)


def _fit_discrete_lognormal_tail(tail, xmin):
    """
    Fit a discretized lognormal tail by maximum likelihood.

    The discrete probability is approximated by binning a continuous lognormal:
        P(K = k) = P(k - 1/2 <= X < k + 1/2 | X >= xmin - 1/2).
    """
    tail = np.asarray(tail, dtype=int)
    xmin = int(xmin)

    if len(tail) < 2:
        return None

    log_tail = np.log(tail)
    mu0 = float(np.mean(log_tail))
    sigma0 = float(np.std(log_tail, ddof=1))
    sigma0 = max(sigma0, 0.25)

    lower_cut = max(xmin - 0.5, np.nextafter(0.0, 1.0))
    lower_bins = np.maximum(tail - 0.5, np.nextafter(0.0, 1.0))
    upper_bins = tail + 0.5

    def neg_log_likelihood(theta):
        mu = float(theta[0])
        sigma = float(np.exp(theta[1]))

        z_cut = (np.log(lower_cut) - mu) / sigma
        tail_norm = 1.0 - norm.cdf(z_cut)

        if not np.isfinite(tail_norm) or tail_norm <= 0:
            return np.inf

        z_upper = (np.log(upper_bins) - mu) / sigma
        z_lower = (np.log(lower_bins) - mu) / sigma

        probs = (norm.cdf(z_upper) - norm.cdf(z_lower)) / tail_norm

        if np.any(~np.isfinite(probs)) or np.any(probs <= 0):
            return np.inf

        return -float(np.sum(np.log(probs)))

    result = minimize(
        neg_log_likelihood,
        x0=np.array([mu0, np.log(sigma0)]),
        method="L-BFGS-B",
        bounds=[(None, None), (np.log(1e-3), np.log(20.0))],
    )

    if not result.success or not np.isfinite(result.fun):
        return None

    mu_hat = float(result.x[0])
    sigma_hat = float(np.exp(result.x[1]))
    loglik = -float(result.fun)

    return {
        "mu": mu_hat,
        "sigma": sigma_hat,
        "loglik": loglik,
        "n_params": 2,
    }


def _truncated_powerlaw_logZ(alpha, lam, xmin, tol=1e-12, max_terms=500000):
    """
    Numerically compute

        Z(alpha, lambda, xmin)
        = sum_{k=xmin}^infinity k^(-alpha) exp(-lambda k)

    on the log scale.
    """
    alpha = float(alpha)
    lam = float(lam)
    xmin = int(xmin)

    if alpha <= 0 or lam < 0:
        return np.inf

    if lam <= 1e-12:
        if alpha <= 1:
            return np.inf
        return float(np.log(zeta(alpha, xmin)))

    log_total = -np.inf
    k_start = xmin
    chunk = 4096
    terms_done = 0
    converged = False

    while terms_done < max_terms:
        m = min(chunk, max_terms - terms_done)
        ks = np.arange(k_start, k_start + m, dtype=float)

        log_terms = -alpha * np.log(ks) - lam * ks
        log_chunk = float(logsumexp(log_terms))

        log_total = float(np.logaddexp(log_total, log_chunk))

        terms_done += m
        k_start += m

        if np.isfinite(log_total) and np.exp(log_chunk - log_total) < tol:
            converged = True
            break

        if not np.isfinite(log_chunk):
            converged = True
            break

    if not converged:
        return np.inf

    return log_total


def _fit_truncated_powerlaw_tail(tail, xmin, alpha_start):
    """
    Fit the exponentially truncated discrete power law

        P(K = k) proportional to k^(-alpha) exp(-lambda k),
        k >= xmin.

    This is used only as an alternative-model diagnostic.
    """
    tail = np.asarray(tail, dtype=int)
    xmin = int(xmin)

    if len(tail) < 2:
        return None

    sum_log = float(np.sum(np.log(tail)))
    sum_k = float(np.sum(tail))
    n_tail = len(tail)
    mean_tail = max(float(np.mean(tail)), 1.0)

    def neg_log_likelihood(params):
        alpha = float(params[0])
        lam = float(params[1])

        if alpha <= 0 or lam < 0:
            return np.inf

        logZ = _truncated_powerlaw_logZ(alpha, lam, xmin)

        if not np.isfinite(logZ):
            return np.inf

        return alpha * sum_log + lam * sum_k + n_tail * logZ

    starts = [
        (max(float(alpha_start), 0.2), 1e-4),
        (max(float(alpha_start) - 0.5, 0.2), 1.0 / mean_tail),
        (1.0, 0.1 / mean_tail),
    ]

    best = None

    for start in starts:
        result = minimize(
            neg_log_likelihood,
            x0=np.array(start, dtype=float),
            method="L-BFGS-B",
            bounds=[(1e-5, 100.0), (1e-10, 10.0)],
        )

        if result.success and np.isfinite(result.fun):
            if best is None or result.fun < best.fun:
                best = result

    if best is None:
        return None

    alpha_hat = float(best.x[0])
    lambda_hat = float(best.x[1])
    loglik = -float(best.fun)

    return {
        "alpha": alpha_hat,
        "lambda": lambda_hat,
        "loglik": loglik,
        "n_params": 2,
    }


def _add_alternative_model_comparisons(result, x):
    """
    Compare the fitted power law against a discretized lognormal and an
    exponentially truncated power law using log-likelihoods and AIC.

    Positive llr_powerlaw_vs_* means the pure power law has higher likelihood.
    Positive aic_delta_*_minus_powerlaw means the pure power law has lower AIC.
    """
    if not result["fit_success"]:
        return result

    xmin = int(result["xmin"])
    tail = x[x >= xmin]
    ll_powerlaw = float(result["loglik_powerlaw"])

    aic_powerlaw = 2 * 1 - 2 * ll_powerlaw
    aic_values = {"powerlaw": aic_powerlaw}

    lognormal_fit = _fit_discrete_lognormal_tail(tail, xmin)

    if lognormal_fit is not None:
        ll_lognormal = float(lognormal_fit["loglik"])
        aic_lognormal = 2 * lognormal_fit["n_params"] - 2 * ll_lognormal

        result["lognormal_loglik"] = ll_lognormal
        result["lognormal_mu"] = float(lognormal_fit["mu"])
        result["lognormal_sigma"] = float(lognormal_fit["sigma"])
        result["llr_powerlaw_vs_lognormal"] = ll_powerlaw - ll_lognormal
        result["aic_delta_lognormal_minus_powerlaw"] = aic_lognormal - aic_powerlaw

        aic_values["lognormal"] = aic_lognormal

    truncated_fit = _fit_truncated_powerlaw_tail(
        tail=tail,
        xmin=xmin,
        alpha_start=result["alpha"],
    )

    if truncated_fit is not None:
        ll_truncated = float(truncated_fit["loglik"])
        aic_truncated = 2 * truncated_fit["n_params"] - 2 * ll_truncated

        result["truncated_powerlaw_loglik"] = ll_truncated
        result["truncated_powerlaw_alpha"] = float(truncated_fit["alpha"])
        result["truncated_powerlaw_lambda"] = float(truncated_fit["lambda"])
        result["llr_powerlaw_vs_truncated_powerlaw"] = ll_powerlaw - ll_truncated
        result["aic_delta_truncated_powerlaw_minus_powerlaw"] = (
            aic_truncated - aic_powerlaw
        )

        aic_values["truncated_powerlaw"] = aic_truncated

    result["best_aic_model"] = min(aic_values, key=aic_values.get)

    return result


def fit_discrete_powerlaw_csn(
    degrees,
    xmin=None,
    min_tail=50,
    min_unique_tail=2,
    min_xmin=2,
    max_xmin_percentile=90,
    alpha_bounds=(1.000001, 100.0),
    n_bootstrap=250,
    p_threshold=0.10,
    rng=None,
    compare_alternatives=True,
):
    """
    Fit a discrete power-law tail using the Clauset-Shalizi-Newman workflow.

    Intended use in this thesis:
        Call this on the indegree sequence, not on undirected total degree.

    Steps:
        1. Clean integer degree data.
        2. For each candidate xmin, fit the discrete power law by MLE.
        3. Choose xmin by minimum KS distance.
        4. Run a semi-parametric bootstrap goodness-of-fit test.
        5. Optionally compare against a discretized lognormal and an
           exponentially truncated power law.

    The exponent should be reported only if result["reportable"] is True.
    """
    x = _clean_integer_degree_data(degrees)
    n = len(x)

    if rng is None:
        rng = np.random.default_rng()
    elif isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(int(rng))

    result = _fit_discrete_powerlaw_by_ks(
        x,
        xmin=xmin,
        min_tail=min_tail,
        min_unique_tail=min_unique_tail,
        min_xmin=min_xmin,
        max_xmin_percentile=max_xmin_percentile,
        alpha_bounds=alpha_bounds,
    )

    if not result["fit_success"]:
        return result

    # Add fields that may not yet exist.
    result["gof_p"] = float("nan")
    result["gof_pass"] = False
    result["n_bootstrap"] = int(n_bootstrap)
    result["n_bootstrap_success"] = 0

    if int(n_bootstrap) > 0:
        p_value, n_success = _semiparametric_powerlaw_bootstrap_pvalue(
            x,
            observed_fit=result,
            n_bootstrap=int(n_bootstrap),
            rng=rng,
            min_tail=min_tail,
            min_unique_tail=min_unique_tail,
            min_xmin=min_xmin,
            max_xmin_percentile=max_xmin_percentile,
            alpha_bounds=alpha_bounds,
        )

        result["gof_p"] = p_value
        result["n_bootstrap_success"] = n_success

        if np.isfinite(p_value):
            result["gof_pass"] = bool(p_value >= p_threshold)
        else:
            result["gof_pass"] = False

        if result["reason"] == "ok" and result["gof_pass"]:
            result["reportable"] = True
        elif result["reason"] == "ok" and not result["gof_pass"]:
            result["reportable"] = False
            result["reason"] = "powerlaw_rejected_by_bootstrap"
        else:
            result["reportable"] = False
    else:
        result["reportable"] = False
        if result["reason"] == "ok":
            result["reason"] = "gof_not_run"

    # Add empty alternative-comparison fields so the output is stable.
    result.setdefault("lognormal_loglik", float("nan"))
    result.setdefault("lognormal_mu", float("nan"))
    result.setdefault("lognormal_sigma", float("nan"))
    result.setdefault("llr_powerlaw_vs_lognormal", float("nan"))
    result.setdefault("aic_delta_lognormal_minus_powerlaw", float("nan"))

    result.setdefault("truncated_powerlaw_loglik", float("nan"))
    result.setdefault("truncated_powerlaw_alpha", float("nan"))
    result.setdefault("truncated_powerlaw_lambda", float("nan"))
    result.setdefault("llr_powerlaw_vs_truncated_powerlaw", float("nan"))
    result.setdefault("aic_delta_truncated_powerlaw_minus_powerlaw", float("nan"))

    result.setdefault("best_aic_model", None)

    if compare_alternatives:
        result = _add_alternative_model_comparisons(result, x)

    return result