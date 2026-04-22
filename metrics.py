import numpy as np
from collections import deque


### Basic Graph Helpers ###

def build_adjacency(V, E, directed=False):
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
    """ Computes in- and out-degree arrays for all vertices from 
        the undirected edge list """
    
    adjacency = build_adjacency(V, E)
    degrees = np.array([len(neigh) for neigh in adjacency], dtype=int)
    dict = {'id': np.asarray(V['id']), 'degree': degrees}
    return dict
    


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

    """ Computes global or average local clustering """

    adj = build_adjacency(V, E, directed=False)
    adj_sets = [set(neighbors) for neighbors in adj]
    index_of = {int(v_id): idx for idx, v_id in enumerate(np.asarray(V['id']))}

    n = len(adj)
    local_coeffs = []
    triangle_sum = 0
    triplet_sum = 0

    for v in range(n):
        deg = len(adj[v])
        if deg < 2:
            local_coeffs.append(0.0)
            continue

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

    avg_local = sum(local_coeffs) / n if n > 0 else 0.0
    global_cc = triangle_sum / triplet_sum if triplet_sum > 0 else 0.0

    return {'avg_local': float(avg_local), 'global': float(global_cc)}


### Paths / Distances ###

def average_shortest_path_length(V, E):
    """ Average shortest-path length over all connected 
        vertex pairs in an undirected graph """

    adjacency = build_adjacency(V, E)
    n = len(adjacency)
    
    total_dist = 0
    total_pairs = 0

    adj = build_adjacency(V, E)
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
    degrees = deg_seq['degree']

    cc = clustering_coefficient(V, E)
    aspl = average_shortest_path_length(V, E)
    pl = estimate_powerlaw_exponent(degrees)

    return {
        'n_vertices': n_vertices,
        'n_edges': n_edges,
        'degree_mean': float(np.mean(degrees)),
        'degree_max': int(np.max(degrees)),
        'degree_min': int(np.min(degrees)),
        'avg_local_clustering': cc['avg_local'],
        'global_clustering': cc['global'],
        'avg_shortest_path_length': aspl,

        'powerlaw_alpha': pl['alpha'],
        'powerlaw_k_min': pl['k_min'],
        'powerlaw_n_tail': pl['n_tail'],
        'powerlaw_ks': pl['ks'],
        'powerlaw_valid': pl['valid_fit'],
        'powerlaw_reason': pl['reason'],
    }
