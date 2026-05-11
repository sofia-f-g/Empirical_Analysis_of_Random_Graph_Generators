import numpy as np
from collections import deque
from scipy.optimize import minimize_scalar, minimize
from scipy.special import zeta, logsumexp
from scipy.stats import norm
from empirical_powerlaw import compute_degree_statistics


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

def compute_metrics(V, E, params=None):

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

    # Powerlaw extractions
    # Calculate formal statistics for the indegree.
    indegree_stats = compute_degree_statistics(indegree)

    metrics = {
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

        # Powerlaw
        "indegree_alpha": indegree_stats["alpha_empirical"],
        "indegree_xmin": indegree_stats["xmin_optimal"],
        "indegree_KS": indegree_stats["KS_distance"],
        "indegree_LRT_exp_R": indegree_stats["LRT_exp_R"],
        "indegree_LRT_exp_p": indegree_stats["LRT_exp_p"],
        "indegree_LRT_log_R": indegree_stats["LRT_log_R"],
        "indegree_LRT_log_p": indegree_stats["LRT_log_p"],
        "indegree_LRT_trunc_R": indegree_stats["LRT_trunc_R"],
        "indegree_LRT_trunc_p": indegree_stats["LRT_trunc_p"],
    }

    return metrics
    