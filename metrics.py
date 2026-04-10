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

def estimate_powerlaw_exponent(deg, k_min=None):
    """ Estimates the power-law tail exponent alpha using Maximum Likelihood
        Estimation (Hill estimator): alpha = 1 + n * [sum(ln(k_i/k_min))]^-1

        k_min is the lower cutoff for the power-law tail. If None, it is chosen
        automatically by minimizing the Kolmogorov-Smirnov distance between the
        empirical tail CDF and the fitted power-law CDF.

        Returns a dict with:
            alpha   - estimated tail exponent
            k_min   - lower cutoff used
            n_tail  - number of vertices in the tail (degree >= k_min)
    """

    degrees = np.asarray(deg, dtype=float)
    degrees = degrees[degrees >= 1]  # exclude isolates (degree 0)

    if len(degrees) < 2:
        return {'alpha': float('nan'), 'k_min': None, 'n_tail': 0}

    if k_min is None:
        # Search over candidate k_min values (up to the 95th percentile to keep
        # enough tail points for a reliable fit)
        candidates = np.unique(degrees.astype(int))
        candidates = candidates[candidates <= np.percentile(degrees, 95)]

        best_k_min = candidates[0]
        best_alpha = float('nan')
        best_D = float('inf')

        for k in candidates:
            tail = degrees[degrees >= k]
            n = len(tail)
            if n < 2:
                continue

            log_sum = np.sum(np.log(tail / k))
            if log_sum == 0:
                continue
            alpha_k = 1.0 + n / log_sum

            # KS distance between empirical and theoretical CDF
            sorted_tail = np.sort(tail)
            empirical_cdf = np.arange(1, n + 1) / n
            theoretical_cdf = 1.0 - (k / sorted_tail) ** (alpha_k - 1)
            D = np.max(np.abs(empirical_cdf - theoretical_cdf))

            if D < best_D:
                best_D = D
                best_k_min = k
                best_alpha = alpha_k

        k_min = best_k_min

    tail = degrees[degrees >= k_min]
    n = len(tail)

    if n < 2:
        return {'alpha': float('nan'), 'k_min': int(k_min), 'n_tail': n}

    log_sum = np.sum(np.log(tail / k_min))
    if log_sum == 0:
        return {'alpha': float('nan'), 'k_min': int(k_min), 'n_tail': int(n)}
    alpha = 1.0 + n / log_sum

    return {'alpha': float(alpha), 'k_min': int(k_min), 'n_tail': int(n)}


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
    }
