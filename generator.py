import numpy as np
### Graph Generation ###

def generate_graph(params, n, seed=None): 
    """ Main wrapper:
        samples vertices and then samples edges according to our model """
    validate_params(params)

    if n is None or int(n) <= 0: 
        raise ValueError('n must be a positive integer')
    n = int(n)

    rng = set_seed(seed)

    dim = int(params.get('dim', 2))
    space_cfg = params.get('space_cfg')
    age_cfg = params.get('age_cfg')

    V = sample_vertices_fixed_n(n, dim, space_cfg, age_cfg, rng)
    E = generate_edges(V, params, rng)
    
    return V, E

### Parameter + RNG utilities ###
def validate_params(params):
    
    """ Checks that parameters are in valid ranged ranges
        and raises clear error if not """
    
    return None #raises error if wrong 

def set_seed(seed):
    """ Creates and returns reproducible RNG object used by all sampling steps """
    
    return np.random.default_rng(seed)


### Vertex sampling ###
def sample_vertices_fixed_n(n, d, space_cfg, age_cfg, rng):
    """ Samples n vertices with positions in our chosen spatial window/torus
        and birth times (0, 1) returning a vertex table/array """
    
    if n < 0:
        raise ValueError('n must be non-negative')
    elif n == 0:
        raise ValueError('Trivial graph, n=0 implies no vertices')
    if d <= 0:
        raise ValueError('d must be a positive integer')
    
    space_cfg = {} if space_cfg is None else dict(space_cfg)
    age_cfg = {} if age_cfg is None else dict(age_cfg)
    
    # --- Sample spatial positions --- # 
    # Default to the d-dimensional torus [-0.5, 0.5)^d unless explicit bounds are given

    bounds = space_cfg.get('bounds')
    if bounds is None:
        low = np.full(d, -0.5, dtype=float)
        high = np.full(d, 0.5, dtype=float)
    else:
        bounds = np.asarray(bounds, dtype=float)
        if bounds.shape != (d,2):
            raise ValueError('space_cfg[bounds] must have shape (d,2)')

        low = bounds[:, 0]
        high = bounds[:,1]

        if np.any(high <= low):
            raise ValueError('Each spatial interval must have high > low')
        
    # --- Uniformly distributes vertices within space_cfg --- #
    positions = rng.uniform(low=low, high=high, size=(n,d))

    # --- Sample ages / 'birth times' --- #
    age_distribution = 'uniform'  # Kan ändras till age_cfg.get('distribution', 'uniform')

    t_min = float(age_cfg.get('min', 0.0))
    t_max = float(age_cfg.get('max', 1.0))

    if not (t_max > t_min):
        raise ValueError('age_cfg must have max > min')
    
    birth_times = rng.uniform(t_min, t_max, size=n)

    # Sort to simplify edge-generation later
    order = np.argsort(birth_times)
    birth_times = birth_times[order]
    positions = positions[order]

    V = {
        'id': np.arange(n, dtype=int),
        'pos': positions,
        'birth_time': birth_times
    }
    
    return V

def sample_vertices_ppp(lambda_param, x, y, z, d, rng):

    """ Samples vertices from a Poisson point process in space x time,
        closer method to the original paper """

    # return V


### Connection rule

def torus_distance(x, y, L=1.0):                                                                                                     
    """Toroidal (wrap-around) distance between two position vectors.                                                                                                                                                                                                    

    Parameters                                                                                                                       
    ----------
    x, y : array-like, shape (d,)
        Spatial coordinates of two vertices.
    L : float
        Side length of the spatial domain. Defaults to 1.0 (unit torus).

    Returns
    -------
    float
        Toroidal L2 distance in [0, L*sqrt(d)/2].
    """

    # distance in the torus is the minimum of the direct distance and the wrap-around distance in each dimension
    # from Peter Gracar's paper, section 2

    diff = np.abs(np.asarray(x, dtype=float) - np.asarray(y, dtype=float))
    diff = np.minimum(diff, L - diff)   # wrap: take shorter path in each dim
    
    return float(np.sqrt(np.sum(diff ** 2)))

def phi_profile_function(r, profile_cfg):
    """ Computes the profile value as a function of distance-like input """
    """ We begin by a hard cut-off function, could be modified later """

    # Profile function from section 2 (iv) in Peter Gracar's paper
    # phi(x) = (1/(2a)) * 1_[0,a](x)
    # Right now we don't know why a must be >= 1/2. research this!

    a = float(profile_cfg.get('a', 0.5))

    if a < 0.5:
        raise ValueError('profile_cfg[a] must satisfy a >= 0.5')
    
    if 0 <= r <= a: 
        return 1.0 / (2.0 * a)
    else: 
        return 0.0


def connection_prob(v_i, v_j, params):
    """ Implements the model's connection probability for a directed edge i -> j,
        using ages, distance and parameters like beta, gamma and phi """
    
    # --- Read parameters from params dictionary --- #
    beta = float(params['beta'])
    gamma = float(params['gamma'])
    d = int(params['dim'])
    profile_cfg = params.get('profile_cfg', {})
        
    # --- Birth times and positions of vertices --- #
    t_i = v_i['birth_time']
    t_j = v_j['birth_time']
    pos_i = v_i['pos']
    pos_j = v_j['pos']

    t_young = max(t_i, t_j)
    t_old = min(t_i, t_j)

    r = (t_young * torus_distance(pos_i, pos_j)**d) / (beta * (t_young/t_old)**gamma)

    return phi_profile_function(r, profile_cfg)
    


### Edge Generation ### 

def generate_edges(V, params, rng):

    """ Builds the edge list by iterating over candidate vertex pairs and
        sampling Bernoulli edges with function connection_prob """

    n = len(V['id'])
    edges = []

    for i in range(1, n): # the younger vertex (starts at 1, since vertex 0 has no older vertices to connect to)
        v_young = {'pos': V['pos'][i], 'birth_time': V['birth_time'][i]}

        for j in range(i): # j = every older vertex than i (only consider j < i to ensure direction from younger to older beacause V is sorted by birth time)
            v_older = {'pos': V['pos'][j], 'birth_time': V['birth_time'][j]}

            p_ij = connection_prob(v_young, v_older, params)

            if p_ij < 0 or p_ij > 1:
                raise ValueError(f'Connection probability must be in [0,1], got {p_ij}')
            
            if rng.uniform() < p_ij:
                edges.append((V['id'][i], V['id'][j]))  # directed edge from younger to older 
    
    if len(edges) == 0:
      return np.empty((0, 2), dtype=int)
    
    return np.asarray(edges, dtype=int)
