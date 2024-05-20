import networkx as nx
import numpy as np

import config
p = config.setup()
lgr = p.logger

import random

seed = 123
random.seed(seed)
np.random.seed(seed)


"""
Synthetic data: generate power law distribution 
"""
def generate_power_law_4d(n,beta=3,t1=0.1,t2=0.1,txy=0.3):
    '''
    input: beta parameter of Pow(beta+0.15), number of samples n, t1 and t2
    t1 (resp. t2): parameter for correlation of U_x, U_y with U_z1 (resp. U_z2)
    txy is the intrinsic correlation of x with y
    X corresponds to first variable, Y the second, Z the two last
    outputs: samples of shape (n,d) with d = 4 with law Pow(beta+0.15)
    '''
    assert (t1 +t2 + txy <= 1)
    U_z1 = np.random.rand(n)
    U_z2 = np.random.rand(n)
    temp = np.random.rand(n)
    U_x = (temp<=t1)*U_z1 + (temp>t1)*(temp<t1+t2)*U_z2 + (temp>t1+t2)*np.random.rand(n)
    temp = np.random.rand(n)
    U_y = (temp<=t1)*U_z1 + (temp>t1)*(temp<t1+t2)*U_z2 + (temp>t1+t2)*(temp<t1+t2+txy)*U_x + (temp>t1+t2+txy)*np.random.rand(n)

    exponent = 1./(beta + 1.15)
    X = np.power(U_x,exponent)
    Y = np.power(U_y,exponent)
    Z1 = np.power(U_z1,exponent)
    Z2 = np.power(U_z2,exponent)

    return np.column_stack((X,Y,Z1,Z2))


def generate_syn_data(n, beta, distribution):
    """
    n: the number of samples
    beta: coefficient parameter
    Z1, Z2: to be used to adapt other CI test algorithms.
            (Z1, Z2) \indep with X and Y
    Distributions in $G$
    """
    # Gaussian distribution
    Z1 = np.random.rand(n)
    Z2 = np.random.rand(n)
    if distribution == 'gaussian':
        mu, sigma = 0, 1
        X = np.random.normal(mu, sigma, n)
        Y = beta * X + np.random.normal(mu, sigma, n)
    # Gumbel distribution
    elif distribution == 'gumbel':
        mu, scale = 0, 1
        eta_x = np.random.gumbel(mu, scale, n)
        X = eta_x - np.mean(eta_x)
        eta_y = np.random.gumbel(mu, scale, n)
        Y = beta * X + eta_y - np.mean(eta_y)
    # Beta distribution
    elif distribution == 'beta_distribution' :
        a, b = 1, 5
        eta_x = np.random.beta(a, b, n)
        eta_y = np.random.beta(a, b, n)
        X = eta_x - np.mean(eta_x)
        Y = beta * X + eta_y - np.mean(eta_y)
    # Fold Gaussian
    elif distribution == 'foldgaussian':
        from scipy.stats import foldnorm
        c = 1.95
        eta_x = foldnorm.rvs(c, size=n)
        X = eta_x - np.mean(eta_x)
        eta_y = foldnorm.rvs(c, size=n)
        Y = beta * X + eta_y - np.mean(eta_y)
    #  irwin-hall distribution
    elif distribution == 'irwin-hall':
        X = np.random.uniform(0, 1, n) + np.random.uniform(0, 1, n)  # + np.random.uniform(0, 1, n)  + np.random.uniform(0, 1, n)
        Y = beta * X + np.random.uniform(0, 1, n) + np.random.uniform(0, 1,n)  # + np.random.uniform(0, 1, n)  + np.random.uniform(0, 1, n)
    # Laplace distribution
    elif distribution == 'laplace':
        from scipy.stats import laplace
        X = laplace.rvs(size=n)
        Y = beta * X + laplace.rvs(size=n)
    # Rayleigh distribution
    elif distribution == 'rayleigh':
        from scipy.stats import rayleigh
        eta_x = rayleigh.rvs(size=n)
        X = eta_x - np.mean(eta_x)
        eta_y = rayleigh.rvs(size=n)
        Y = beta * X + eta_y - np.mean(eta_y)

    # Weibull distribution
    elif distribution == 'weibull':
        eta_x = np.random.weibull(a, n)
        X = eta_x - np.mean(eta_x)
        eta_y = np.random.weibull(a, n)
        Y = beta * X + eta_y - np.mean(eta_y)

    # student-t distribution
    elif distribution == 'student-t':
        df = 2.74
        from scipy.stats import t
        eta_x = t.rvs(df, size=n)
        X = eta_x - np.mean(eta_x)
        eta_y = t.rvs(df, size=n)
        Y = beta * X + eta_y - np.mean(eta_y)
    """
    Distributions not in $G$
    """
    # exponential distribution
    if distribution == 'exponential':
        eta_x = np.random.exponential(2.0, n)
        X = eta_x - np.mean(eta_x)
        eta_y = np.random.exponential(2.0, n)
        Y = beta * X + eta_y - np.mean(eta_y)
    # uniform distribution
    if distribution == 'uniform':
        X = np.random.uniform(-5, 5, n)
        Y = beta * X + np.random.uniform(-5, 5, n)
    # Cauchy distribution
    if distribution == 'cauchy':
        from scipy.stats import cauchy
        X = cauchy.rvs(size=n)
        Y = beta * X + cauchy.rvs(size=n)
    # Chi-square distribution
    if distribution == 'chisquare':
        df = 1
        from scipy.stats import chi2
        eta_x = chi2.rvs(df, size=1000)
        eta_y = chi2.rvs(df, size=1000)
        X = eta_x - np.mean(eta_x)
        Y = beta * X + eta_y - np.mean(eta_y)
    return np.column_stack((X, Y))


"""
p.n: number of nodes
p.s: number of samples
"""
#n, s = p.n, p.s
#n = 5    # number of nodes
#s = 100000  # number of samples

"""
Function: Synthetic data generation. Move it to data.py
"""
def ER(p):
    '''
    simulate Erdos-Renyi (ER) DAG through networkx package

    Arguments:
        p: argparse arguments

    Uses:
        p.n: number of nodes
        p.d: degree of graph
        p.rs: numpy.random.RandomState

    Returns:
        B: (n, n) numpy.ndarray binary adjacency of ER DAG
    '''
    n, d = p.n, p.d
    p = float(d)/(n-1)

    G = nx.generators.erdos_renyi_graph(n=n, p=p, seed=5)

    U = nx.to_numpy_matrix(G)

    B = np.tril(U, k=-1)
    return B

def RT(p):
    '''
    simulate Random Tree DAG through networkx package
    Arguments:
    Arguments:
        p: argparse arguments
    Uses:
        p.n: number of nodes
        p.s: number of samples
        p.rs: numpy.random.RandomState
    '''
    n, s = p.n, p.s
    G = nx.random_tree(n, seed=15)
    U = nx.to_numpy_array(G)
    B = np.tril(U, k=-1)

    A = np.zeros((n, n))
    root = np.random.randint(n)
    for i in range(n):
        for j in range(n):
            # i,j not edge, continue
            if U[i, j] == 0:
                continue;
            elif j in nx.shortest_path(G, root, i):
                A[j, i] = 1
            elif i in nx.shortest_path(G, root, j):
                A[i, j] = 1

    return B


def WDAG(B, p):
    """
    Generate weighted DAG
    """
    A = np.zeros(B.shape)
    s = 1 # s is the scalling factor
    R = ((-0.5 * s, -0.1 * s), (0.1 * s, 0.5 * s))
    rs = np.random.RandomState(100) # generate random stets based on a seed
    S = rs.randint(len(R), size=A.shape)

    for i, (l, h) in enumerate(R):
        U = rs.uniform(low=l, high=h, size=A.shape)
        A += B * (S == i) * U

    return A

def SEM(A, p):
    '''
    simulate samples from linear structural equation model (SEM) with specified type of noise.
    Arguments:
        A: (n, n) weighted adjacency matrix of DAG
        p: argparse arguments
    Uses:
        p.n: number of nodes
        p.s: number of samples
        p.rs (numpy.random.RandomState): Random number generator
        p.tn: type of noise, options: ev, nv, exp, gum
            ev: equal variance
            uv: unequal variance
            exp: exponential noise
            gum: gumbel noise
    Returns:
        numpy.ndarray: (s, n) data matrix.
    '''
    n, s = p.n, p.s
    r = np.random.RandomState(500)
    def _SEM(X, I):

        '''
        simulate samples from linear SEM for the i-th vertex
        Arguments:
            X (numpy.ndarray): (s, number of parents of vertex i) data matrix
            I (numpy.ndarray): (n, 1) weighted adjacency vector for the i-th node
        Returns:
            numpy.ndarray: (s, 1) data matrix.
        '''

        N = np.random.uniform(low=-1.0, high=1.0, size=s)
        # TODO add another centralized distribution EG. Lapla...
        #mu, sigma = 0, 1
        #N = r.normal(scale=r.uniform(low=1.0, high=2.0), size=s)
        #N = np.random.normal(mu, sigma, size=s)
        loc, scale = 0., 2.
        #N = np.random.laplace(loc, scale, size=s)
        return X @ I + N

    X = np.zeros([s, n])
    G = nx.DiGraph(A)

    ''' Radomly set ill conditioned nodes'''
    nodes = np.arange(n)
    np.random.shuffle(nodes)


    for v in list(nx.topological_sort(G)):
        P = list(G.predecessors(v))
        X[:, v] = _SEM(X[:, P], A[P, v])

    return X
