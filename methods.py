
import numpy as np
### define our polynomial kernel of order beta
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import time
from scipy.special import gamma as gamma_function
from scipy import integrate
import pandas as pd
from scipy.special import legendre


def ourbetasquare(samples):
    """
    I(X;Y)=O(beta^2)
    """
    x = samples[:,0]
    y = samples[:,1]
    ''' Equation (4.3) of our paper'''
    from sklearn.linear_model import LinearRegression

    reg = LinearRegression().fit(x.reshape(-1, 1), y.reshape(-1, 1))
    beta = float(reg.coef_)
    return np.square(beta)

def ourlogapproximation(samples):
    """
    I(X; Y) \approx log(1/(1 - rho_xy^2/(rho_x * rho_y)))
    """
    x = samples[:,0]
    y = samples[:,1]
    rho_x = np.mean(x * x)
    rho_y = np.mean(y * y)
    rho_xy = np.mean(x * y)

    ''' Equation (4.4) of our paper'''
    return np.log(1 / (1 - (np.square(rho_xy) / (rho_x * rho_y))))


"""
Chow-Liu structure learning and distribution learning
"""
def mi_tester_est(X, n):
    import networkx as nx
    Mi = np.identity(n)
    for i in range(n):
        for j in range(n):
            mi = mutual_info_est(X[:,i], X[:,j])
            Mi[i,j] = mi
    G_mi= nx.from_numpy_array(np.abs(Mi))
    T_est = nx.maximum_spanning_tree(G_mi)
    return nx.to_numpy_array(T_est)

"""
Baseline 5: sklearn: Entropy estimation from k-nearest neighbors distances
"""

def mutual_info(samples):
    """
    Mutual information for a continuous target variable.
    """
    x = samples[:,0]
    y = samples[:,1]
    import warnings
    from sklearn.exceptions import DataConversionWarning
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)

    from sklearn.feature_selection import mutual_info_regression
    mi = mutual_info_regression(y.reshape(-1,1), x.reshape(-1,1))
    return mi

"""
Baseline 6: Pearson correlation
"""

def pearson_corr(samples):
    """
    Pearson's correlation coefficient = covariance(X, Y) / (stdv(X) * stdv(Y)).
    """
    x = samples[:,0]
    y = samples[:,1]
    from scipy.stats import pearsonr
    corr, _ = pearsonr(x, y)
    return corr

"""
Baseline 7: Gaussian MI (test the correctness of other method under Gaussian case)
"""
def mutual_info_est(samples):
    """
    Estimate coefficient beta use OLS
    """
    x = samples[:,0]
    y = samples[:,1]
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(x.reshape(-1, 1), y.reshape(-1, 1))
    beta = float(reg.coef_)

    z = np.stack((x, y), axis=0)
    sigma_x = np.var(x)
    sigma_y = np.var(y)
    sigma_xy = np.cov(z)
    det_xy = np.linalg.det(sigma_xy)

    I_xy = 1 / 2 * np.log(1 + np.square(beta)*np.square(sigma_x)/np.square(sigma_y))
    return I_xy

"""
Baseline: 1. KDE-based estimator

"""
def legendre_kernel_poly(beta):
    '''
    outputs Legendre kernel of order beta as poly1d object
    Note that when beta is even, the kernel is also of order beta+1, by symmetry
    '''
    p = np.poly1d([])
    for order in np.arange(0,beta+1,2):
        p += ((2*order+1)/2)*legendre(order)(0)*legendre(order)
    return p

### define kernel method for density estimation

def KDE(x,kernel,samples,beta,h):
    '''
    this outputs the estimate of the density at x through KDE
    with underlying density beta-Holder-smooth beta >= 0
    setting h = n^{-1/(2*beta+dim)}
    '''
    n,dim = samples.shape
    args = kernel((x.reshape(1,dim)-samples)/h)*(1.*(1.-(np.abs((x.reshape(1,dim)-samples)/h)>1)))
    return (1./(n*(h**dim)))*np.sum(np.prod(args,axis=1))

### KDE + Von mises for entropy

def KDEVM_entropy(samples,beta,gamma):
    n,d = samples.shape

    # we divide the samples in two parts
    samples1, samples2 = samples[:(n//2)], samples[(n//2):]

    # we estimate the density via KDE on samples1
    kernel = legendre_kernel_poly(beta)
    h = gamma*(n**(-1./(2*beta + d)))
    p_hat = (lambda x: KDE(x,kernel,samples1,beta,h)) # estimated density

    # we estimate the entropy with samples2
    temp = np.array([np.abs(p_hat(x)) for x in samples2])
    temp = temp[temp>0]
    est_h = np.mean(-np.log(temp))

    return est_h

### KDE + Von mises for mutual information

def KDEVM_mutual_information(samples,beta,gamma):
    n,d = samples.shape
    samplesxz = np.column_stack((samples[:,0],samples[:,2:]))
    samplesyz = np.column_stack((samples[:,1],samples[:,2:]))
    samplesz = samples[:,2:]

    hxz = KDEVM_entropy(samplesxz,beta,gamma)
    hyz = KDEVM_entropy(samplesyz,beta,gamma)
    hz = KDEVM_entropy(samplesz,beta,gamma)
    hxyz = KDEVM_entropy(samples,beta,gamma)

    return (hxz+hyz-hz-hxyz)

## define monte carlo integrate

def monte_carlo_integrate(f,d,a=0.,b=1.,error=5e-2):
    n_points = int(1/(error**2))
    sample_points = np.random.uniform(a,b,size=(n_points,d))
    return (1./(n_points))*np.sum([f(sample_points[i,:]) for i in range(n_points)])

# KDE + plugin for entropy

def KDEPI_entropy(samples,beta,gamma,x_min=0.,x_max=1.):
    n,d = samples.shape

    # we divide the samples in two parts
    samples1, samples2 = samples[:(n//2)], samples[(n//2):]

    # we estimate the density via KDE on samples1
    kernel = legendre_kernel_poly(beta)
    h = gamma*(n**(-1./(2*beta + d)))

    def g(x):
        t = KDE(x,kernel,samples1,beta,h)
        return -t*np.log(t)
    return monte_carlo_integrate(g,d,x_min,x_max,error=5e-2)


# KDE + plugin for mutual information

def KDEPI_mutual_information(samples,beta,gamma):
    n,d = samples.shape
    samplesxz = np.column_stack((samples[:,0],samples[:,2:]))
    samplesyz = np.column_stack((samples[:,1],samples[:,2:]))
    samplesz = samples[:,2:]

    hxz = KDEPI_entropy(samplesxz,beta,gamma)
    hyz = KDEPI_entropy(samplesyz,beta,gamma)
    hz = KDEPI_entropy(samplesz,beta,gamma)
    hxyz = KDEPI_entropy(samples,beta,gamma)

    return (hxz+hyz-hz-hxyz)

"""
Baseline 2: KNN-Based estimator
"""
### define volume of d-ball

def volume_ball(radius,d):
    return ((np.sqrt(np.pi)*radius)**d)/gamma_function(d/2.+1.)

### define K-nearest neighbors density estimate

def KNNDE(x,samples,k=0):
    '''
    this outputs the estimate of the density at x through KNNDE
    with k nearest neighbors with k = n**(4/(4 + dim)), minimizing MSE
    '''
    n,dim = samples.shape
    if k==0:
        k = int(n**(4/(4 + dim)))
    k = int(k)
    radius = np.sort(np.linalg.norm(samples-x,axis=1))[k-1]
    return k/(n*volume_ball(radius,dim))

# KNN + Von Mises for entropy

def KNNVM_entropy(samples):
    n,d = samples.shape

    # we divide the samples in two parts
    samples1, samples2 = samples[:(n//2)], samples[(n//2):]

    # we estimate the density via KNN on samples1
    p_hat = (lambda x: KNNDE(x,samples1)) # estimated density

    # we estimate the entropy with samples2
    temp = np.array([p_hat(x) for x in samples2])
    temp = temp[temp>0]
    est_h = np.mean(-np.log(temp))

    return est_h

# KNN + Von Mises for mutual information

def KNNVM_mutual_information(samples):
    n,d = samples.shape
    samplesxz = np.column_stack((samples[:,0],samples[:,2:]))
    samplesyz = np.column_stack((samples[:,1],samples[:,2:]))
    samplesz = samples[:,2:]

    hxz = KNNVM_entropy(samplesxz)
    hyz = KNNVM_entropy(samplesyz)
    hz = KNNVM_entropy(samplesz)
    hxyz = KNNVM_entropy(samples)

    return (hxz+hyz-hz-hxyz)

# KNN + plugin for entropy

def KNNPI_entropy(samples,x_min=0.,x_max=1.):
    n,d = samples.shape
    def g(x):
        t = KNNDE(x,samples)
        return -t*np.log(t)
    return monte_carlo_integrate(g,d,x_min,x_max,error=5e-3)


# KNN + plugin for mutual information

def KNNPI_mutual_information(samples):
    n,d = samples.shape
    samplesxz = np.column_stack((samples[:,0],samples[:,2:]))
    samplesyz = np.column_stack((samples[:,1],samples[:,2:]))
    samplesz = samples[:,2:]

    hxz = KNNPI_entropy(samplesxz)
    hyz = KNNPI_entropy(samplesyz)
    hz = KNNPI_entropy(samplesz)
    hxyz = KNNPI_entropy(samples)

    return (hxz+hyz-hz-hxyz)

"""
Baseline 3: OT-Based tests
"""


def SING(data, p_order, ordering, delta, REG=None,pr=False):
    import networkx as nx
    # import cdt
    # cdt.SETTINGS.rpath="/Users/ganassal/opt/anaconda3/envs/causalOT/lib/R/bin/Rscript"
    # from cdt.causality.graph import PC
    # from cdt.causality.graph import GS
    # from cdt.data import load_dataset
    import itertools
    from itertools import combinations as comb
    import TransportMaps as TM
    import TransportMaps.Distributions.FrozenDistributions as FD
    from TransportMaps.Distributions import \
        StandardNormalDistribution, \
        PullBackParametricTransportMapDistribution, \
        DistributionFromSamples
    from TransportMaps.Algorithms.SparsityIdentification.GeneralizedPrecision import gen_precision, var_omega
    from TransportMaps.Algorithms.SparsityIdentification.RegularizedMap import regularized_map
    from TransportMaps.KL.minimize_KL_divergence import minimize_kl_divergence_objective as kl

    #### SING algorithm by Marzouk et al., modified to output the KL value
    r""" Identify a sparse edge set of the undirected graph corresponding to the data.

    Args:
      data: data :math:`{\bf x}_i, i = 1,\dots,n`,
      p_order (int): polynomial order of the transport map representation,
      ordering: scheme used to reorder the variables,
      delta (float): tolerance :math:`\delta`
      REG (dict): regularization dictionary with keys 'type' and 'reweight'
      pr (bool): if True, the algo prints some info along the way
    Returns:
      the generalized precision :math:`\Omega`
    """
    # Print inputs to user
    if pr:
        print("Problem inputs:\n\n dimension = ",data.shape[1],
          "\n number of samples = ",data.shape[0],
          "\n polynomial order = ",p_order,
          "\n ordering type = ", ordering.__class__,
          "\n delta = ",delta,"\n")
    # Initial setup
    # data is (n x d)
    dim = data.shape[1]
    n_samps = data.shape[0]
    nax = np.newaxis
    pi = {}

    # Target density is standard normal
    eta = StandardNormalDistribution(dim)
    # Quadrature type and params
    # 0: Monte Carlo quadrature with n point
    qtype = 0
    qparams = n_samps
    # Gradient information
    # 0: derivative free
    # 1: gradient information
    # 2: Hessian information
    ders = 2
    # Tolerance in optimization
    tol = 1e-5
    # Log stores information from optimization
    log = [ ]

    # All variables are active variables to begin
    active_vars=None
    # Set initial sparsity level
    sparsity = [0]
    sparsityIncreasing = True
    # Number of active variables
    n_active_vars = [(np.power(dim,2) + dim)/2]
    # Create list to store permutations
    perm_list = []
    # Create total_perm vector
    total_perm = np.arange(dim)
    # create counter for iterations
    counter = 0
    while sparsityIncreasing:

        # Define base_density from samples
        pi = DistributionFromSamples(data)

        # Build the transport map (isotropic for each entry)
        # tm_approx = Default_IsotropicIntegratedSquaredTriangularTransportMap(
        #        dim, p_order, active_vars=active_vars)
        # # Construct density T_\sharp \pi
        # tm_density = PushForwardTransportMapDistribution(tm_approx, pi)

        # # SOLVE
        # tm_density.minimize_kl_divergence(eta, qtype=qtype,
        #                                   qparams=qparams,
        #                                   regularization=REG,
        #                                   tol=tol, ders=ders)

        tm_approx, opt_kl = regularized_map(eta, pi, data, qtype, qparams, dim, p_order, active_vars, tol, ders, REG)

        pb_density = PullBackParametricTransportMapDistribution(tm_approx, eta)

        # Compute generalized precision
        omegaHat = gen_precision(pb_density, data)

        # Compute variance of generalized precision
        gp_var = var_omega(pb_density, data)
        # Compute tolerance (matrix)
        tau = delta * np.sqrt(gp_var) # set tolerance as square root of variance
#         print(f'the tolerance matrix after applying delta:{tau}')
#         print(f'Omega hat matrix to be thresholded:{omegaHat}')

        # Save diagonal elements
        omegaHat_diagonal = np.copy(np.diag(omegaHat))
        # Threshold omegaHat
        omegaHat[np.abs(omegaHat) < tau] = 0
        # Put diagonal elements back in
        omegaHat[np.diag_indices_from(omegaHat)] = omegaHat_diagonal

        # Reorder variables and data
        perm1 = ordering.perm(omegaHat)
        # Apply to omegaHat
        omegaHat_temp = omegaHat[:,perm1][perm1,:]
        # Check if ordering would flip if applied again (problem for reverse Cholesky)
        perm2 = ordering.perm(omegaHat_temp)

        if (perm2 == perm1).all():
        #if (perm2 == perm1):
           perm_vect = np.arange(dim) # set to identity permutation
        else:
            perm_vect = perm1

        # Apply to omegaHat
        omegaHat = omegaHat[:,perm_vect][perm_vect,:]
        # Apply re-ordering to data
        data = data[:, perm_vect]

        # Add permutation to perm_list
        perm_list.append(perm_vect)
        # Convolve permutation with total_perm
        total_perm = total_perm[perm_vect]
        inverse_perm = [0] * dim
        for i, p in enumerate(total_perm):
            inverse_perm[p] = i

        # Extract lower triangular matrix
        omegaHatLower = np.tril(omegaHat)
        # Count edges
        edge_count = np.count_nonzero(omegaHatLower) - dim

        # Variable elimination moving from highest node (dim-1) to node 2 (at most)
        for i in range(dim-1,1,-1):
            non_zero_ind  = np.where(omegaHatLower[i,:i] != 0)[0]
            if len(non_zero_ind) > 1:
                co_parents = list(itertools.combinations(non_zero_ind,2))
                for j in range(len(co_parents)):
                    row_index = max(co_parents[j])
                    col_index = min(co_parents[j])
                    omegaHatLower[row_index, col_index] = 1.0

        # Find list of active_vars
        active_vars = []
        for i in range(dim):
            actives = np.where(omegaHatLower[i,:] != 0)
            active_list = list(set(actives[0]) | set([i]))
            active_list.sort(key=int)
            active_vars.append(active_list)

        # Find n_active_vars
        n_active_vars.append(np.sum([len(x) for x in active_vars]))

        # Find current sparsity level
        sparsity.append(n_active_vars[0] - n_active_vars[-1])

        # Set sparsityIncreasing
        if sparsity[-1] <= sparsity[-2]:
            sparsityIncreasing = False

        # Print statement for SING
        counter = counter + 1
        if pr:
            print('\nSING Iteration: ', counter)
            print(f'KL divergence of the recoverd map was {opt_kl}')
            print('Active variables: ', active_vars, '\n  Note variables may be permuted.')
            print('Number of edges: ', edge_count,' out of ', np.floor((dim**2 - dim)/2),' possible')

    # Recovered omega (same variable order as input ordering)
    rec_omega = omegaHat[:,inverse_perm][inverse_perm,:]

    if pr:
        print('\nSING has terminated.')
        print('Total permutation used: ',total_perm)
    graph = np.zeros((dim,dim))
    graph[np.nonzero(rec_omega)] = 1
    if pr:
        print('Recovered graph: \n', graph)
    return rec_omega, graph, opt_kl

### OT-based CI test


def OT_CI_test(data, delta):
    import TransportMaps.Algorithms.SparsityIdentification as SI
    '''
    data should be of shape n*d
    typical value for delta : 0.3 when X,Y,Z are dim 1 each, 0.4 if dim Z = 2
    CAFUL: return Flase (i.e. 0) when X,Y are *not* independent given Z (hence you may need to preprocess 1-result)
    '''
    p_order = 2
    ordering = SI.ReverseCholesky()
    rec_omega, graph, opt_kl = SING(data, p_order, ordering, delta)

    #print(rec_omega)

    return (rec_omega[1,0]==0)

"""
Baseline 4: Neural Network-based mutual information
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, fc1_size, fc2_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(fc1_size, H)
        self.fc2 = nn.Linear(fc2_size, H)
        self.fc3 = nn.Linear(H, 1)  #or what?

    def forward(self, x, y):
        h1 = F.relu(self.fc1(x) + self.fc2(y))
        h2 = self.fc3(h1)
        return h2

### to use this we use I(X;Y |Z) = I(X;Y,Z)-I(X;Z)
def mine(x_sample, y_sample, n_epoch):
    model = Net(int(x_sample.shape[-1]), int(y_sample.shape[-1]))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    plot_loss = []
    nx = x_sample.shape[0]
    part_size = x_sample.shape[0] // n_epoch
    xparts = [x_sample[i:i + part_size, :] for i in range(0, nx, part_size)]
    yparts = [y_sample[i:i + part_size, :] for i in range(0, nx, part_size)]
    for i in range(n_epoch):
        xsample = xparts[i]
        ysample = yparts[i]
        y_shuffle = np.random.permutation(ysample)

        x_sample = Variable(torch.from_numpy(xsample).type(torch.FloatTensor), requires_grad=True)
        y_sample = Variable(torch.from_numpy(ysample).type(torch.FloatTensor), requires_grad=True)
        y_shuffle = Variable(torch.from_numpy(y_shuffle).type(torch.FloatTensor), requires_grad=True)

        pred_xy = model(x_sample, y_sample)
        pred_x_y = model(x_sample, y_shuffle)

        ret = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
        loss = - ret  # maximize
        plot_loss.append(loss.data.numpy())
        model.zero_grad()
        loss.backward()
        optimizer.step()
    plot_x = np.arange(len(plot_loss))
    plot_y = np.array(plot_loss).reshape(-1,)

    return -plot_y[-1]

H = 10 #depth of the layer
