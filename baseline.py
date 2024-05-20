import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from scipy.special import legendre

from methods import mutual_info, pearson_corr, mutual_info_est
sample = [500, 600, 700, 800, 900, 1000, 2000, 10000, 100000]
beta = 0
def mibinary(value):
  return (value > .05).astype(int)

MI = []
COR = []
MI_EST = []
MI_EST2 = []
MI_EST3 = []

a, b = 2, 5
mu, scale = 0, 2
# x = np.random.normal(0, 1, n)
# y = beta * x + np.random.normal(0, 1, n)


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
    #samplesxy = np.column_stack((samples[:,0],samples[:,1:]))
    samplesx = samples[:, 0]
    samplesy = samples[:, 1]

    hx = KDEVM_entropy(samplesx,beta,gamma)
    hy = KDEVM_entropy(samplesy,beta,gamma)
    hxy = KDEVM_entropy(samples,beta,gamma)

    return (hx+hy-hxy)

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

for n in sample:
    # x = np.random.beta(a, b, n)
    # y = beta * x + np.random.beta(a, b, n)
    eta_x = np.random.gumbel(mu, scale, n)
    x = eta_x - np.mean(eta_x)
    eta_y = np.random.gumbel(mu, scale, n)
    y = beta * x + eta_y - np.mean(eta_y)
    rho_x = np.mean(x*x)
    rho_y = np.mean(y*y)
    rho_xy = np.mean(x*y)
    '''Entropy estimation from k-nearest neighbors distances '''
    mi = mutual_info(x, y)
    MI.append(mi)
    #print('mi = ', mi)

    ''' Pearson correlation'''
    cor = pearson_corr(x, y)
    COR.append(cor)
    #print('cor =', cor)

    ''' Gaussian MI (test the correctness of other method under Gaussian case) '''
    mi_est = mutual_info_est(x, y)
    MI_EST.append(mi_est)
    #print('mi_est_gaussian', mi_est)

    ''' Equation (4.4) of our paper'''
    mi_est2 = np.log(1/(1-(np.square(rho_xy)/(rho_x * rho_y))))
    MI_EST2.append(mi_est2)
    #print('mi_general 1 = ', mi_est2)

    ''' Equation (4.3) of our paper'''
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(x.reshape(-1, 1), y.reshape(-1, 1))
    beta = float(reg.coef_)
    mi_est3 = np.square(beta)
    MI_EST3.append(mi_est3)
    #print('mi_general 2 ', mi_est3)

    ''' KDE + Von mises estimator'''


MI_alt = []
COR_alt = []
MI_EST_alt = []
MI_EST2_alt = []
MI_EST3_alt = []

a, b = 2, 5
mu, scale = 0, 2
beta_alt = 0.5

for n in sample:
    # x = np.random.beta(a, b, n)
    # y = beta * x + np.random.beta(a, b, n)
    eta_x = np.random.gumbel(mu, scale, n)
    x = eta_x - np.mean(eta_x)
    eta_y = np.random.gumbel(mu, scale, n)
    y = beta_alt * x + eta_y - np.mean(eta_y)
    rho_x = np.mean(x*x)
    rho_y = np.mean(y*y)
    rho_xy = np.mean(x*y)
    '''Entropy estimation from k-nearest neighbors distances '''
    mi = mutual_info(x, y)
    MI_alt.append(mi)
    #print('mi = ', mi)

    ''' Pearson correlation'''
    cor = pearson_corr(x, y)
    COR_alt.append(cor)
    #print('cor =', cor)

    ''' Gaussian MI (test the correctness of other method under Gaussian case) '''
    mi_est = mutual_info_est(x, y)
    MI_EST_alt.append(mi_est)
    #print('mi_est_gaussian', mi_est)

    ''' Equation (4.4) of our paper'''
    mi_est2 = np.log(1/(1-(np.square(rho_xy)/(rho_x * rho_y))))
    MI_EST2_alt.append(mi_est2)
    #print('mi_general 1 = ', mi_est2)

    ''' Equation (4.3) of our paper'''
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(x.reshape(-1, 1), y.reshape(-1, 1))
    beta = float(reg.coef_)
    mi_est3 = np.square(beta)
    MI_EST3_alt.append(mi_est3)
    #print('mi_general 2 ', mi_est3)

import matplotlib.pyplot as plt
X = np.arange(-1, 1.03, 0.1)
plt.figure(figsize=(10,6))

plt.plot(sample, MI, 'r-', label='Entropy estimation')
#plt.plot(sample, COR, 'b-', label='Correlation')
plt.plot(sample, MI_EST, 'g-', label='Gaussian MI')
plt.plot(sample, MI_EST2, 'c-', label='Ours')
plt.plot(sample, MI_EST3, 'm-', label='Ours beta^2')

plt.plot(sample, MI_alt, 'r-', linestyle = 'dashed', label='Entropy estimation')
plt.plot(sample, MI_EST_alt, 'g-', linestyle = 'dashed', label='Gaussian MI')
plt.plot(sample, MI_EST2_alt, 'c-', linestyle = 'dashed', label='Ours')
plt.plot(sample, MI_EST3_alt, 'm-', linestyle = 'dashed', label='Ours beta^2')
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)
plt.legend( borderpad=2, prop={'size': 16})
plt.title("Gumbel distribution", fontsize=20)
plt.xlabel('number of samples', fontsize=18)

plt.grid()
plt.show()