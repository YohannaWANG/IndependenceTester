from data import generate_syn_data, generate_power_law_4d
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from methods import ourbetasquare, ourlogapproximation, KDEVM_mutual_information,KDEPI_mutual_information

def mibinary(value):
  return (value > .05).astype(int)


N = [50,100,250,500,1000,1500,2000,2500, 3000]
#N= [500, 600, 700, 800, 900, 1000, 2000, 10000, 100000 ]
num_of_experiments = 100
distribution = 'gaussian'
beta = 3
gamma = 0.35
'''
Results for 
'''
MI_mean_null = []
MI_std_null = []
MI_mean_alt = []
MI_std_alt = []

MI2_mean_null = []
MI2_std_null = []
MI2_mean_alt = []
MI2_std_alt = []

KDEVM_mean_null = []
KDEVM_std_null = []
KDEVM_mean_alt = []
KDEVM_std_alt = []
''' 
for n in N:
    MI = []
    MI2 = []
    KDEVM = []
    for j in range(num_of_experiments):
        samples = generate_syn_data(n, 0, distribution) # beta = 0, X \indep Y
        """
        Our methods: I(X;Y)=O(beta^2)
        """
        mi_est = ourbetasquare(samples)
        MI.append(mi_est)
        """
        Our methods: I(X; Y) approx log(1/(1 - rho_xy^2/(rho_x * rho_y)))
        """
        mi2_est = ourlogapproximation(samples)
        MI2.append(mi2_est)

        """
        Baseline: KDE + VM
        """
        kdevm = KDEVM_mutual_information(samples, beta, gamma)
        # print('kdevm', kdevm)
        KDEVM.append(kdevm)
        """
        Baseline: KDE + Plugin estimator
        """
        #kdepi = KDEPI_mutual_information(samples, beta, gamma)

    MI_mean_null.append(np.mean(MI))
    MI_std_null.append(np.std(MI))

    MI2_mean_null.append(np.mean(MI2))
    MI2_std_null.append(np.std(MI2))

    KDEVM_mean_null.append(np.mean(KDEVM))
    KDEVM_std_null.append(np.std(KDEVM))
'''
for n in N:
    MI = []
    MI2 = []
    KDEVM = []
    for j in range(num_of_experiments):
        #samples = generate_syn_data(n, 0.3, distribution) # beta = 0, X \not \indep Y
        samples = generate_power_law_4d(n, beta=3, t1=0.1, t2=0.1, txy=0.3)
        """
        Our methods: I(X;Y)=O(beta^2)
        """
        mi_est = ourbetasquare(samples)
        MI.append(mi_est)
        """
        Our methods: I(X; Y) approx log(1/(1 - rho_xy^2/(rho_x * rho_y)))
        """
        mi2_est = ourlogapproximation(samples)
        MI2.append(mi2_est)
        """
        Baseline: KDE + VM
        """
        kdevm = KDEVM_mutual_information(samples, beta, gamma)
        KDEVM.append(kdevm)

    MI_mean_alt.append(np.mean(MI))
    MI_std_alt.append(np.std(MI))

    MI2_mean_alt.append(np.mean(MI2))
    MI2_std_alt.append(np.std(MI2))

    KDEVM_mean_alt.append(np.mean(KDEVM))
    KDEVM_std_alt.append(np.std(KDEVM))

print("MI_mean_null  = ", MI_mean_null)
print("MI_std_null = ", MI_std_null)

print("MI2_mean_alt  = ", MI2_mean_alt)
print("MI2_std_alt = ", MI2_std_alt)

print("KDEVM_mean_null", KDEVM_mean_null)
print("KDEVM_std_null", KDEVM_std_null)

print("KDEVM_mean_alt", KDEVM_mean_alt)
print("KDEVM_std_alt", KDEVM_std_alt)

import matplotlib.pyplot as plt
plt.figure(figsize=(18, 10))
plt.errorbar(N, MI_mean_null, yerr=MI_std_null, color='red', label='Ours(beta^2) Null', linewidth=5)
plt.errorbar(N, MI_mean_alt, yerr=MI_std_alt, color='red', label='Ours(beta^2) Alt', linestyle='--', dashes=(5, 5), linewidth=5)

plt.errorbar(N, MI2_mean_null, yerr=MI2_std_null, color='blue', label='Ours(log(*)) Null', linewidth=5)
plt.errorbar(N, MI2_mean_alt, yerr=MI2_std_alt, color='blue', label='Ours(log(*)) Alt', linestyle='--', dashes=(5, 5), linewidth=5)

plt.errorbar(N, KDEVM_mean_null, yerr=KDEVM_std_null,  color='green', alpha=1.0, label='KDEVM Null', linewidth=4)
plt.errorbar(N, KDEVM_mean_alt, yerr=KDEVM_std_alt,  color='green', alpha=1.0, linestyle='--', dashes=(5, 5), label='KDEVM Alt', linewidth=4)
#plt.errorbar(sample, SHD_PCtree_mean_1, yerr=SHD_PCtree_std_1, color='red', label='PC-Tree', linestyle='dotted', linewidth=5)
#plt.errorbar(sample, SHD_PCtree_mean_2, yerr=SHD_PCtree_std_2, color='red', label='PC-Tree', linewidth=7)
#plt.errorbar(sample, SHD_Pearson_mean, yerr=SHD_Pearson_std, linestyle='--', dashes=(5, 8), label='Pearson', linewidth=4)
#plt.errorbar(sample, SHD_PC_mean, yerr=SHD_PC_std,  color='green', alpha=1.0, label='PC (0.0001)', linewidth=4)
#plt.errorbar(sample, SHD_PC_mean_1, yerr=SHD_PC_std_1,  color='green', label='PC', linewidth=5) #(0.001)
#plt.errorbar(sample, SHD_PC_mean_2, yerr=SHD_PC_std_2,  color='green', label='PC',  linewidth=7) #(0.001)
#plt.errorbar(sample, SHD_PC_mean_2, yerr=SHD_PC_std_2,  color='green', alpha=0.6, label='PC(0.01)', linewidth=4)
#plt.errorbar(sample, SHD_PC_mean_3, yerr=SHD_PC_std_3, label='PC(0.05)', linewidth=4)
#plt.errorbar(sample, SHD_FCI_mean, yerr=SHD_FCI_std,  color='red', alpha=1.0, linestyle='--', dashes=(5, 5), label='FCI (0.0001)', linewidth=4)
#plt.errorbar(sample, SHD_FCI_mean_1, yerr=SHD_FCI_std_1, color='red', alpha=0.8, linestyle='--', dashes=(5, 5), label='FCI (0.001)', linewidth=4)
#plt.errorbar(sample, SHD_FCI_mean_2, yerr=SHD_FCI_std_2, color='red', alpha=0.6, linestyle='--', dashes=(5, 5), label='FCI (0.01)', linewidth=4)
#plt.errorbar(sample, SHD_FCI_mean_3, yerr=SHD_FCI_std_3, linestyle='--', dashes=(5, 5), label='FCI (0.05)', linewidth=4)
#plt.errorbar(sample, SHD_GES_mean_1, yerr=SHD_GES_std_1, color='purple', label='GES', marker='o', linewidth=5)
#plt.errorbar(sample, SHD_GES_mean_2, yerr=SHD_GES_std_2, color='purple', label='GES', alpha=0.6, linewidth=7)

plt.legend(loc='upper right', fontsize=22)
plt.tick_params(axis='both', which='major', labelsize=25)
plt.xlabel('Sample size', fontsize=30)
plt.ylabel('MI', fontsize=35)
plt.grid(True)
#plt.savefig("gaussian_100_shd_noniid.pdf", format="pdf", bbox_inches="tight")
plt.show()

''' 
plt.plot(N, MI, 'r-', label='Entropy estimation')
#plt.plot(sample, COR, 'b-', label='Correlation')
plt.plot(N, MI_EST, 'g-', label='Gaussian MI')
plt.plot(N, MI_EST2, 'c-', label='Ours')
plt.plot(N, MI_EST3, 'm-', label='Ours beta^2')

plt.plot(N, MI_alt, 'r-', linestyle = 'dashed', label='Entropy estimation')
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
'''