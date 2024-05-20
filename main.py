from data import generate_syn_data, generate_power_law_4d
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from methods import ourbetasquare, ourlogapproximation, KDEVM_mutual_information,KDEPI_mutual_information, KNNVM_mutual_information, KNNPI_mutual_information

def mibinary(value):
  return (value > .05).astype(int)


N = 100, 300, 500, 800#,1500,2000,2500, 3000
#N= [500, 600, 700, 800, 900, 1000, 2000, 10000, 100000 ]
num_of_experiments = 50
distribution = 'beta_distribution '#'gaussian'
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


KDEPI_mean_null = []
KDEPI_std_null = []
KDEPI_mean_alt = []
KDEPI_std_alt = []

KNNVM_mean_null = []
KNNVM_std_null = []
KNNVM_mean_alt = []
KNNVM_std_alt = []

KNNPI_mean_null = []
KNNPI_std_null = []
KNNPI_mean_alt = []
KNNPI_std_alt = []

for n in N:
    print('samples = ', n)
    MI = []
    MI2 = []
    KDEVM = []
    KDEPI = []
    KNNVM = []
    KNNPI = []
    for j in range(num_of_experiments):
        #print('experiments ', j)
        samples = generate_syn_data(n, 0, 'beta_distribution') # beta = 0, X \indep Y
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
        """
        Baseline: KDE + Plugin estimator
        """
        kdepi = KDEPI_mutual_information(samples, beta, gamma)
        KDEPI.append(kdepi)
        """
        Baseline: KNN + VM
        """
        knnvm = KNNVM_mutual_information(samples)
        KNNVM.append(knnvm)
        """
        Baseline: KNN + PI
        """
        knnpi = KNNPI_mutual_information(samples)
        KNNPI.append(knnpi)

    MI_mean_null.append(np.mean(MI))
    MI_std_null.append(np.std(MI))

    MI2_mean_null.append(np.mean(MI2))
    MI2_std_null.append(np.std(MI2))

    KDEVM_mean_null.append(np.mean(KDEVM))
    KDEVM_std_null.append(np.std(KDEVM))

    KDEPI_mean_null.append(np.mean(KDEPI))
    KDEPI_std_null.append(np.std(KDEPI))

    KNNVM_mean_null.append(np.mean(KNNVM))
    KNNVM_std_null.append(np.std(KNNVM))

    KNNPI_mean_null.append(np.mean(KNNPI))
    KNNPI_std_null.append(np.std(KNNPI))

for n in N:
    print('samples = ', n)
    MI = []
    MI2 = []
    KDEVM = []
    KDEPI= []
    KNNVM = []
    KNNPI = []
    for j in range(num_of_experiments):
        samples = generate_syn_data(n, 0.3, 'beta_distribution') # beta = 0, X \not \indep Y
        #samples = generate_power_law_4d(n, beta=3, t1=0.1, t2=0.1, txy=0.3)
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
        """
        Baseline: KDE + Plugin estimator
        """
        kdepi = KDEPI_mutual_information(samples, beta, gamma)
        KDEPI.append(kdepi)
        """
        Baseline: KNN + VM
        """
        knnvm = KNNVM_mutual_information(samples)
        KNNVM.append(knnvm)
        """
        Baseline: KNN + PI
        """
        knnpi = KNNPI_mutual_information(samples)
        KNNPI.append(knnpi)

    MI_mean_alt.append(np.mean(MI))
    MI_std_alt.append(np.std(MI))

    MI2_mean_alt.append(np.mean(MI2))
    MI2_std_alt.append(np.std(MI2))

    KDEVM_mean_alt.append(np.mean(KDEVM))
    KDEVM_std_alt.append(np.std(KDEVM))

    KDEPI_mean_alt.append(np.mean(KDEPI))
    KDEPI_std_alt.append(np.std(KDEPI))

    KNNVM_mean_alt.append(np.mean(KNNVM))
    KNNVM_std_alt.append(np.std(KNNVM))

    KNNPI_mean_alt.append(np.mean(KNNPI))
    KNNPI_std_alt.append(np.std(KNNPI))



print("MI2_mean_null  = ", MI2_mean_null)
print("MI2_std_null = ", MI2_std_null)

print("MI2_mean_alt  = ", MI2_mean_alt)
print("MI2_std_alt = ", MI2_std_alt)


print("KDEVM_mean_null = ", KDEVM_mean_null)
print("KDEVM_std_null = ", KDEVM_std_null)

print("KDEVM_mean_alt = ", KDEVM_mean_alt)
print("KDEVM_std_alt = ", KDEVM_std_alt)


print("KDEPI_mean_null = ", KDEPI_mean_null)
print("KDEPI_std_null = ", KDEPI_std_null)

print("KDEPI_mean_alt = ", KDEPI_mean_alt)
print("KDEPI_std_alt = ", KDEPI_std_alt)



print("KNNVM_mean_null = ", KNNVM_mean_null)
print("KNNVM_std_null = ", KNNVM_std_null)

print("KNNVM_mean_alt = ", KNNVM_mean_alt)
print("KNNVM_std_alt = ", KNNVM_std_alt)



print("KNNPI_mean_null = ", KNNPI_mean_null)
print("KNNPI_std_null = ", KNNPI_std_null)

print("KNNPI_mean_alt = ", KNNPI_mean_alt)
print("KNNPI_std_alt = ", KNNPI_std_alt)





import matplotlib.pyplot as plt
plt.figure(figsize=(18, 10))
#plt.errorbar(N, MI_mean_null, yerr=MI_std_null, color='blue', label='Ours(beta^2) Null', linewidth=5)
#plt.errorbar(N, MI_mean_alt, yerr=MI_std_alt, color='blue', label='Ours(beta^2) Alt', linestyle='--', dashes=(5, 5), linewidth=5)

plt.errorbar(N, KDEVM_mean_null, yerr=KDEVM_std_null,  color='green', alpha=1.0, label='KDEVM Null', linewidth=4)
plt.errorbar(N, KDEVM_mean_alt, yerr=KDEVM_std_alt,  color='green', alpha=1.0, linestyle='--', dashes=(5, 5), label='KDEVM Alt', linewidth=4)

plt.errorbar(N, KDEPI_mean_null, yerr=KDEPI_std_null,  color='purple', alpha=1.0, label='KDEPI Null', linewidth=4)
plt.errorbar(N, KDEPI_mean_alt, yerr=KDEPI_std_alt,  color='purple', alpha=1.0, linestyle='--', dashes=(5, 5), label='KDEPI Alt', linewidth=4)

plt.errorbar(N, KNNVM_mean_null, yerr=KNNVM_std_null,  color='black', alpha=1.0, label='KNNVM Null', linewidth=4)
plt.errorbar(N, KNNVM_mean_alt, yerr=KNNVM_std_alt,  color='black', alpha=1.0, linestyle='--', dashes=(5, 5), label='KNNVM Alt', linewidth=4)

plt.errorbar(N, KNNPI_mean_null, yerr=KNNPI_std_null,  color='blue', alpha=1.0, label='KNNPI Null', linewidth=4)
plt.errorbar(N, KNNPI_mean_alt, yerr=KNNPI_std_alt,  color='blue', alpha=1.0, linestyle='--', dashes=(5, 5), label='KNNPI Alt', linewidth=4)

plt.errorbar(N, MI2_mean_null, yerr=MI2_std_null, color='red', label='Ours Null', linewidth=5)
plt.errorbar(N, MI2_mean_alt, yerr=MI2_std_alt, color='red', label='Ours Alt', linestyle='--', dashes=(5, 5), linewidth=5)


plt.legend(loc='upper right', fontsize=22)
plt.tick_params(axis='both', which='major', labelsize=25)
plt.xlabel('Sample size', fontsize=30)
plt.ylabel('MI', fontsize=35)
plt.grid(True)
#plt.savefig("gaussian_100_shd_noniid.pdf", format="pdf", bbox_inches="tight")
plt.show()
