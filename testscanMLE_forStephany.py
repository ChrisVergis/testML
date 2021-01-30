import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
np.random.seed(1234)

N=100000
lambda_true=20.1
binning = range(0,61)
lam_min  = 19.0
lam_max  = 21.0
lam_step = 0.01

n_trades = np.random.poisson(lambda_true,N)
plt.hist(n_trades,bins=binning, density=True)
plt.show()
actual_data, bins =   np.histogram(n_trades, bins=binning)
actual_data=actual_data/N

best = []
bestL =-10
lam = lam_min
CHI2 = []
LAMS = []
while lam<=lam_max:
    pois_i = np.zeros(len(binning)-1)
    for k in range(len(binning)-1):
        pois_i[k] = poisson.pmf(k,lam)    
    if len(best)==0: 
        best = np.array(pois_i,'d')
    else:        
        COMPARE = np.sum((pois_i-actual_data)**2) < np.sum((best-actual_data)**2)
        LAMS.append(lam)
        CHI2.append(np.sum((pois_i-actual_data)**2))
        best = np.array(pois_i,'d') if COMPARE else best
        bestL= lam if COMPARE else bestL
        
    lam += lam_step
plt.plot(LAMS,CHI2)
plt.show()

plt.plot(best)
plt.hist(n_trades,bins=bins,density=True)
plt.show()
print("True lambda =",lambda_true)
print("Average = ",n_trades.mean())
print("Best Lambda (Scan):",bestL)

best_Lambdas =[]
for itoy in range(100):
    print(itoy,"/",10)
    n_trades = np.random.poisson(lambda_true,N)
    actual_data, bins =   np.histogram(n_trades, bins=binning)
    actual_data=actual_data/N
    best =[]
    bestL =-10
    lam = lam_min
    while lam<=lam_max:
        pois_i = np.zeros(len(binning)-1)
        for k in range(len(binning)-1):
            pois_i[k] = poisson.pmf(k,lam)    
        if len(best)==0: 
            best = np.array(pois_i,'d')
        else:        
            COMPARE = np.sum((pois_i-actual_data)**2) < np.sum((best-actual_data)**2)
            best = np.array(pois_i,'d') if COMPARE else best
            bestL= lam if COMPARE else bestL
        lam += lam_step
    best_Lambdas.append(bestL)
best_Lambdas=np.array(best_Lambdas)
print(best_Lambdas)
print(best_Lambdas.mean())
plt.hist(best_Lambdas,bins=np.linspace(20,20.25,25))
